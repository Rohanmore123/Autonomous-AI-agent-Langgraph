"""
app/api/v1/endpoints/chat.py
============================
Autonomous Multi-Agent Chat — a Manager Agent coordinates all specialists.

ARCHITECTURE:
  Every message goes to the Manager (LangGraph ReAct loop).
  The Manager autonomously decides which specialist agents to call,
  in what order, and synthesises a single final response.

  User message → Manager → [rag_agent | gmail_agent | task_agent | calendar_agent | direct_llm]
                         → synthesised final answer

PROVIDER SUPPORT:
  Driven entirely by .env — no hardcoded provider.
  PRIMARY_LLM_PROVIDER = openai | anthropic | google

STREAMING:
  POST /chat/stream → SSE events (thinking / tool_result / chunk / done / error)

CONVERSATION HISTORY:
  Last 20 DB messages (~10 turns) injected into every Manager run.
"""

from __future__ import annotations

import asyncio
import json
import operator
import time
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncIterator, Literal, TypedDict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool, ToolException
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.v1.dependencies.auth import CurrentUser
from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.models.models import AgentRun, Conversation, Message
from app.schemas.schemas import (
    APIResponse,
    ChatRequest,
    ChatResponse,
    PaginatedResponse,
    SourceChunk,
)
from app.services.llm.router import llm_router

settings = get_settings()
router = APIRouter(prefix="/chat", tags=["Chat"])
logger = get_logger(__name__)

MAX_MANAGER_ITERATIONS = 12


# ---------------------------------------------------------------------------
# LLM Factory — provider agnostic, driven by .env
# ---------------------------------------------------------------------------

def _build_llm(temperature: float = 0, max_tokens: int = 8192):
    """
    Instantiate the LLM from config.
    Reads PRIMARY_LLM_PROVIDER and PRIMARY_LLM_MODEL from .env.
    Supports: openai | anthropic | google
    """
    provider = settings.llm.primary_provider
    model    = settings.llm.primary_model

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=settings.llm.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=settings.llm.anthropic_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.llm.google_ai_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    raise ValueError(
        f"Unsupported PRIMARY_LLM_PROVIDER='{provider}'. "
        "Must be one of: openai, anthropic, google"
    )


# ---------------------------------------------------------------------------
# Manager State
# ---------------------------------------------------------------------------

class ManagerState(TypedDict):
    messages:        Annotated[list[BaseMessage], operator.add]
    user_id:         str
    iteration_count: int
    final_answer:    str | None
    sources:         list[dict]
    tool_calls_log:  list[dict]
    active_agents:   list[str]
    error:           str | None


# ---------------------------------------------------------------------------
# Manager System Prompt
# ---------------------------------------------------------------------------

MANAGER_SYSTEM_PROMPT = """You are an autonomous AI manager that coordinates specialist agents.

AVAILABLE AGENTS:
1. rag_agent       — Search user's uploaded documents.
2. gmail_agent     — Email operations (search, read, draft, send).
3. task_agent      — Create, list, update, complete tasks and todos.
4. calendar_agent  — View schedule, check availability, create/edit/delete events.
5. direct_llm      — General knowledge, writing, coding, reasoning.

RULES:
- Pick the right agent(s) for each request. Chain them when needed.
- Pass results forward: include prior agent output as context in the next call.
- Synthesise — don't just paste agent outputs. Write one cohesive final answer.
- If one agent fails, continue with the others and note the gap.
- Never fabricate. If agents found nothing, say so.

CONFIRMATION HANDLING (CRITICAL):
- When gmail_agent shows a draft and asks "Shall I send this?" — relay that question
  to the user exactly as-is. Do NOT call the agent again yet.
- When calendar_agent shows an event draft and asks "Shall I create/update/delete this?"
  — relay that question to the user exactly as-is. Do NOT call the agent again yet.

- When the user replies with ANY confirmation phrase such as:
  "yes", "send", "send it", "go ahead", "please send", "confirm", "do it",
  "ok send", "yes send", "create it", "yes create", "schedule it", "delete it",
  "yes delete", "yes update", "update it" — treat it as explicit confirmation.

- On EMAIL confirmation: call gmail_agent again with:
  "User confirmed. Send the email to [recipient] with subject [subject] and body [body]."
  Include the full draft details so the agent calls send_email directly.

- On CALENDAR confirmation: call calendar_agent again with:
  "User confirmed. Create/update/delete the event: [full event details from the draft]."
  Include all event details (title, date, time, duration, attendees) so the agent
  calls the calendar write tool directly.

- NEVER ask for confirmation again once the user has already said yes.
- NEVER say "the draft is ready" again after user confirms — just execute.

CHAINING EXAMPLE:
  "Check if I am free Friday at 3pm then email Alice to schedule a call"
  → Step 1: calendar_agent("Check availability Friday 3pm")
  → Step 2: gmail_agent("Draft email to Alice proposing Friday 3pm [from step 1]")

RESPONSE FORMAT:
Write clear, professional responses. Use markdown where it aids readability."""

# ---------------------------------------------------------------------------
# Specialist Agent Tools
# ---------------------------------------------------------------------------

def _build_tools(user_id: str, db: AsyncSession) -> list:
    """Wrap each specialist agent as a LangChain tool for the Manager."""

    @tool
    async def rag_agent(query: str) -> str:
        """
        Search the user's uploaded document library and return cited answers.
        Use for questions about PDFs, reports, policies, contracts, or any uploaded file.
        Args:
            query: Specific question or search request. Can include doc scope hints.
        Returns:
            Cited answer from matching document chunks.
        """
        try:
            from app.services.agents.rag_agent import RAGAgent
            agent = RAGAgent(user_id=user_id, db=db)
            response, results = await agent.run(query=query, owner_id=user_id)
            sources = ""
            if results:
                sources = "\n\nSources: " + ", ".join(
                    f"{r.filename}[chunk {r.chunk_index}]" for r in results[:5]
                )
            return response.content + sources
        except Exception as e:
            logger.error("tool.rag_agent.error", error=str(e), user_id=user_id)
            raise ToolException(f"RAG agent failed: {e}")

    @tool
    async def gmail_agent(instruction: str) -> str:
        """
        Handle Gmail operations: search, read, summarise, draft, and send emails.
        SAFETY: Will NEVER send without showing a draft and asking confirmation first.
        Relay any "shall I send this?" questions to the user verbatim.
        Args:
            instruction: Natural language email instruction with sender/subject details.
        Returns:
            Email content, inbox summary, draft text, or confirmation request.
        """
        try:
            from app.services.agents.gmail_agent import GmailAgent
            agent = GmailAgent(user_id=user_id, db=db)
            response, _ = await agent.run(instruction)
            return response
        except Exception as e:
            logger.error("tool.gmail_agent.error", error=str(e), user_id=user_id)
            raise ToolException(f"Gmail agent failed: {e}")

    @tool
    async def task_agent(instruction: str) -> str:
        """
        Manage tasks and todos: create, list, update, complete, or delete.
        Handles bulk creation (e.g. from document action items).
        Args:
            instruction: Natural language task instruction, including full list for bulk ops.
        Returns:
            Confirmation with task IDs and details.
        """
        try:
            from app.services.agents.task_agent import TaskAgent
            agent = TaskAgent(user_id=user_id, db=db)
            response, _ = await agent.run(instruction)
            return response
        except Exception as e:
            logger.error("tool.task_agent.error", error=str(e), user_id=user_id)
            raise ToolException(f"Task agent failed: {e}")

    @tool
    async def calendar_agent(instruction: str) -> str:
        """
        Manage Google Calendar: view events, check availability, create/edit/delete events.
        SAFETY: Will NEVER write a calendar change without showing details and confirming first.
        Relay any "shall I create/update/delete this?" to the user verbatim.
        Args:
            instruction: Natural language with date, time, duration, attendees as relevant.
        Returns:
            Schedule, availability, or confirmation of the calendar action.
        """
        try:
            from app.services.agents.calendar_agent import CalendarAgent
            agent = CalendarAgent(user_id=user_id, db=db)
            response, _ = await agent.run(instruction)
            return response
        except Exception as e:
            logger.error("tool.calendar_agent.error", error=str(e), user_id=user_id)
            raise ToolException(f"Calendar agent failed: {e}")

    @tool
    async def direct_llm(prompt: str) -> str:
        """
        Answer using LLM general knowledge — no documents, emails, tasks, or calendar.
        Use for: general questions, writing, coding, math, reasoning, explanations.
        Do NOT use if the user is asking about their personal data — use specialist agents.
        Args:
            prompt: Complete question or task for the LLM.
        Returns:
            Direct LLM response.
        """
        try:
            resp = await llm_router.complete(
                messages=[{"role": "user", "content": prompt}],
                system="You are a helpful, knowledgeable assistant. Answer clearly and accurately.",
                use_cache=True,
            )
            return resp.content
        except Exception as e:
            logger.error("tool.direct_llm.error", error=str(e))
            raise ToolException(f"Direct LLM failed: {e}")

    return [rag_agent, gmail_agent, task_agent, calendar_agent, direct_llm]


# ---------------------------------------------------------------------------
# Manager Graph
# ---------------------------------------------------------------------------

def _build_graph(llm_with_tools, tools: list) -> StateGraph:
    """Compile the Manager ReAct loop."""

    async def call_llm(state: ManagerState) -> dict[str, Any]:
        iteration = state.get("iteration_count", 0) + 1

        logger.info(
            "manager.iteration",
            iteration=iteration,
            messages=len(state["messages"]),
            user_id=state["user_id"],
        )

        if iteration > MAX_MANAGER_ITERATIONS:
            logger.warning("manager.max_iterations", max=MAX_MANAGER_ITERATIONS)
            stop = AIMessage(content=(
                f"I've reached my step limit ({MAX_MANAGER_ITERATIONS} iterations). "
                "Here's what I gathered so far. For complex requests, try breaking them up."
            ))
            return {"messages": [stop], "iteration_count": iteration, "final_answer": stop.content}

        response: AIMessage = await llm_with_tools.ainvoke(state["messages"])

        update: dict[str, Any] = {
            "messages":        [response],
            "iteration_count": iteration,
        }

        if getattr(response, "tool_calls", None):
            active = list(state.get("active_agents", []))
            for tc in response.tool_calls:
                if tc["name"] not in active:
                    active.append(tc["name"])
            update["active_agents"] = active
        else:
            update["final_answer"] = response.content

        return update

    def should_continue(state: ManagerState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    g = StateGraph(ManagerState)
    g.add_node("call_llm", call_llm)
    g.add_node("tools", ToolNode(tools))
    g.set_entry_point("call_llm")
    g.add_conditional_edges("call_llm", should_continue, {"tools": "tools", "end": END})
    g.add_edge("tools", "call_llm")
    return g.compile()


# ---------------------------------------------------------------------------
# Manager Runner (non-streaming)
# ---------------------------------------------------------------------------

async def _run_manager(
    user_id: str,
    db: AsyncSession,
    user_message: str,
    conversation_history: list[BaseMessage] | None = None,
) -> tuple[str, list[dict], list[str], list[dict]]:
    """Run the Manager end-to-end. Returns (answer, tool_calls, active_agents, sources)."""

    tools          = _build_tools(user_id=user_id, db=db)
    llm            = _build_llm(temperature=0, max_tokens=8192)
    llm_with_tools = llm.bind_tools(tools)
    graph          = _build_graph(llm_with_tools, tools)

    messages: list[BaseMessage] = [SystemMessage(content=MANAGER_SYSTEM_PROMPT)]
    if conversation_history:
        messages.extend(conversation_history[-20:])
    messages.append(HumanMessage(content=user_message))

    initial_state: ManagerState = {
        "messages":        messages,
        "user_id":         user_id,
        "iteration_count": 0,
        "final_answer":    None,
        "sources":         [],
        "tool_calls_log":  [],
        "active_agents":   [],
        "error":           None,
    }

    final_state = await graph.ainvoke(initial_state)

    # Extract final answer
    answer = final_state.get("final_answer")
    if not answer:
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break
    answer = answer or "I completed the requested operations."

    # Build tool call log
    tool_calls_log: list[dict] = []
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_calls_log.append({"tool": tc["name"], "input": tc.get("args", {})})
        elif isinstance(msg, ToolMessage) and tool_calls_log:
            tool_calls_log[-1]["result"] = (msg.content or "")[:800]

    logger.info(
        "manager.complete",
        user_id=user_id,
        iterations=final_state.get("iteration_count", 0),
        agents_used=final_state.get("active_agents", []),
        tool_calls=len(tool_calls_log),
    )

    return answer, tool_calls_log, final_state.get("active_agents", []), final_state.get("sources", [])


# ---------------------------------------------------------------------------
# Manager Runner (streaming SSE)
# ---------------------------------------------------------------------------

_AGENT_DISPLAY = {
    "rag_agent":      "Document Research",
    "gmail_agent":    "Gmail",
    "task_agent":     "Task Manager",
    "calendar_agent": "Calendar",
    "direct_llm":     "AI Assistant",
}


async def _stream_manager(
    user_id: str,
    db: AsyncSession,
    user_message: str,
    conversation_history: list[BaseMessage] | None = None,
) -> AsyncIterator[str]:
    """Stream Manager execution as SSE events."""

    tools          = _build_tools(user_id=user_id, db=db)
    llm            = _build_llm(temperature=0, max_tokens=8192)
    llm_with_tools = llm.bind_tools(tools)
    graph          = _build_graph(llm_with_tools, tools)

    messages: list[BaseMessage] = [SystemMessage(content=MANAGER_SYSTEM_PROMPT)]
    if conversation_history:
        messages.extend(conversation_history[-20:])
    messages.append(HumanMessage(content=user_message))

    initial_state: ManagerState = {
        "messages":        messages,
        "user_id":         user_id,
        "iteration_count": 0,
        "final_answer":    None,
        "sources":         [],
        "tool_calls_log":  [],
        "active_agents":   [],
        "error":           None,
    }

    agents_used: list[str] = []

    try:
        async for event in graph.astream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():

                if node_name == "call_llm":
                    for msg in node_output.get("messages", []):
                        if not isinstance(msg, AIMessage):
                            continue
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                name    = tc["name"]
                                display = _AGENT_DISPLAY.get(name, name)
                                if name not in agents_used:
                                    agents_used.append(name)
                                yield _sse({"type": "thinking", "agent": name,
                                            "message": f"Consulting {display}…"})
                        elif msg.content:
                            # Chunk the final synthesis
                            for i in range(0, len(msg.content), 80):
                                yield _sse({"type": "chunk", "content": msg.content[i:i + 80]})
                                await asyncio.sleep(0)

                elif node_name == "tools":
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            name    = getattr(msg, "name", "agent")
                            display = _AGENT_DISPLAY.get(name, name)
                            yield _sse({"type": "tool_result", "agent": name,
                                        "message": f"{display} completed"})

        yield _sse({"type": "done", "agents_used": agents_used})

    except Exception as exc:
        logger.error("manager.stream.error", error=str(exc), user_id=user_id)
        yield _sse({"type": "error", "message": str(exc)})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# DB Helpers
# ---------------------------------------------------------------------------

async def _get_or_create_conversation(
    conversation_id: str | None,
    user_id: str,
    db: AsyncSession,
) -> Conversation:
    if conversation_id:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == user_id)
        )
        conv = result.scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv

    conv = Conversation(user_id=user_id, agent_type="manager", title="New conversation")
    db.add(conv)
    await db.flush()
    return conv


async def _load_history(conversation: Conversation, db: AsyncSession) -> list[BaseMessage]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())
        .limit(20)
    )
    rows = list(reversed(result.scalars().all()))
    out: list[BaseMessage] = []
    for m in rows:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
    return out


async def _save_messages(
    conversation: Conversation,
    user_content: str,
    assistant_content: str,
    tool_calls: list[dict],
    active_agents: list[str],
    latency_ms: float,
    db: AsyncSession,
) -> Message:
    db.add(Message(conversation_id=conversation.id, role="user", content=user_content))

    msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_content,
        model=settings.llm.primary_model,
        provider=settings.llm.primary_provider,
        latency_ms=latency_ms,
        tool_calls=tool_calls,
        sources=[{"agent": a} for a in active_agents],
    )
    db.add(msg)
    conversation.model_used = settings.llm.primary_model
    conversation.agent_type = "manager"
    await db.flush()
    return msg


async def _log_agent_run(
    user_id: str,
    conversation_id: str,
    active_agents: list[str],
    tool_calls: list[dict],
    latency_ms: float,
    success: bool,
    db: AsyncSession,
    error_type: str | None = None,
) -> None:
    db.add(AgentRun(
        user_id=user_id,
        conversation_id=conversation_id,
        agent_type="manager",
        latency_ms=latency_ms,
        provider_used=settings.llm.primary_provider,
        model_used=settings.llm.primary_model,
        success=success,
        error_type=error_type,
        tool_calls_made=tool_calls,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
    ))


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@router.post("", response_model=APIResponse[ChatResponse], summary="Autonomous multi-agent chat")
async def chat(
    body: ChatRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse[ChatResponse]:
    start = time.perf_counter()

    conversation = await _get_or_create_conversation(body.conversation_id, str(current_user.id), db)

    history: list[BaseMessage] = []
    if body.conversation_id:
        history = await _load_history(conversation, db)

    try:
        answer, tool_calls, active_agents, sources = await _run_manager(
            user_id=str(current_user.id),
            db=db,
            user_message=body.message,
            conversation_history=history,
        )
    except Exception as exc:
        logger.error("chat.error", error=str(exc), user_id=str(current_user.id))
        raise HTTPException(status_code=503, detail=f"Manager agent error: {exc}")

    latency_ms = (time.perf_counter() - start) * 1000

    assistant_msg = await _save_messages(
        conversation=conversation,
        user_content=body.message,
        assistant_content=answer,
        tool_calls=tool_calls,
        active_agents=active_agents,
        latency_ms=latency_ms,
        db=db,
    )
    await _log_agent_run(
        user_id=str(current_user.id),
        conversation_id=str(conversation.id),
        active_agents=active_agents,
        tool_calls=tool_calls,
        latency_ms=latency_ms,
        success=True,
        db=db,
    )
    await db.commit()

    logger.info(
        "chat.complete",
        user_id=str(current_user.id),
        latency_ms=round(latency_ms, 1),
        agents_used=active_agents,
        tool_calls=len(tool_calls),
    )

    source_chunks = []
    if body.include_sources:
        for s in sources:
            if isinstance(s, dict) and "document_id" in s:
                source_chunks.append(SourceChunk(
                    document_id=s.get("document_id", ""),
                    chunk_id=s.get("weaviate_id", ""),
                    filename=s.get("filename", ""),
                    content=s.get("content", ""),
                    score=s.get("score", 0.0),
                ))

    return APIResponse(
        data=ChatResponse(
            conversation_id=conversation.id,
            message_id=assistant_msg.id,
            content=answer,
            model_used=settings.llm.primary_model,
            provider_used=settings.llm.primary_provider,
            tokens_used=0,
            latency_ms=latency_ms,
            was_cached=False,
            used_fallback=False,
            sources=source_chunks,
            tool_calls=tool_calls,
            created_at=assistant_msg.created_at,
            active_agents=active_agents,
        )
    )


# ---------------------------------------------------------------------------
# POST /chat/stream
# ---------------------------------------------------------------------------

@router.post("/stream", summary="Streaming autonomous multi-agent response (SSE)")
async def chat_stream(
    body: ChatRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    history: list[BaseMessage] = []
    if body.conversation_id:
        try:
            conversation = await _get_or_create_conversation(
                body.conversation_id, str(current_user.id), db
            )
            history = await _load_history(conversation, db)
        except Exception:
            pass

    async def event_generator():
        try:
            async for chunk in _stream_manager(
                user_id=str(current_user.id),
                db=db,
                user_message=body.message,
                conversation_history=history,
            ):
                yield chunk
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("chat.stream.error", error=str(exc))
            yield _sse({"type": "error", "message": str(exc)})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# GET /conversations
# ---------------------------------------------------------------------------

@router.get("/conversations", response_model=APIResponse[PaginatedResponse],
            summary="List user conversations")
async def list_conversations(
    page: int = 1,
    page_size: int = 20,
    current_user: CurrentUser = ...,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    from sqlalchemy import func

    offset = (page - 1) * page_size
    q = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .where(Conversation.is_archived == False)  # noqa: E712
        .order_by(Conversation.updated_at.desc())
        .offset(offset).limit(page_size)
    )
    total_q = (
        select(func.count()).select_from(Conversation)
        .where(Conversation.user_id == current_user.id)
        .where(Conversation.is_archived == False)  # noqa: E712
    )

    conversations = (await db.execute(q)).scalars().all()
    total         = (await db.execute(total_q)).scalar_one()

    return APIResponse(
        data=PaginatedResponse(
            items=[{
                "id":             c.id,
                "title":          c.title,
                "agent_type":     c.agent_type,
                "model_used":     c.model_used,
                "total_tokens":   c.total_tokens,
                "total_cost_usd": c.total_cost_usd,
                "updated_at":     c.updated_at.isoformat(),
            } for c in conversations],
            total=total,
            page=page,
            page_size=page_size,
            pages=(total + page_size - 1) // page_size,
        )
    )


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}
# ---------------------------------------------------------------------------

@router.get("/conversations/{conversation_id}", summary="Get conversation with full history")
async def get_conversation(
    conversation_id: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == current_user.id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return APIResponse(data={
        "id":             conv.id,
        "title":          conv.title,
        "agent_type":     conv.agent_type,
        "total_tokens":   conv.total_tokens,
        "total_cost_usd": conv.total_cost_usd,
        "messages": [{
            "id":          m.id,
            "role":        m.role,
            "content":     m.content,
            "model":       m.model,
            "tokens_used": m.tokens_used,
            "latency_ms":  m.latency_ms,
            "sources":     m.sources,
            "tool_calls":  m.tool_calls,
            "created_at":  m.created_at.isoformat(),
        } for m in conv.messages],
    })


# ---------------------------------------------------------------------------
# DELETE /conversations/{conversation_id}
# ---------------------------------------------------------------------------

@router.delete("/conversations/{conversation_id}", response_model=APIResponse)
async def delete_conversation(
    conversation_id: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == current_user.id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.is_archived = True
    await db.commit()
    return APIResponse(message="Conversation archived")