"""
app/services/agents/graph.py
==============================
LangGraph ReAct agent graph — shared state machine used by all agents.

LANGGRAPH ReAct PATTERN:
  The agent runs in a loop:
    ┌──────────────────────────────────────────────────────┐
    │                   START                              │
    │                     │                               │
    │               ┌─────▼──────┐                        │
    │               │  call_llm  │  LLM decides: answer   │
    │               │   (node)   │  or call a tool        │
    │               └─────┬──────┘                        │
    │                     │                               │
    │          ┌──────────▼──────────┐                    │
    │          │  should_continue?   │  (conditional edge) │
    │          └──────────┬──────────┘                    │
    │                     │                               │
    │         ┌───────────┴───────────┐                   │
    │         │ tool_call?            │ end?              │
    │         ▼                       ▼                   │
    │   ┌────────────┐          ┌─────────┐               │
    │   │  tool_node │          │   END   │               │
    │   │ (execute   │          └─────────┘               │
    │   │  @tools)   │                                    │
    │   └─────┬──────┘                                    │
    │         │ tool result → back to call_llm            │
    │         └────────────────────────────────────────── │
    └──────────────────────────────────────────────────────┘

AGENT STATE (AgentState TypedDict):
  messages:         Full conversation history (HumanMessage, AIMessage, ToolMessage)
  user_id:          Injected at graph build time — never changes
  agent_type:       Which agent is running (for logging)
  iteration_count:  Safety counter — prevents runaway loops
  final_answer:     Set when the agent finishes (no more tool calls)
  error:            Last error message (if any)

AUTONOMOUS BEHAVIOUR:
  The agent is fully autonomous — it decides on its own:
    • Which tool to call next
    • How many times to retry after a tool error
    • When it has enough information to answer
    • Whether to ask a clarifying question

MAX_ITERATIONS = 10:
  Safety limit. The graph transitions to END if exceeded.
  In practice, most tasks complete in 2-4 iterations.
  Complex multi-step tasks (Gmail → Task) may use 6-8.

STREAMING:
  LangGraph supports streaming node outputs. The chat endpoint
  can yield partial results using graph.astream() instead of graph.ainvoke().
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_ITERATIONS = 10


# ---------------------------------------------------------------------------
# Shared Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The state that flows through every node in the LangGraph.

    messages uses Annotated[list, operator.add] which means:
      When a node returns {"messages": [new_msg]}, LangGraph APPENDS it
      to the existing list — it does NOT replace. This gives us automatic
      message history accumulation without manual list management.
    """
    messages:        Annotated[list[BaseMessage], operator.add]
    user_id:         str
    agent_type:      str
    iteration_count: int
    final_answer:    str | None
    error:           str | None


# ---------------------------------------------------------------------------
# Graph node: call_llm
# ---------------------------------------------------------------------------

def make_call_llm_node(llm_with_tools):
    """
    Create the call_llm node function bound to a specific LLM instance.
    The LLM already has tools bound (llm.bind_tools(tools)).

    Node logic:
      1. Increment iteration counter
      2. Check safety limit
      3. Invoke the LLM with current messages
      4. Append AIMessage to state
    """
    async def call_llm(state: AgentState) -> dict[str, Any]:
        iteration = state.get("iteration_count", 0) + 1

        logger.info(
            "agent.iteration",
            agent=state["agent_type"],
            iteration=iteration,
            messages=len(state["messages"]),
        )

        if iteration > MAX_ITERATIONS:
            # Force stop — append a final message and signal END
            logger.warning(
                "agent.max_iterations",
                agent=state["agent_type"],
                max=MAX_ITERATIONS,
            )
            stop_msg = AIMessage(
                content=(
                    f"I've reached the maximum number of steps ({MAX_ITERATIONS}). "
                    "Here's what I've accomplished so far, though I may not have finished everything. "
                    "Please try breaking your request into smaller parts."
                )
            )
            return {
                "messages":        [stop_msg],
                "iteration_count": iteration,
                "final_answer":    stop_msg.content,
            }

        # Invoke the LLM — it sees all messages and decides: answer or tool
        response: AIMessage = await llm_with_tools.ainvoke(state["messages"])

        logger.info(
            "agent.llm_response",
            agent=state["agent_type"],
            has_tool_calls=bool(getattr(response, "tool_calls", None)),
            content_length=len(response.content or ""),
        )

        update: dict[str, Any] = {
            "messages":        [response],
            "iteration_count": iteration,
        }

        # If no tool calls → this is the final answer
        if not getattr(response, "tool_calls", None):
            update["final_answer"] = response.content

        return update

    return call_llm


# ---------------------------------------------------------------------------
# Conditional edge: should_continue?
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Routing function: after call_llm, decide whether to:
      → "tools"  : execute the tool calls the LLM requested
      → "end"    : the LLM gave a text answer (no tool calls)

    LangGraph uses the return value of this function to pick the next node.
    """
    last_message = state["messages"][-1]

    # AIMessage with tool_calls → run the tools
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"

    # No tool calls OR iteration limit reached → done
    return "end"


# ---------------------------------------------------------------------------
# Tool error handler node
# ---------------------------------------------------------------------------

def make_tool_error_node():
    """
    Wraps ToolNode to handle ToolException gracefully.
    Instead of crashing, failed tools return an error ToolMessage
    so the LLM can self-correct (retry with different args, or explain the issue).
    """
    async def handle_tool_error(state: AgentState) -> dict[str, Any]:
        """Called when ToolNode raises an exception."""
        error = state.get("error", "Unknown tool error")
        last_msg = state["messages"][-1]

        # Build a ToolMessage error response for every failed tool call
        error_messages = []
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                error_messages.append(
                    ToolMessage(
                        content=f"Tool '{tc['name']}' failed with error: {error}. "
                                "Please try a different approach or correct your parameters.",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )

        return {"messages": error_messages}

    return handle_tool_error


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent_graph(llm_with_tools, tools: list) -> StateGraph:
    """
    Construct and compile a LangGraph ReAct agent.

    Args:
        llm_with_tools: LLM with .bind_tools() already called.
        tools:          List of @tool decorated functions for ToolNode.

    Returns:
        Compiled StateGraph ready for .ainvoke() or .astream().

    Graph topology:
        __start__ → call_llm → [tools → call_llm loop] → __end__

    The ToolNode handles:
      - Parallel tool execution (calls all tools in a single AIMessage simultaneously)
      - Automatic async dispatch (calls async tools with await)
      - Error wrapping (ToolException → ToolMessage with error)
    """
    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────
    graph.add_node("call_llm",  make_call_llm_node(llm_with_tools))
    graph.add_node("tools",     ToolNode(tools))

    # ── Entry point ───────────────────────────────────────────────────
    graph.set_entry_point("call_llm")

    # ── Conditional edge after call_llm ───────────────────────────────
    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "tools": "tools",   # → run tool node
            "end":   END,       # → graph is done
        },
    )

    # ── After tools → always go back to call_llm ─────────────────────
    graph.add_edge("tools", "call_llm")

    return graph.compile()


# ---------------------------------------------------------------------------
# Agent runner — unified entry point
# ---------------------------------------------------------------------------

async def run_agent(
    graph,
    user_message: str,
    user_id: str,
    agent_type: str,
    system_prompt: str,
    conversation_history: list[BaseMessage] | None = None,
    config: RunnableConfig | None = None,
) -> tuple[str, list[dict]]:
    """
    Run a compiled LangGraph agent and return the final answer + tool calls made.

    Args:
        graph:                Compiled LangGraph (from build_agent_graph).
        user_message:         The user's current input.
        user_id:              For state injection and logging.
        agent_type:           Label for logging ("task", "rag", "gmail").
        system_prompt:        System instructions for the LLM.
        conversation_history: Prior messages for multi-turn context.
        config:               Optional RunnableConfig (tracing, callbacks).

    Returns:
        (final_answer_text, list_of_tool_calls_made)
    """
    # Build initial messages
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    # Inject conversation history (last 10 turns to stay in context window)
    if conversation_history:
        messages.extend(conversation_history[-10:])

    messages.append(HumanMessage(content=user_message))

    initial_state: AgentState = {
        "messages":        messages,
        "user_id":         user_id,
        "agent_type":      agent_type,
        "iteration_count": 0,
        "final_answer":    None,
        "error":           None,
    }

    # Run the graph
    final_state = await graph.ainvoke(initial_state, config=config)

    # Extract final answer
    final_answer = final_state.get("final_answer")
    if not final_answer:
        # Fallback: get text from last AIMessage
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break
    final_answer = final_answer or "I completed the task but couldn't generate a summary."

    # Extract all tool calls made during this run
    tool_calls = []
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_calls.append({
                    "tool":  tc["name"],
                    "input": tc["args"],
                })
        elif isinstance(msg, ToolMessage):
            # Attach result to the last tool call entry
            if tool_calls:
                tool_calls[-1]["result"] = msg.content[:500]  # Truncate for storage

    logger.info(
        "agent.complete",
        agent=agent_type,
        tool_calls=len(tool_calls),
        answer_length=len(final_answer),
        iterations=final_state.get("iteration_count", 0),
    )

    return final_answer, tool_calls