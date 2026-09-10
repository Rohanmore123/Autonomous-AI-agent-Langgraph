"""
app/services/agents/router_agent.py  [REWRITTEN — LangGraph Supervisor]
========================================================================
Multi-Agent Supervisor — routes queries to specialist agents and
orchestrates multi-agent workflows using LangGraph.

ARCHITECTURE:
  The supervisor is itself a LangGraph agent whose "tools" are the
  other agents (RAG, Gmail, Task). This is the LangGraph multi-agent
  pattern: agents as tools for a higher-level orchestrator.

  ┌─────────────────────────────────────────────────────────────────┐
  │                    SUPERVISOR AGENT                             │
  │  Decides: which specialist agents to invoke, and in what order  │
  │                                                                 │
  │    ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌────────┐   │
  │    │   RAG    │    │  Gmail  │    │  Task   │   │ Direct │   │
  │    │  Agent   │    │  Agent  │    │  Agent  │   │  LLM   │   │
  │    └──────────┘    └──────────┘    └──────────┘   └────────┘   │
  └─────────────────────────────────────────────────────────────────┘

ROUTING STRATEGY (two-stage, same as before):
  1. Rule-based fast path (~1ms) — no LLM cost for obvious queries
  2. LLM supervisor for ambiguous or multi-agent queries

MULTI-AGENT WORKFLOWS:
  "Summarise my inbox and create tasks for action items"
  → Supervisor calls gmail_agent_tool("summarise inbox")
  → Supervisor calls task_agent_tool("create task: [item 1]")
  → Supervisor calls task_agent_tool("create task: [item 2]")
  → Supervisor synthesises final response

SINGLE-AGENT QUERIES:
  "What does my uploaded report say about revenue?"
  → Fast-path routes to RAG agent directly (no supervisor overhead)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool, ToolException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.agents.graph import build_agent_graph, run_agent

settings = get_settings()
logger = get_logger(__name__)


class AgentType(str, Enum):
    RAG    = "rag"
    GMAIL  = "gmail"
    TASK   = "task"
    DIRECT = "direct"
    ROUTER = "router"


# ---------------------------------------------------------------------------
# Rule-based fast-path patterns (unchanged — still zero-cost routing)
# ---------------------------------------------------------------------------

_GMAIL_PATTERNS = re.compile(
    r"\b(email|gmail|inbox|send|reply|forward|unread|message|mail|compose|draft)\b",
    re.IGNORECASE,
)
_TASK_PATTERNS = re.compile(
    r"\b(task|todo|reminder|deadline|due|schedule|priorit|checklist|"
    r"create a task|add task|complete task|mark done|finish task)\b",
    re.IGNORECASE,
)
_RAG_PATTERNS = re.compile(
    r"\b(document|pdf|file|knowledge base|according to|find in|search docs|"
    r"what does the|based on the|from the|summarize the document)\b",
    re.IGNORECASE,
)


def _rule_based_route(query: str) -> AgentType | None:
    """Fast keyword routing. Returns None if ambiguous → LLM supervisor."""
    hits = [
        bool(_GMAIL_PATTERNS.search(query)),
        bool(_TASK_PATTERNS.search(query)),
        bool(_RAG_PATTERNS.search(query)),
    ]
    total = sum(hits)
    if total == 0:
        return AgentType.DIRECT
    if total == 1:
        if hits[0]: return AgentType.GMAIL
        if hits[1]: return AgentType.TASK
        if hits[2]: return AgentType.RAG
    return None   # Ambiguous → supervisor


async def route_query(
    query: str,
    explicit_agent: str | None = None,
) -> AgentType:
    """Determine which single agent should handle this query."""
    if explicit_agent:
        try:
            return AgentType(explicit_agent.lower())
        except ValueError:
            pass

    rule = _rule_based_route(query)
    if rule is not None:
        logger.info("router.rule_based", agent=rule, query=query[:60])
        return rule

    # LLM classification for ambiguous single queries
    from app.services.llm.router import llm_router
    resp = await llm_router.complete(
        messages=[{"role": "user", "content": query}],
        system=(
            "Classify this query into one word: rag, gmail, task, or direct. "
            "rag=document questions, gmail=email, task=todos, direct=general. "
            "Reply with ONLY the one word."
        ),
        max_tokens=10,
        use_cache=True,
    )
    label = resp.content.strip().lower()
    try:
        result = AgentType(label)
    except ValueError:
        result = AgentType.DIRECT
    logger.info("router.llm_classified", agent=result, query=query[:60])
    return result


async def plan_multi_agent(query: str) -> list[dict]:
    """Check if a query needs multiple agents. Returns a plan list."""
    hits = sum([
        bool(_GMAIL_PATTERNS.search(query)),
        bool(_TASK_PATTERNS.search(query)),
        bool(_RAG_PATTERNS.search(query)),
    ])
    if hits < 2:
        agent = await route_query(query)
        return [{"agent": agent, "sub_query": query, "depends_on": None}]

    from app.services.llm.router import llm_router
    import json
    resp = await llm_router.complete(
        messages=[{"role": "user", "content": f"Decompose: {query}"}],
        system=(
            "Decompose into sub-tasks for agents: rag, gmail, task, direct. "
            'Return JSON: [{"agent":"...","sub_query":"...","depends_on":null}]. '
            "JSON only, no markdown."
        ),
        max_tokens=300,
        use_cache=False,
    )
    try:
        return json.loads(resp.content)
    except Exception:
        agent = await route_query(query)
        return [{"agent": agent, "sub_query": query, "depends_on": None}]


# ---------------------------------------------------------------------------
# Supervisor Agent (LangGraph) — used for multi-agent queries
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """You are a multi-agent orchestration supervisor. You coordinate specialist agents to handle complex user requests.

AVAILABLE AGENTS (call these tools to delegate work):
- rag_agent:   Searches and answers questions from the user's uploaded documents
- gmail_agent: Manages Gmail — search, read, summarise, draft, send emails
- task_agent:  Manages tasks — create, list, update, complete, delete todos

ORCHESTRATION RULES:
1. For simple requests, delegate to ONE specialist agent.
2. For complex multi-step requests, chain agents in logical order.
3. Wait for each agent's result before calling the next if there's a dependency.
4. Synthesise all agent responses into one coherent final answer.
5. If an agent fails, try an alternative approach or explain what happened.

EXAMPLE CHAINING:
  "Summarise my inbox and create tasks for action items":
    Step 1: gmail_agent("Summarise my inbox for action items")
    Step 2: task_agent("Create tasks: [list from step 1]")
    Step 3: Final summary of what was done

Always be clear about which agent handled which part of the request."""


class SupervisorAgent:
    """
    LangGraph Supervisor — orchestrates multiple specialist agents.
    Used when a query spans multiple agent domains.
    
    The supervisor's "tools" are the other agents themselves.
    This is the LangGraph multi-agent pattern.
    """

    def __init__(self, user_id: str, db: AsyncSession) -> None:
        self.user_id = user_id
        self.db = db
        self._graph = None
        self._tools = None

    def _build_agent_tools(self) -> list:
        """Build @tool decorated wrappers around each specialist agent."""

        user_id = self.user_id
        db = self.db

        @tool
        async def rag_agent(query: str) -> str:
            """
            Search and answer questions from the user's uploaded documents.
            Use this for any question about document content, knowledge base queries,
            or when the user asks about information in their uploaded files.

            Args:
                query: The question or search query about document contents.

            Returns:
                Answer synthesised from document search results with source citations.
            """
            from app.services.agents.rag_agent import RAGAgent
            agent = RAGAgent(user_id=user_id, db=db)
            response, _ = await agent.run(query=query, owner_id=user_id)
            return response.content

        @tool
        async def gmail_agent(instruction: str) -> str:
            """
            Handle Gmail operations — search, read, summarise, draft, and send emails.
            Use this for any email-related task.

            Args:
                instruction: Natural language email instruction (e.g., "Summarise unread emails",
                             "Draft a reply to the latest email from John about the project").

            Returns:
                Result of the email operation with details.
            """
            from app.services.agents.gmail_agent import GmailAgent
            agent = GmailAgent(user_id=user_id, db=db)
            response, _ = await agent.run(instruction)
            return response

        @tool
        async def task_agent(instruction: str) -> str:
            """
            Manage tasks — create, list, update, complete, or delete todos.
            Use this for any task management request.

            Args:
                instruction: Natural language task instruction (e.g., "Create a high priority
                             task to review the proposal by Friday", "List all urgent tasks").

            Returns:
                Result of the task operation with confirmation details.
            """
            from app.services.agents.task_agent import TaskAgent
            agent = TaskAgent(user_id=user_id, db=db)
            response, _ = await agent.run(instruction)
            return response

        return [rag_agent, gmail_agent, task_agent]

    def _get_graph(self):
        if self._graph is None:
            self._tools = self._build_agent_tools()
            llm = ChatAnthropic(
                model=settings.llm.primary_model,
                api_key=settings.llm.anthropic_api_key,
                temperature=0,
                max_tokens=4096,
            )
            self._graph = build_agent_graph(llm.bind_tools(self._tools), self._tools)
        return self._graph

    async def run(
        self,
        query: str,
        conversation_history: list[BaseMessage] | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Orchestrate multi-agent response.

        Args:
            query:                Complex multi-agent request.
            conversation_history: Prior conversation context.

        Returns:
            (synthesised_answer, all_tool_calls_made)
        """
        graph = self._get_graph()
        return await run_agent(
            graph=graph,
            user_message=query,
            user_id=self.user_id,
            agent_type="supervisor",
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            conversation_history=conversation_history,
        )