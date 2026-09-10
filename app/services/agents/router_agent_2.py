"""
app/services/agents/router_agent.py
=====================================
Agent Router — the meta-agent that decides which specialist agent handles a request.

ROUTING STRATEGY:
  1. Rule-based fast path (keyword/regex matching) — zero LLM cost, ~1ms
  2. LLM classification fallback — for ambiguous queries

AGENTS AVAILABLE:
  rag    — answer questions from the user's documents
  gmail  — email management (read, search, send, summarise)
  task   — task/todo management (create, list, complete)
  direct — general conversation (no tool, no RAG)

WHY TWO-STAGE ROUTING?
  LLM classification costs tokens and adds latency.
  Most routing decisions are obvious from keywords (e.g., "send email to..." → gmail).
  We only invoke LLM classification for genuinely ambiguous queries.
  This reduces routing cost by ~80% in practice.

MULTI-AGENT ORCHESTRATION:
  The router can also decompose a single user request into sub-tasks across
  multiple agents and merge the results. Example:
    "Summarise my inbox and create a task for any email needing a reply"
    → gmail.summarise_inbox() + task_agent.create_task()
"""

from __future__ import annotations

import re
from enum import Enum

from app.core.logging import get_logger
from app.services.llm.router import llm_router

logger = get_logger(__name__)


class AgentType(str, Enum):
    RAG    = "rag"
    GMAIL  = "gmail"
    TASK   = "task"
    DIRECT = "direct"


# ---------------------------------------------------------------------------
# Rule-based patterns (ordered by specificity)
# ---------------------------------------------------------------------------

_GMAIL_PATTERNS = re.compile(
    r"\b(emails?|gmail|inbox|send|reply|forward|unread|messages?|mail|compose|draft)\b",
    re.IGNORECASE,
)

_TASK_PATTERNS = re.compile(
    r"\b(tasks?|todos?|reminder|deadline|due|schedule|priorit|checklist)\b",
    re.IGNORECASE,
)

_RAG_PATTERNS = re.compile(
    r"\b(documents?|pdf|file|knowledge base|according to|find in|search|"
    r"what does the|based on the|from the|summarize)\b",
    re.IGNORECASE,
)


def _rule_based_route(query: str) -> AgentType | None:
    """
    Fast keyword-based routing. Returns None if ambiguous (→ LLM classification).
    """
    gmail_hit = bool(_GMAIL_PATTERNS.search(query))
    task_hit  = bool(_TASK_PATTERNS.search(query))
    rag_hit   = bool(_RAG_PATTERNS.search(query))

    # Unambiguous single match
    hits = sum([gmail_hit, task_hit, rag_hit])
    if hits == 1:
        if gmail_hit:
            return AgentType.GMAIL
        if task_hit:
            return AgentType.TASK
        if rag_hit:
            return AgentType.RAG

    # No match → general conversation
    if hits == 0:
        return AgentType.DIRECT

    # Multiple matches → ambiguous → defer to LLM
    return None


async def _llm_classify(query: str) -> AgentType:
    """
    Ask the LLM to classify the query into one of the four agent types.
    Returns a single enum value.
    """
    system = """You are a routing classifier. Given a user query, respond with EXACTLY one word:
- "rag"    → if the query is about searching or asking questions from documents/knowledge base
- "gmail"  → if the query involves emails, inbox, composing or reading messages
- "task"   → if the query involves creating, updating, or managing tasks/todos
- "direct" → if the query is general conversation or doesn't fit the above

Respond with ONLY the single word. No explanation."""

    response = await llm_router.complete(
        messages=[{"role": "user", "content": query}],
        system=system,
        max_tokens=10,
        use_cache=True,
    )

    label = response.content.strip().lower()
    try:
        return AgentType(label)
    except ValueError:
        logger.warning("router.llm_classify.unknown_label", label=label)
        return AgentType.DIRECT


# ---------------------------------------------------------------------------
# Public router
# ---------------------------------------------------------------------------

async def route_query(
    query: str,
    explicit_agent: str | None = None,
) -> AgentType:
    """
    Determine which agent should handle this query.

    Args:
        query:          The user's input text.
        explicit_agent: If the caller already knows (e.g., from API param), trust it.

    Returns:
        AgentType enum value.
    """
    # Explicit override from API request (e.g., user selected agent in UI)
    if explicit_agent:
        try:
            agent = AgentType(explicit_agent.lower())
            logger.info("router.explicit", agent=agent)
            return agent
        except ValueError:
            pass  # Fall through to auto-routing

    # Rule-based fast path
    rule_result = _rule_based_route(query)
    if rule_result is not None:
        logger.info("router.rule_based", agent=rule_result, query=query[:60])
        return rule_result

    # LLM classification for ambiguous queries
    llm_result = await _llm_classify(query)
    logger.info("router.llm_classified", agent=llm_result, query=query[:60])
    return llm_result


# ---------------------------------------------------------------------------
# Multi-agent orchestration plan
# ---------------------------------------------------------------------------

async def plan_multi_agent(query: str) -> list[dict]:
    """
    Decompose a complex query into an ordered list of agent sub-tasks.
    Returns a list of: {"agent": AgentType, "sub_query": str, "depends_on": int | None}

    Used for queries like:
      "Summarise my inbox then create tasks for any action items"
      → [{agent: gmail, sub_query: "summarise inbox"}, {agent: task, depends_on: 0}]

    Only called when the router detects multiple agent keywords in one query.
    """
    multi_hit = sum([
        bool(_GMAIL_PATTERNS.search(query)),
        bool(_TASK_PATTERNS.search(query)),
        bool(_RAG_PATTERNS.search(query)),
    ])

    if multi_hit < 2:
        # Not a multi-agent query
        agent = await route_query(query)
        return [{"agent": agent, "sub_query": query, "depends_on": None}]

    system = """You decompose user queries into sub-tasks for different agents.
Available agents: rag, gmail, task, direct.
Return a JSON array like:
[
  {"agent": "gmail", "sub_query": "summarise unread emails", "depends_on": null},
  {"agent": "task", "sub_query": "create tasks from action items", "depends_on": 0}
]
Return ONLY valid JSON. No markdown, no explanation."""

    response = await llm_router.complete(
        messages=[{"role": "user", "content": f"Decompose this query: {query}"}],
        system=system,
        max_tokens=300,
        use_cache=False,
    )

    import json
    try:
        plan = json.loads(response.content)
        return plan
    except Exception:
        logger.warning("router.plan.parse_failed", content=response.content[:200])
        agent = await route_query(query)
        return [{"agent": agent, "sub_query": query, "depends_on": None}]