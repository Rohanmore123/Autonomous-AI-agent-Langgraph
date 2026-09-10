"""
app/services/agents/tools/__init__.py
======================================
Centralised @tool registry for all LangChain tools used by every agent.

DESIGN DECISIONS:
  • Every tool is decorated with @tool (langchain_core.tools)
    This gives us:
      - Automatic JSON schema generation from Python type hints + docstring
      - First-class integration with LangGraph ToolNode
      - Unified error handling via ToolException
      - Free observability hooks (LangSmith tracing if configured)

  • Tools are PURE FUNCTIONS with injected state via closures or context vars.
    DB sessions and user_id are injected at agent-construction time, not at
    call time, so the @tool signature stays clean (no hidden dependencies).

  • ToolResult dataclass gives a typed response envelope that every tool
    returns, making result parsing in the agent graph predictable.

TOOL CATEGORIES:
  task_tools.py   — create, list, update, complete, delete tasks (PostgreSQL)
  rag_tools.py    — hybrid_search, get_document_info (Weaviate + PostgreSQL)
  gmail_tools.py  — search_emails, read_email, send_email, summarise_inbox (Gmail API)
  web_tools.py    — web_search (for the direct/general agent)
"""

from app.services.agents.tools.task_tools import build_task_tools
from app.services.agents.tools.rag_tools import build_rag_tools
from app.services.agents.tools.gmail_tools import build_gmail_tools

__all__ = [
    "build_task_tools",
    "build_rag_tools",
    "build_gmail_tools",
]