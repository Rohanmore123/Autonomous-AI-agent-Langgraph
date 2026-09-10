"""
app/services/agents/rag_agent.py  [REWRITTEN — LangGraph + @tool]
==================================================================
Autonomous RAG (Retrieval-Augmented Generation) Agent powered by LangGraph.

WHAT CHANGED:
  BEFORE: Single fixed pipeline — search → build prompt → LLM
  AFTER:  Autonomous agent that decides HOW to search based on the question.
          It can call multiple search tools, refine queries, and combine results.

AUTONOMOUS CAPABILITIES:
  The agent can:
    1. Choose between hybrid, keyword, or semantic search per query type
    2. List available documents before searching (to scope search correctly)
    3. Run multiple searches and combine evidence from different documents
    4. Identify when documents don't have the answer and say so clearly
    5. Ask for clarification if the question is too vague
    6. Search iteratively — if first search returns poor results, try different terms

EXAMPLE MULTI-STEP RAG RUN:
  User: "Compare the security policies in all my uploaded policy documents"
  Iteration 1: LLM → list_available_documents() → sees 3 policy PDFs
  Iteration 2: LLM → hybrid_search("security policy", top_k=10) → 10 chunks
  Iteration 3: LLM → keyword_search("access control", document_ids=[d1,d2,d3])
  Iteration 4: LLM → synthesises answer from all results → END
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.agents.graph import build_agent_graph, run_agent
from app.services.agents.tools.rag_tools import build_rag_tools

settings = get_settings()
logger = get_logger(__name__)

RAG_AGENT_SYSTEM_PROMPT = """You are a precise document research assistant with access to the user's uploaded document library.

Your job is to find accurate, relevant information from the user's documents and answer their questions faithfully.

SEARCH STRATEGY (choose based on query type):
- hybrid_search_documents:   Best for most questions (balanced keyword + semantic)
- keyword_search_documents:  Use for exact terms, codes, IDs, product names
- semantic_search_documents: Use for conceptual questions, paraphrases, related ideas

WORKFLOW:
1. If you don't know what documents are available, call list_available_documents first.
2. Choose the most appropriate search tool(s) for the query type.
3. If the first search returns poor results (low scores or irrelevant content), 
   try different search terms or a different search tool.
4. Synthesise findings from all search results into a clear, cited answer.
5. ALWAYS cite your sources: [Source: filename, chunk N, score: X.XX]

HONESTY RULES:
- If the documents don't contain the answer, say clearly: 
  "The uploaded documents don't contain information about this topic."
- Never fabricate information or use knowledge outside the provided documents.
- If results are partially relevant, share what was found and note the gap.

CITATION FORMAT: [Source: {filename}, chunk {index}]"""


class RAGAgent:
    """
    LangGraph-powered autonomous RAG agent.
    
    Stateless — instantiate fresh per request. The tools are bound
    to the user_id at construction time for data isolation.
    """

    def __init__(self, user_id: str, db) -> None:
        self.user_id = user_id
        self.db = db

        self.tools = build_rag_tools(user_id=user_id, db=db)

        llm = ChatAnthropic(
            model=settings.llm.primary_model,
            api_key=settings.llm.anthropic_api_key,
            temperature=0,
            max_tokens=4096,    # Longer for synthesis of multiple search results
        )
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.graph = build_agent_graph(self.llm_with_tools, self.tools)

    async def run(
        self,
        query: str,
        owner_id: str | None = None,
        top_k: int = 5,
        alpha: float = 0.5,
        document_ids: list[str] | None = None,
        conversation_history: list[BaseMessage] | None = None,
        use_cache: bool = True,
    ) -> tuple[Any, list]:
        """
        Run the autonomous RAG agent.

        The agent autonomously decides how to search and synthesise the answer.

        Args:
            query:                The user's question.
            owner_id:             User ID (uses self.user_id if not provided).
            top_k:                Default top_k hint for tools (tools can override).
            alpha:                Default alpha hint for tools (tools can override).
            document_ids:         Optional document scope restriction.
            conversation_history: Prior conversation for multi-turn.
            use_cache:            Whether LLM responses can be cached.

        Returns:
            (LLMResponse-compatible object, [SearchResult-like dicts])
        """
        # Build query context for the agent
        context_parts = [query]
        if document_ids:
            context_parts.append(f"\n[Search scope: limit to document IDs: {document_ids}]")
        if top_k != 5:
            context_parts.append(f"[Preferred result count: {top_k}]")

        full_query = "\n".join(context_parts)

        final_answer, tool_calls = await run_agent(
            graph=self.graph,
            user_message=full_query,
            user_id=self.user_id,
            agent_type="rag",
            system_prompt=RAG_AGENT_SYSTEM_PROMPT,
            conversation_history=conversation_history,
        )

        # Extract search results from tool calls for the API response
        search_results = []
        for tc in tool_calls:
            if "search" in tc.get("tool", "") and "result" in tc:
                try:
                    import json
                    result_data = json.loads(tc["result"])
                    for r in result_data.get("results", []):
                        search_results.append(_make_search_result(r))
                except Exception:
                    pass

        # Return a duck-typed response compatible with the chat endpoint
        return _LLMResponseCompat(final_answer), search_results


class _LLMResponseCompat:
    """Makes RAGAgent response compatible with the chat endpoint's LLMResponse interface."""
    def __init__(self, content: str):
        self.content = content
        self.provider = settings.llm.primary_provider
        self.model = settings.llm.primary_model
        self.input_tokens = 0     # LangGraph doesn't expose this easily; tracked separately
        self.output_tokens = 0
        self.latency_ms = 0.0
        self.was_cached = False
        self.used_fallback = False
        self.retry_count = 0
        self.cost_usd = 0.0


class _SearchResultCompat:
    """Duck-typed SearchResult for API response compatibility."""
    def __init__(self, doc_id: str, filename: str, content: str, score: float, chunk_index: int):
        self.document_id = doc_id
        self.weaviate_id = ""
        self.filename = filename
        self.content = content
        self.score = score
        self.chunk_index = chunk_index


def _make_search_result(r: dict) -> _SearchResultCompat:
    return _SearchResultCompat(
        doc_id=r.get("document_id", ""),
        filename=r.get("filename", ""),
        content=r.get("content", ""),
        score=r.get("relevance_score", r.get("similarity_score", 0.0)),
        chunk_index=r.get("chunk_index", 0),
    )