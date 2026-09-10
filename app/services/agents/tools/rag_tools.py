"""
app/services/agents/tools/rag_tools.py
========================================
LangChain @tool decorated RAG (Retrieval-Augmented Generation) tools.

TOOLS:
  hybrid_search_documents  — Weaviate BM25+vector hybrid search
  get_document_list        — List user's documents from PostgreSQL
  get_document_info        — Get details about a specific document
  semantic_search          — Pure vector similarity search (no BM25)
  keyword_search           — Pure BM25 keyword search (no vector)

WHY MULTIPLE SEARCH TOOLS?
  The LLM can pick the right tool for the right query type:
  - "Find documents mentioning invoice #12345" → keyword_search (exact match)
  - "Find documents about machine learning concepts" → semantic_search (paraphrase)
  - "Find documents about Q3 sales performance" → hybrid_search (best of both)
  Giving the agent these options makes it smarter than a fixed alpha value.

OWNERSHIP SAFETY:
  Every tool injects owner_id into Weaviate filter.
  Users CANNOT access other users' documents regardless of what they ask.
  This is enforced at the vector DB query level, not just application logic.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool, ToolException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.models import Document
from app.services.vector_db.weaviate_client import hybrid_search

logger = get_logger(__name__)


def build_rag_tools(user_id: str, db: AsyncSession) -> list:
    """
    Factory: create RAG tools bound to a specific user_id and DB session.
    Weaviate search is always filtered by owner_id for data isolation.
    """

    # ------------------------------------------------------------------
    # Tool 1: hybrid_search_documents
    # ------------------------------------------------------------------

    @tool
    async def hybrid_search_documents(
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        document_ids: Optional[list[str]] = None,
    ) -> str:
        """
        Search the user's documents using hybrid search (BM25 keywords + dense vector similarity).
        This is the recommended search method for most questions.

        Args:
            query:        Natural language question or search terms.
            top_k:        Number of results to return (1-20, default 5).
            alpha:        Search blend. 0.0=keyword only, 1.0=semantic only, 0.5=balanced.
                          Use lower alpha for exact terms, higher for concepts/paraphrases.
            document_ids: Optional list of document IDs to restrict search scope.

        Returns:
            JSON string with ranked search results including content and relevance scores.
        """
        top_k = max(1, min(20, top_k))
        alpha = max(0.0, min(1.0, alpha))

        if not query.strip():
            raise ToolException("Search query cannot be empty.")

        try:
            results = await hybrid_search(
                query=query,
                owner_id=user_id,
                top_k=top_k,
                alpha=alpha,
                document_ids=document_ids,
            )
        except Exception as e:
            raise ToolException(f"Search failed: {str(e)}")

        if not results:
            return json.dumps({
                "found": 0,
                "results": [],
                "message": "No relevant documents found for this query. "
                           "Make sure documents are uploaded and fully ingested.",
            })

        return json.dumps({
            "found": len(results),
            "query": query,
            "search_type": "hybrid",
            "alpha": alpha,
            "results": [
                {
                    "rank":          i + 1,
                    "document_id":   r.document_id,
                    "filename":      r.filename,
                    "chunk_index":   r.chunk_index,
                    "relevance_score": round(r.score, 4),
                    "content":       r.content,
                }
                for i, r in enumerate(results)
            ],
        })

    # ------------------------------------------------------------------
    # Tool 2: keyword_search_documents
    # ------------------------------------------------------------------

    @tool
    async def keyword_search_documents(
        query: str,
        top_k: int = 5,
        document_ids: Optional[list[str]] = None,
    ) -> str:
        """
        Search documents using exact keyword matching (BM25). Best for:
        - Specific terms, codes, names, or IDs (e.g. "invoice INV-2024-001")
        - Technical jargon that must appear verbatim
        - Product names, error codes, identifiers

        Args:
            query:        Keywords to search for. Supports BM25 scoring.
            top_k:        Number of results (1-20, default 5).
            document_ids: Restrict to specific document IDs.

        Returns:
            JSON string with keyword-matched results.
        """
        top_k = max(1, min(20, top_k))

        results = await hybrid_search(
            query=query,
            owner_id=user_id,
            top_k=top_k,
            alpha=0.0,       # Pure BM25, no vector
            document_ids=document_ids,
        )

        return json.dumps({
            "found":       len(results),
            "query":       query,
            "search_type": "keyword_bm25",
            "results": [
                {
                    "rank":          i + 1,
                    "document_id":   r.document_id,
                    "filename":      r.filename,
                    "chunk_index":   r.chunk_index,
                    "relevance_score": round(r.score, 4),
                    "content":       r.content,
                }
                for i, r in enumerate(results)
            ],
        })

    # ------------------------------------------------------------------
    # Tool 3: semantic_search_documents
    # ------------------------------------------------------------------

    @tool
    async def semantic_search_documents(
        query: str,
        top_k: int = 5,
        document_ids: Optional[list[str]] = None,
    ) -> str:
        """
        Search documents using semantic similarity (dense vectors only). Best for:
        - Conceptual questions ("what does the document say about performance?")
        - Paraphrases or synonyms of terms in the document
        - Cross-language matches (if using multilingual embedding model)
        - When you want meaning-based matching, not keyword matching

        Args:
            query:        Natural language question describing the concept to find.
            top_k:        Number of results (1-20, default 5).
            document_ids: Restrict to specific document IDs.

        Returns:
            JSON string with semantically similar results.
        """
        top_k = max(1, min(20, top_k))

        results = await hybrid_search(
            query=query,
            owner_id=user_id,
            top_k=top_k,
            alpha=1.0,       # Pure vector, no BM25
            document_ids=document_ids,
        )

        return json.dumps({
            "found":       len(results),
            "query":       query,
            "search_type": "semantic_vector",
            "results": [
                {
                    "rank":            i + 1,
                    "document_id":     r.document_id,
                    "filename":        r.filename,
                    "chunk_index":     r.chunk_index,
                    "similarity_score": round(r.score, 4),
                    "content":         r.content,
                }
                for i, r in enumerate(results)
            ],
        })

    # ------------------------------------------------------------------
    # Tool 4: list_available_documents
    # ------------------------------------------------------------------

    @tool
    async def list_available_documents(
        status: str = "ready",
        limit: int = 20,
    ) -> str:
        """
        List the user's uploaded documents. Use this to discover what documents
        are available before searching, or to check ingestion status.

        Args:
            status: Filter by ingestion status. One of: ready, pending, failed, all.
                    Use 'ready' (default) to see searchable documents only.
            limit:  Max documents to return (1-100, default 20).

        Returns:
            JSON string listing document metadata (id, filename, status, chunk_count).
        """
        limit = max(1, min(100, limit))

        q = (
            select(Document)
            .where(Document.owner_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        if status != "all":
            q = q.where(Document.status == status)

        result = await db.execute(q)
        docs = result.scalars().all()

        return json.dumps({
            "total":  len(docs),
            "status_filter": status,
            "documents": [
                {
                    "document_id":  d.id,
                    "filename":     d.filename,
                    "content_type": d.content_type,
                    "size_kb":      round(d.size_bytes / 1024, 1),
                    "chunk_count":  d.chunk_count,
                    "status":       d.status,
                    "uploaded_at":  d.created_at.isoformat(),
                }
                for d in docs
            ],
        })

    # ------------------------------------------------------------------
    # Tool 5: get_document_info
    # ------------------------------------------------------------------

    @tool
    async def get_document_info(document_id: str) -> str:
        """
        Get detailed information about a specific document.

        Args:
            document_id: The UUID of the document.

        Returns:
            JSON string with full document metadata and ingestion status.
        """
        result = await db.execute(
            select(Document)
            .where(Document.id == document_id, Document.owner_id == user_id)
        )
        doc = result.scalar_one_or_none()

        if not doc:
            raise ToolException(
                f"Document '{document_id}' not found. "
                "Use list_available_documents to see your documents."
            )

        return json.dumps({
            "document_id":   doc.id,
            "filename":      doc.filename,
            "content_type":  doc.content_type,
            "size_kb":       round(doc.size_bytes / 1024, 1),
            "chunk_count":   doc.chunk_count,
            "status":        doc.status,
            "error_message": doc.error_message,
            "uploaded_at":   doc.created_at.isoformat(),
            "note": (
                "Document is ready to search."
                if doc.status == "ready"
                else f"Document is not yet searchable (status: {doc.status})."
            ),
        })

    return [
        hybrid_search_documents,
        keyword_search_documents,
        semantic_search_documents,
        list_available_documents,
        get_document_info,
    ]