"""
app/services/agents/rag_agent.py
=================================
Retrieval-Augmented Generation (RAG) Agent.

PIPELINE:
  1. Hybrid search Weaviate for top-K relevant chunks
  2. Re-rank chunks by score (already sorted by Weaviate)
  3. Build grounded prompt: system prompt + context chunks + user question
  4. Call LLMRouter.complete()
  5. Return answer + source citations

GROUNDING PROMPT DESIGN:
  The system prompt instructs the model to:
    a) Only use information from the provided context
    b) Cite which document chunks it used
    c) Say "I don't know" if context is insufficient (reduces hallucination)
  This is called "faithful RAG" — the model is anchored to retrieved facts.

CONTEXT WINDOW MANAGEMENT:
  We cap total context at MAX_CONTEXT_CHARS.
  If top-K chunks exceed this, we trim the lowest-scored ones.
  This prevents context window overflow which would cause LLM errors.
"""

from __future__ import annotations

import time

from app.core.logging import get_logger
from app.services.llm.router import LLMResponse
from app.services.llm import router as llm_router_module
from app.services.vector_db.weaviate_client import SearchResult, hybrid_search

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = 12_000   # ~3K tokens for context, leaves room for output


RAG_SYSTEM_PROMPT = """You are a precise, factual assistant. Answer the user's question
using ONLY the information provided in the CONTEXT DOCUMENTS below.

Rules:
1. Base your answer exclusively on the provided context.
2. If the context does not contain enough information, say: "I don't have enough information in the provided documents to answer this."
3. After your answer, list the document sources you used as: [Source: <filename>, chunk <index>].
4. Do not fabricate, infer beyond the context, or use prior knowledge.
5. Be concise and precise.

CONTEXT DOCUMENTS:
{context}
"""


class RAGAgent:
    """
    RAG Agent: retrieves → augments → generates.
    Instantiated once per request (stateless).
    """

    async def run(
        self,
        query: str,
        owner_id: str,
        top_k: int = 5,
        alpha: float = 0.5,
        document_ids: list[str] | None = None,
        conversation_history: list[dict] | None = None,
        use_cache: bool = True,
    ) -> tuple[LLMResponse, list[SearchResult]]:
        """
        Execute the RAG pipeline.

        Returns:
          (LLMResponse, list[SearchResult]) — the answer and its source chunks.
        """
        start = time.perf_counter()

        # --- Step 1: Retrieve ---
        chunks = await hybrid_search(
            query=query,
            owner_id=owner_id,
            top_k=top_k,
            alpha=alpha,
            document_ids=document_ids,
        )

        logger.info("rag.retrieved", chunks=len(chunks), query=query[:60])

        # --- Step 2: Trim context to fit window ---
        context_parts: list[str] = []
        total_chars = 0
        used_chunks: list[SearchResult] = []

        for chunk in chunks:
            snippet = (
                f"[Document: {chunk.filename}, Chunk {chunk.chunk_index}]\n"
                f"{chunk.content}\n"
                f"(Relevance score: {chunk.score:.3f})"
            )
            if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(snippet)
            total_chars += len(snippet)
            used_chunks.append(chunk)

        context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        # --- Step 3: Build messages ---
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)

        messages: list[dict] = []

        # Include conversation history (last 10 turns to stay within context window)
        if conversation_history:
            messages.extend(conversation_history[-10:])

        messages.append({"role": "user", "content": query})

        # --- Step 4: Generate ---
        llm_response = await llm_router_module.llm_router.complete(
          messages,
          system_prompt,
            max_tokens=1024,
            use_cache=use_cache,
        )

        total_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "rag.complete",
            latency_ms=round(total_ms, 2),
            chunks_used=len(used_chunks),
            tokens=llm_response.input_tokens + llm_response.output_tokens,
        )

        return llm_response, used_chunks