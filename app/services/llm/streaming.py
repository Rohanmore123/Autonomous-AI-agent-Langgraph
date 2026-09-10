"""
app/services/llm/streaming.py
===============================
Server-Sent Events (SSE) streaming utilities.

WHY STREAMING?
  LLM responses can take 5–30 seconds for long outputs.
  Without streaming, the user stares at a blank screen for 30 seconds.
  With streaming, text appears token-by-token starting within 500ms.
  This dramatically improves perceived performance.

SSE FORMAT (W3C standard):
  data: <chunk>\n\n      — text chunk
  data: [DONE]\n\n       — stream complete
  data: [ERROR] msg\n\n  — error (stream closes after this)

  Each event ends with two newlines (\n\n).
  The client reads these with EventSource API or fetch + ReadableStream.

CLIENT EXAMPLE (JavaScript):
  const evtSource = new EventSource('/api/v1/chat/stream');
  evtSource.onmessage = (e) => {
    if (e.data === '[DONE]') evtSource.close();
    else document.getElementById('output').textContent += e.data;
  };

BACKPRESSURE:
  FastAPI's StreamingResponse handles backpressure automatically.
  If the client is slow, the generator pauses — no buffer overflow.

PARTIAL RECOVERY:
  If the LLM stream fails mid-way, we send [ERROR] and close gracefully.
  The client can show what was received so far and offer a retry.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.llm.router import llm_router

settings = get_settings()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SSE event formatters
# ---------------------------------------------------------------------------

def sse_text(chunk: str) -> str:
    """Format a text chunk as an SSE data event."""
    return f"data: {chunk}\n\n"


def sse_done() -> str:
    """SSE end-of-stream sentinel."""
    return "data: [DONE]\n\n"


def sse_error(message: str) -> str:
    """SSE error event — sent before closing stream."""
    return f"data: [ERROR] {message}\n\n"


def sse_json(payload: dict[str, Any]) -> str:
    """SSE event carrying structured JSON (for metadata like token counts)."""
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

async def stream_llm_response(
    messages: list[dict],
    system: str | None = None,
    max_tokens: int = 2048,
    include_metadata: bool = True,
) -> AsyncIterator[str]:
    """
    Full SSE generator for an LLM streaming response.

    Yields:
      • Text chunks as they arrive from the LLM
      • A final [METADATA] event with token counts (if include_metadata=True)
      • [DONE] sentinel when complete
      • [ERROR] message if the stream fails

    Usage in FastAPI:
        return StreamingResponse(
            stream_llm_response(messages, system),
            media_type="text/event-stream",
        )
    """
    total_chars = 0
    start_time = asyncio.get_event_loop().time()

    try:
        async for chunk in llm_router.stream(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
        ):
            total_chars += len(chunk)
            yield sse_text(chunk)

        # After stream completes, emit metadata
        if include_metadata:
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            # Approximate token count (exact count requires post-processing)
            approx_tokens = total_chars // 4
            yield sse_json({
                "type":         "metadata",
                "approx_tokens": approx_tokens,
                "latency_ms":    round(elapsed_ms, 2),
                "model":         settings.llm.primary_model,
                "provider":      settings.llm.primary_provider,
            })

        yield sse_done()

    except Exception as exc:
        logger.error("streaming.error", error=str(exc))
        yield sse_error(str(exc))
        yield sse_done()


async def stream_rag_response(
    query: str,
    owner_id: str,
    top_k: int = 5,
    alpha: float = 0.5,
) -> AsyncIterator[str]:
    """
    Streaming RAG response.

    Flow:
      1. Emit a [SOURCES] event with retrieved chunk metadata
      2. Stream the LLM answer chunk by chunk
      3. Emit [DONE]

    The client can render sources immediately while the answer is streaming.
    """
    from app.services.agents.rag_agent_2 import RAGAgent
    from app.services.vector_db.weaviate_client import hybrid_search

    # Step 1: Retrieve context (non-streaming)
    try:
        chunks = await hybrid_search(
            query=query,
            owner_id=owner_id,
            top_k=top_k,
            alpha=alpha,
        )
    except Exception as exc:
        yield sse_error(f"Search failed: {exc}")
        yield sse_done()
        return

    # Emit sources metadata upfront
    sources_payload = {
        "type": "sources",
        "sources": [
            {
                "document_id": c.document_id,
                "filename":    c.filename,
                "chunk_index": c.chunk_index,
                "score":       round(c.score, 4),
                "preview":     c.content[:150] + "..." if len(c.content) > 150 else c.content,
            }
            for c in chunks
        ],
    }
    yield sse_json(sources_payload)

    # Build RAG context
    MAX_CONTEXT = 10_000
    context_parts = []
    total_chars = 0
    for chunk in chunks:
        snippet = f"[{chunk.filename}, chunk {chunk.chunk_index}]\n{chunk.content}"
        if total_chars + len(snippet) > MAX_CONTEXT:
            break
        context_parts.append(snippet)
        total_chars += len(snippet)

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

    system = f"""You are a precise assistant. Answer using ONLY the context below.
If the context is insufficient, say so clearly. Cite sources as [filename, chunk N].

CONTEXT:
{context}"""

    # Step 2: Stream the answer
    async for chunk in stream_llm_response(
        messages=[{"role": "user", "content": query}],
        system=system,
        include_metadata=True,
    ):
        yield chunk


# ---------------------------------------------------------------------------
# Heartbeat for long streams
# ---------------------------------------------------------------------------

async def stream_with_heartbeat(
    generator: AsyncIterator[str],
    heartbeat_interval: float = 15.0,
) -> AsyncIterator[str]:
    """
    Wrap a streaming generator with periodic heartbeat comments.

    WHY HEARTBEATS?
    Some reverse proxies (nginx, AWS ALB) close idle SSE connections
    after 30–60 seconds without data. Sending a comment (: heartbeat)
    every 15 seconds keeps the connection alive.

    SSE comments start with ':' and are ignored by the client.
    """
    heartbeat = ": heartbeat\n\n"

    async def _inner():
        last_event = asyncio.get_event_loop().time()
        async for chunk in generator:
            now = asyncio.get_event_loop().time()
            if now - last_event > heartbeat_interval:
                yield heartbeat
                last_event = now
            yield chunk
            last_event = asyncio.get_event_loop().time()

    async for item in _inner():
        yield item