"""
app/services/llm/embeddings.py
================================
Embedding service for converting text → dense vectors.

EMBEDDING STRATEGY:
  Primary:  sentence-transformers/all-MiniLM-L6-v2 (local, 384-dim, fast)
  Fallback: OpenAI text-embedding-3-small (API, 1536-dim, higher quality)

WHY LOCAL EMBEDDINGS AS PRIMARY?
  • Zero API cost (runs on CPU in the same process)
  • No latency added by network calls
  • ~80ms per sentence on CPU, ~5ms on GPU
  • 384 dimensions is sufficient for most RAG use cases
  • Privacy: documents never leave your infrastructure

WHEN TO USE OPENAI EMBEDDINGS:
  • When document quality/recall is critical
  • When you have GPU-constrained infrastructure
  • For multilingual documents (OpenAI embeddings are stronger cross-lingual)

CACHING:
  Embeddings are cached in Redis for 24 hours.
  Cache key = SHA-256(model_name + text)
  This is extremely effective for repeated queries (same question asked by different users).

NORMALISATION:
  All embeddings are L2-normalised so cosine similarity = dot product.
  Weaviate stores normalised vectors and uses dot product internally.
  Normalisation ensures consistent scoring across different text lengths.

BATCH PROCESSING:
  Always embed in batches — single model.encode() call per batch is 10–50x
  faster than calling embed_text() in a loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.redis_client import get_cache_redis

settings = get_settings()
logger = get_logger(__name__)

CACHE_TTL_SECONDS = 86_400   # 24 hours
CACHE_PREFIX = "embed:"


class EmbeddingService:
    """
    Manages embedding model loading, caching, and batch inference.

    Thread Safety:
      SentenceTransformer is thread-safe for inference.
      The model is loaded once and shared across all requests.
      In multi-worker setups, each worker process loads its own copy.

    Memory:
      all-MiniLM-L6-v2 uses ~90MB RAM — acceptable per worker.
      For GPU deployment, set SENTENCE_TRANSFORMERS_HOME and the model
      will be shared via mmap if multiple workers load the same model.
    """

    _model: SentenceTransformer | None = None
    _model_name: str = ""

    def __init__(self) -> None:
        self._model_name = settings.llm.embedding_model

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load and cache the embedding model."""
        if EmbeddingService._model is None or EmbeddingService._model_name != self._model_name:
            logger.info("embedding.model.loading", model=self._model_name)
            t0 = time.perf_counter()
            EmbeddingService._model = SentenceTransformer(
                self._model_name,
                cache_folder="/tmp/sentence_transformers",  # Mount as volume in Docker
            )
            EmbeddingService._model_name = self._model_name
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("embedding.model.loaded", model=self._model_name, ms=round(elapsed))
        return EmbeddingService._model

    # -----------------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _cache_key(model: str, text: str) -> str:
        content = json.dumps({"model": model, "text": text}, ensure_ascii=False)
        return CACHE_PREFIX + hashlib.sha256(content.encode()).hexdigest()

    async def _get_cached(self, text: str) -> list[float] | None:
        if not settings.enable_cache:
            return None
        try:
            redis = get_cache_redis()
            key = self._cache_key(self._model_name, text)
            raw = await redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning("embedding.cache.get_failed", error=str(exc))
        return None

    async def _set_cached(self, text: str, vector: list[float]) -> None:
        if not settings.enable_cache:
            return
        try:
            redis = get_cache_redis()
            key = self._cache_key(self._model_name, text)
            await redis.setex(key, CACHE_TTL_SECONDS, json.dumps(vector))
        except Exception as exc:
            logger.warning("embedding.cache.set_failed", error=str(exc))

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def embed_sync(self, text: str) -> list[float]:
        """
        Synchronous embedding — use in Celery workers (no event loop).
        Returns L2-normalised vector as list of floats.
        """
        model = self._get_model()
        vector = model.encode(
            text,
            normalize_embeddings=True,   # L2 normalise
            show_progress_bar=False,
        )
        return vector.tolist()

    async def embed(self, text: str) -> list[float]:
        """
        Async embedding with Redis cache.
        Cache hit: ~1ms
        Cache miss: ~80ms (CPU) or ~5ms (GPU)
        """
        # Check cache first
        cached = await self._get_cached(text)
        if cached is not None:
            return cached

        # Run CPU-bound embedding in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, self.embed_sync, text)

        # Cache the result
        await self._set_cached(text, vector)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Batch embed multiple texts.

        Strategy:
          1. Check cache for each text (parallel Redis MGET)
          2. Embed only cache-miss texts in one model.encode() call
          3. Store new embeddings in cache
          4. Return results in original order

        This is 10–50x faster than calling embed() in a loop.
        """
        if not texts:
            return []

        t0 = time.perf_counter()

        # Parallel cache lookup
        redis = get_cache_redis()
        keys = [self._cache_key(self._model_name, t) for t in texts]

        cached_results: list[list[float] | None] = [None] * len(texts)
        try:
            raw_values = await redis.mget(keys)
            for i, raw in enumerate(raw_values):
                if raw:
                    cached_results[i] = json.loads(raw)
        except Exception as exc:
            logger.warning("embedding.batch_cache.get_failed", error=str(exc))

        # Find cache misses
        miss_indices = [i for i, v in enumerate(cached_results) if v is None]
        miss_texts   = [texts[i] for i in miss_indices]

        if miss_texts:
            # Batch embed all misses in one call (fast!)
            loop = asyncio.get_event_loop()

            def _batch_encode():
                model = self._get_model()
                return model.encode(
                    miss_texts,
                    batch_size=64,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

            vectors_np = await loop.run_in_executor(None, _batch_encode)
            vectors = [v.tolist() for v in vectors_np]

            # Store in cache (fire-and-forget, no await needed for perf)
            try:
                pipe = redis.pipeline()
                for i, (idx, vector) in enumerate(zip(miss_indices, vectors)):
                    cached_results[idx] = vector
                    pipe.setex(keys[idx], CACHE_TTL_SECONDS, json.dumps(vector))
                await pipe.execute()
            except Exception as exc:
                logger.warning("embedding.batch_cache.set_failed", error=str(exc))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        cache_hits = len(texts) - len(miss_texts)

        logger.info(
            "embedding.batch.complete",
            total=len(texts),
            cache_hits=cache_hits,
            computed=len(miss_texts),
            latency_ms=round(elapsed_ms, 2),
        )

        return cached_results  # type: ignore[return-value]

    async def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts.
        Returns value in [-1, 1] where 1 = identical, 0 = unrelated.
        Since vectors are L2-normalised, cosine similarity = dot product.
        """
        vec_a, vec_b = await asyncio.gather(self.embed(text_a), self.embed(text_b))
        return float(np.dot(vec_a, vec_b))

    async def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find top-K most similar candidates to a query.
        Returns: [(candidate_index, score), ...] sorted by score desc.
        Useful for in-memory search over small candidate sets (<1000 items).
        For large sets, use Weaviate.
        """
        query_vec = np.array(await self.embed(query))
        candidate_vecs = np.array(await self.embed_batch(candidates))

        # Vectorised dot product (fast with numpy)
        scores = candidate_vecs @ query_vec

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


# Module-level singleton (loaded once per worker process)
embedding_service = EmbeddingService()