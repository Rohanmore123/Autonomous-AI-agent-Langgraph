"""
app/core/redis_client.py
========================
Async Redis connection pools for four distinct purposes:

  DB 0 — SESSIONS:      JWT refresh token store, user session data
  DB 1 — CACHE:         LLM response caching, embedding caching
  DB 2 — CELERY:        Task queue broker + result backend
  DB 3 — RATE LIMIT:    Sliding-window counters per user/IP

Using separate logical databases (DB 0–3 on the same Redis instance) keeps
concerns isolated and allows selective FLUSHDB without side effects.
In large-scale deployments, point each URL at a separate Redis cluster.

WHY REDIS FOR CACHING LLM RESPONSES?
  LLM calls are expensive (~$0.01–$0.10 each) and slow (1–10 seconds).
  Identical questions from different users can be served from cache in <1ms.
  Cache key = SHA-256(model + messages) ensures correctness.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Connection pools (created once at startup)
# ---------------------------------------------------------------------------

_session_pool: aioredis.Redis | None = None
_cache_pool: aioredis.Redis | None = None
_rate_limit_pool: aioredis.Redis | None = None


async def init_redis() -> None:
    """Create connection pools. Called from app lifespan."""
    global _session_pool, _cache_pool, _rate_limit_pool

    pool_kwargs = dict(
        max_connections=settings.redis.max_connections,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )

    _session_pool = await aioredis.from_url(settings.redis.url, **pool_kwargs)
    _cache_pool = await aioredis.from_url(settings.redis.cache_url, **pool_kwargs)
    _rate_limit_pool = await aioredis.from_url(settings.redis.rate_limit_url, **pool_kwargs)

    # Verify connectivity
    for name, pool in [
        ("session", _session_pool),
        ("cache", _cache_pool),
        ("rate_limit", _rate_limit_pool),
    ]:
        await pool.ping()
        logger.info("redis.connected", pool=name)


async def close_redis() -> None:
    """Close all Redis connections. Called from app lifespan shutdown."""
    for pool in [_session_pool, _cache_pool, _rate_limit_pool]:
        if pool:
            await pool.aclose()
    logger.info("redis.closed")


def get_session_redis() -> aioredis.Redis:
    if _session_pool is None:
        raise RuntimeError("Redis session pool not initialised")
    return _session_pool


def get_cache_redis() -> aioredis.Redis:
    if _cache_pool is None:
        raise RuntimeError("Redis cache pool not initialised")
    return _cache_pool


def get_rate_limit_redis() -> aioredis.Redis:
    if _rate_limit_pool is None:
        raise RuntimeError("Redis rate limit pool not initialised")
    return _rate_limit_pool


# ---------------------------------------------------------------------------
# LLM Response Cache
# ---------------------------------------------------------------------------

class LLMCache:
    """
    Semantic cache for LLM responses.

    Cache key = SHA-256(provider + model + sorted messages JSON).
    TTL defaults to 1 hour — tune per use case.
    """

    DEFAULT_TTL = 3600  # seconds

    def __init__(self) -> None:
        self._redis = get_cache_redis()

    @staticmethod
    def _make_key(provider: str, model: str, messages: list[dict]) -> str:
        raw = json.dumps(
            {"provider": provider, "model": model, "messages": messages},
            sort_keys=True,
        )
        return "llm:cache:" + hashlib.sha256(raw.encode()).hexdigest()

    async def get(
        self, provider: str, model: str, messages: list[dict]
    ) -> str | None:
        key = self._make_key(provider, model, messages)
        value = await self._redis.get(key)
        if value:
            logger.info("llm.cache.hit", key=key[:16])
        return value

    async def set(
        self,
        provider: str,
        model: str,
        messages: list[dict],
        response: str,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        key = self._make_key(provider, model, messages)
        await self._redis.setex(key, ttl, response)
        logger.info("llm.cache.set", key=key[:16], ttl=ttl)

    async def invalidate(self, pattern: str = "llm:cache:*") -> int:
        """Bulk invalidation — use with care in production."""
        keys = await self._redis.keys(pattern)
        if keys:
            return await self._redis.delete(*keys)
        return 0


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def check_redis_health() -> bool:
    try:
        await get_cache_redis().ping()
        return True
    except Exception as exc:
        logger.error("redis.health_check.failed", error=str(exc))
        return False