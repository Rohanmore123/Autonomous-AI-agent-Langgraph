"""
app/middleware/rate_limit.py
=============================
Sliding-window rate limiter implemented with Redis.

WHY SLIDING WINDOW vs FIXED WINDOW?
  Fixed window: "100 requests per minute" resets at :00 each minute.
  Problem: a user can fire 100 at :59 + 100 at :01 → 200 in 2 seconds.

  Sliding window: counts requests in the last N seconds from NOW.
  Much fairer. We implement it with a Redis sorted set:
    Key:   rate:<identifier>:<window_name>
    Score: unix timestamp (float)
    Member: uuid (unique per request)

  Algorithm per request:
    1. ZREMRANGEBYSCORE — drop members older than window
    2. ZCARD           — count remaining members
    3. if count >= limit → 429
    4. else ZADD new member, EXPIRE key

  This is O(log N) + O(1) per request — fast enough at 10K concurrent users.

IDENTIFIER STRATEGY:
  • Authenticated user → rate:<user_id>:<window>
  • Unauthenticated    → rate:<ip>:<window>
  Using user_id prevents IP-spoofing by rotating IPs.

WINDOWS:
  per_minute: 60 req/min  — protects against burst abuse
  per_hour:   1000 req/hr — protects against sustained abuse

HEADERS RETURNED:
  X-RateLimit-Limit     — window limit
  X-RateLimit-Remaining — remaining requests
  X-RateLimit-Reset     — Unix timestamp when window resets
  Retry-After           — seconds to wait (only on 429)
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.redis_client import get_rate_limit_redis

settings = get_settings()
logger = get_logger(__name__)

# Endpoints exempt from rate limiting
EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}

# Rate limit windows: (limit, window_seconds, window_name)
WINDOWS = [
    (settings.rate_limit.per_minute, 60,   "per_min"),
    (settings.rate_limit.per_hour,   3600, "per_hour"),
]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter applied to every request.
    Checks both per-minute and per-hour windows.
    The strictest window that's exceeded triggers the 429.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip exempt paths
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # Identify requester
        identifier = self._get_identifier(request)

        # Check all windows
        for limit, window_seconds, window_name in WINDOWS:
            allowed, remaining, reset_at = await self._check_window(
                identifier=identifier,
                limit=limit,
                window_seconds=window_seconds,
                window_name=window_name,
            )

            if not allowed:
                retry_after = max(0, int(reset_at - time.time()))
                logger.warning(
                    "rate_limit.exceeded",
                    identifier=identifier,
                    window=window_name,
                    limit=limit,
                    retry_after=retry_after,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "errors": [{
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": f"Too many requests. Limit: {limit}/{window_name}. "
                                       f"Retry after {retry_after} seconds.",
                        }],
                    },
                    headers={
                        "X-RateLimit-Limit":     str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset":     str(int(reset_at)),
                        "Retry-After":           str(retry_after),
                    },
                )

        # All windows passed — proceed
        response = await call_next(request)

        # Attach rate limit headers from the per-minute window (most visible)
        _, remaining_min, reset_min = await self._peek_window(identifier, WINDOWS[0][0], 60, "per_min")
        response.headers["X-RateLimit-Limit"]     = str(WINDOWS[0][0])
        response.headers["X-RateLimit-Remaining"] = str(remaining_min)
        response.headers["X-RateLimit-Reset"]     = str(int(reset_min))

        return response

    # -----------------------------------------------------------------------

    @staticmethod
    def _get_identifier(request: Request) -> str:
        """
        Extract rate-limit identity.
        JWT sub claim is the gold standard (survives IP rotation).
        Falls back to X-Forwarded-For (behind reverse proxy) then client IP.
        """
        # Check if JWT is present — parse it without full validation for speed
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            try:
                # Fast decode without signature verification (we only need 'sub')
                import base64, json
                payload_b64 = token.split(".")[1]
                # Pad base64
                padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
                payload = json.loads(base64.urlsafe_b64decode(padded))
                if sub := payload.get("sub"):
                    return f"user:{sub}"
            except Exception:
                pass

        # API key hash (first 16 chars of hash for privacy)
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            import hashlib
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"apikey:{key_hash}"

        # IP address fallback
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        ip = forwarded_for.split(",")[0].strip() if forwarded_for else (
            request.client.host if request.client else "unknown"
        )
        return f"ip:{ip}"

    @staticmethod
    async def _check_window(
        identifier: str,
        limit: int,
        window_seconds: int,
        window_name: str,
    ) -> tuple[bool, int, float]:
        """
        Sliding window check using Redis sorted set.
        Returns: (is_allowed, remaining, reset_timestamp)
        """
        redis = get_rate_limit_redis()
        now = time.time()
        window_start = now - window_seconds
        key = f"rate:{identifier}:{window_name}"
        member = str(uuid.uuid4())

        pipe = redis.pipeline()
        # 1. Remove expired members
        pipe.zremrangebyscore(key, 0, window_start)
        # 2. Count current members
        pipe.zcard(key)
        # 3. Add this request
        pipe.zadd(key, {member: now})
        # 4. Set expiry on the key itself
        pipe.expire(key, window_seconds + 1)
        results = await pipe.execute()

        current_count = results[1]  # count BEFORE adding this request

        if current_count >= limit:
            # Undo the add (we won't count this rejected request)
            await redis.zrem(key, member)
            reset_at = now + window_seconds
            return False, 0, reset_at

        remaining = limit - current_count - 1
        reset_at = now + window_seconds
        return True, remaining, reset_at

    @staticmethod
    async def _peek_window(
        identifier: str,
        limit: int,
        window_seconds: int,
        window_name: str,
    ) -> tuple[bool, int, float]:
        """Non-modifying window peek for attaching headers to successful responses."""
        redis = get_rate_limit_redis()
        now = time.time()
        window_start = now - window_seconds
        key = f"rate:{identifier}:{window_name}"

        current_count = await redis.zcount(key, window_start, "+inf")
        remaining = max(0, limit - current_count)
        return True, remaining, now + window_seconds