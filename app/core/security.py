"""
app/core/security.py
====================
Authentication primitives:
  • Password hashing with bcrypt (via passlib)
  • JWT access token creation / verification
  • JWT refresh token creation / revocation (stored in Redis)

SECURITY DECISIONS:
  • bcrypt work factor = 12 — fast enough for login, slow enough to resist brute force.
  • Short-lived access tokens (30 min) — compromise window is small.
  • Long-lived refresh tokens (7 days) stored in Redis — enables instant revocation
    (logout, password change, account suspension all invalidate refresh tokens server-side).
  • Token payload includes sub (user_id), role, jti (unique token ID), iat, exp.
    The jti lets us blacklist individual tokens without invalidating the user's session.
  • We do NOT store the access token server-side (stateless JWT).
    We DO store the refresh token ID in Redis (stateful, revocable).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.redis_client import get_session_redis

settings = get_settings()
logger = get_logger(__name__)

# bcrypt context — automatically handles salt generation and versioning
pwd_context = CryptContext(
    schemes=["bcrypt_sha256", "bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    """Hash a plaintext password. Returns a bcrypt hash string."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time comparison prevents timing attacks."""
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT Tokens
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: str,
    role: str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """
    Create a signed JWT access token.

    Payload fields:
      sub  — subject (user UUID)
      role — user role string (used for RBAC checks)
      jti  — unique token ID (for future blacklisting)
      iat  — issued at
      exp  — expiry
      type — "access" (distinguishes from refresh)
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.jwt.access_token_expire_minutes)

    payload: dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": expire,
        "type": "access",
        **(extra_claims or {}),
    }

    token = jwt.encode(
        payload,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm,
    )
    return token


async def create_refresh_token(user_id: str, role: str) -> str:
    """
    Create a refresh token and register its JTI in Redis.
    Redis key:  refresh:<jti>  → user_id
    TTL:        REFRESH_TOKEN_EXPIRE_DAYS * 86400 seconds
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.jwt.refresh_token_expire_days)
    jti = str(uuid.uuid4())

    payload: dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "jti": jti,
        "iat": now,
        "exp": expire,
        "type": "refresh",
    }

    token = jwt.encode(
        payload,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm,
    )

    # Register in Redis — this is the server-side state that enables revocation
    redis = get_session_redis()
    ttl = settings.jwt.refresh_token_expire_days * 86400
    await redis.setex(f"refresh:{jti}", ttl, user_id)

    logger.info("auth.refresh_token.created", user_id=user_id, jti=jti)
    return token


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT.
    Raises JWTError (caught by auth dependency) on any failure:
      • Expired
      • Invalid signature
      • Malformed
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt.secret_key,
            algorithms=[settings.jwt.algorithm],
        )
        return payload
    except JWTError as exc:
        logger.warning("auth.token.invalid", error=str(exc))
        raise


async def revoke_refresh_token(jti: str) -> None:
    """Remove refresh token from Redis → immediate logout."""
    redis = get_session_redis()
    await redis.delete(f"refresh:{jti}")
    logger.info("auth.refresh_token.revoked", jti=jti)


async def is_refresh_token_valid(jti: str) -> bool:
    """Check whether a refresh token JTI still exists in Redis."""
    redis = get_session_redis()
    return bool(await redis.exists(f"refresh:{jti}"))


async def revoke_all_user_tokens(user_id: str) -> None:
    """
    Revoke all refresh tokens for a user (e.g., password change, account ban).
    Scans for keys matching refresh:* and deletes those pointing to user_id.
    NOTE: In very high-volume systems, use a user→[jti] set instead of scan.
    """
    redis = get_session_redis()
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = await redis.scan(cursor, match="refresh:*", count=100)
        for key in keys:
            if await redis.get(key) == user_id:
                await redis.delete(key)
                deleted += 1
        if cursor == 0:
            break
    logger.info("auth.tokens.revoked_all", user_id=user_id, count=deleted)