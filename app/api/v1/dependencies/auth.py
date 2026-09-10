"""
app/api/v1/dependencies/auth.py
================================
FastAPI dependency functions for authentication and authorization.

AUTHENTICATION METHODS SUPPORTED:
  1. JWT Bearer Token (primary — web/mobile clients)
  2. API Key (header X-API-Key — server-to-server / CI bots)

HOW FASTAPI DEPENDS() CHAINS WORK:
  Dependencies are resolved top-down before the endpoint runs.
  If any dependency raises HTTPException, FastAPI returns the error response
  immediately — the endpoint function never executes.

  Example chain:
    get_current_user
      └→ extract Bearer token from Authorization header
      └→ decode JWT
      └→ load User from DB
      └→ check is_active
    require_role("admin")
      └→ calls get_current_user
      └→ checks user has 'admin' role
    endpoint(user=Depends(require_role("admin")))
      └→ runs only if both deps succeed

RBAC DESIGN:
  Roles are stored in the DB (roles + user_roles tables).
  Permissions within a role are stored as JSONB:
    {"documents": ["read", "write"], "admin": ["*"]}
  For simplicity, role-level access control is enforced in deps;
  permission-level checks are done inline in endpoints where needed.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db_session
from app.core.logging import get_logger
from app.core.security import decode_token, is_refresh_token_valid
from app.models.models import APIKey, User, UserRole, Role

logger = get_logger(__name__)

# FastAPI security schemes — define how tokens arrive
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Token extraction helpers
# ---------------------------------------------------------------------------

async def _get_user_from_jwt(
    token: str,
    db: AsyncSession,
) -> User:
    """
    Decode a JWT and load the corresponding User from PostgreSQL.
    Raises 401 on any auth failure.
    """
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token type must be 'access'",
        )

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )

    # Load user with roles eagerly (single DB round-trip)
    result = await db.execute(
        select(User)
        .options(
            selectinload(User.user_roles).selectinload(UserRole.role)
        )
        .where(User.id == user_id)
        .where(User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return user


async def _get_user_from_api_key(
    raw_key: str,
    db: AsyncSession,
) -> User:
    """
    Look up an API key by its SHA-256 hash and load the owning user.
    API keys are never stored in plaintext — only hashes.
    """
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    result = await db.execute(
        select(APIKey)
        .options(selectinload(APIKey.user).selectinload(User.user_roles).selectinload(UserRole.role))
        .where(APIKey.key_hash == key_hash)
        .where(APIKey.is_active == True)  # noqa: E712
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Check expiry
    if api_key.expires_at:
        from datetime import datetime, timezone
        if api_key.expires_at < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has expired",
            )

    # Update last used timestamp (non-blocking — don't await commit here)
    from datetime import datetime, timezone
    api_key.last_used_at = datetime.now(timezone.utc)

    user = api_key.user
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return user


# ---------------------------------------------------------------------------
# Core dependency: get_current_user
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    api_key: Annotated[str | None, Security(api_key_header)],
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """
    Primary auth dependency. Accepts EITHER:
      • Authorization: Bearer <jwt>
      • X-API-Key: <raw_api_key>

    Raises 401 if neither is present or valid.
    """
    if credentials and credentials.credentials:
        return await _get_user_from_jwt(credentials.credentials, db)

    if api_key:
        return await _get_user_from_api_key(api_key, db)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Type alias for clean endpoint signatures
CurrentUser = Annotated[User, Depends(get_current_user)]


# ---------------------------------------------------------------------------
# RBAC role checker factory
# ---------------------------------------------------------------------------

def require_role(*roles: str):
    """
    Returns a FastAPI dependency that enforces role membership.

    Usage in endpoint:
        @router.delete("/users/{id}")
        async def delete_user(
            user_id: str,
            current_user: User = Depends(require_role("admin")),
        ):
            ...

    Multiple roles are OR-ed: require_role("admin", "operator") passes if the
    user has EITHER role.
    """
    async def _check_role(
        current_user: User = Depends(get_current_user),
    ) -> User:
        user_roles = {ur.role.name for ur in current_user.user_roles}
        if not user_roles.intersection(set(roles)):
            logger.warning(
                "auth.rbac.denied",
                user_id=current_user.id,
                required_roles=roles,
                user_roles=list(user_roles),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role(s) required: {', '.join(roles)}",
            )
        return current_user

    return _check_role


def require_permission(resource: str, action: str):
    """
    Granular permission check within a role's permissions JSONB.

    Example: require_permission("documents", "delete")
    Checks: role.permissions["documents"] contains "delete" or "*"
    """
    async def _check_permission(
        current_user: User = Depends(get_current_user),
    ) -> User:
        for user_role in current_user.user_roles:
            perms = user_role.role.permissions or {}
            allowed = perms.get(resource, [])
            if "*" in allowed or action in allowed:
                return current_user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {resource}.{action}",
        )

    return _check_permission


# ---------------------------------------------------------------------------
# Optional auth (for public endpoints that enrich response if logged in)
# ---------------------------------------------------------------------------

async def get_optional_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: AsyncSession = Depends(get_db_session),
) -> User | None:
    """Returns the current user if authenticated, None otherwise. Never raises 401."""
    if not credentials or not credentials.credentials:
        return None
    try:
        return await _get_user_from_jwt(credentials.credentials, db)
    except HTTPException:
        return None


# ---------------------------------------------------------------------------
# Ownership guard
# ---------------------------------------------------------------------------

def require_owner_or_role(*roles: str):
    """
    Checks that current_user.id == resource_owner_id OR the user has one of the roles.
    Used for endpoints like GET /conversations/{id} — users can only see their own,
    admins can see all.
    """
    async def _check(
        resource_owner_id: str,
        current_user: User = Depends(get_current_user),
    ) -> User:
        if current_user.id == resource_owner_id:
            return current_user

        user_roles = {ur.role.name for ur in current_user.user_roles}
        if user_roles.intersection(set(roles)):
            return current_user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you don't own this resource",
        )

    return _check