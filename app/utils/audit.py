"""
app/utils/audit.py
===================
Centralised audit logging service.

WHY AN AUDIT LOG?
  In a multi-user, RBAC platform, you MUST be able to answer:
    • "Who deleted document X?"
    • "When was user Y's role changed to admin, and who did it?"
    • "Show all login attempts from IP 1.2.3.4 in the last 24h"
    • "What did user Z do before the data was corrupted?"

  The audit_logs table is:
    • Append-only (no UPDATE/DELETE ever)
    • Written for every security-relevant action
    • Indexed on user_id, action, created_at for fast queries
    • Retained for 90 days (configurable)

ACTION NAMING CONVENTION:
  <resource>.<verb>   e.g.:
    user.login.success        user.login.failed
    user.registered           user.password.changed
    user.role.assigned        user.deactivated
    document.uploaded         document.deleted
    conversation.created      conversation.deleted
    api_key.created           api_key.revoked
    admin.user.updated        admin.role.created

USAGE:
  from app.utils.audit import audit_log

  # In an endpoint:
  await audit_log(
      db=db,
      user_id=current_user.id,
      action="document.deleted",
      resource_type="document",
      resource_id=doc.id,
      request=request,
      details={"filename": doc.filename},
  )
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.models import AuditLog

logger = get_logger(__name__)


async def audit_log(
    db: AsyncSession,
    action: str,
    user_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    details: dict[str, Any] | None = None,
    request: Request | None = None,
) -> None:
    """
    Write a record to the audit_logs table.

    Designed to be called inside an existing DB transaction — the commit
    happens at the endpoint level so the audit log and the business action
    are written atomically (either both succeed or both fail).

    Args:
        db:            Active SQLAlchemy async session
        action:        Dot-namespaced action string, e.g. "document.deleted"
        user_id:       ID of the acting user (None for unauthenticated events)
        resource_type: Type of resource affected, e.g. "document", "user"
        resource_id:   ID of the specific resource
        details:       Additional context (filename, old/new values, etc.)
        request:       FastAPI Request object — used to extract IP and user agent
    """
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str | None = None

    if request:
        # Handle reverse proxy — trust X-Forwarded-For if present
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        ip_address = (
            forwarded_for.split(",")[0].strip()
            if forwarded_for
            else (request.client.host if request.client else None)
        )
        user_agent = request.headers.get("User-Agent", "")[:500]
        request_id = getattr(request.state, "request_id", None)

    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
        details=details or {},
    )
    db.add(entry)

    # Also emit to structured log so it appears in log aggregators immediately
    # (DB write is authoritative; log is for real-time alerting)
    logger.info(
        "audit",
        action=action,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        ip=ip_address,
        request_id=request_id,
    )


class AuditActions:
    """
    Constants for all audit action strings.
    Using constants (not raw strings) prevents typos and enables IDE autocomplete.
    """

    # Auth
    USER_REGISTERED       = "user.registered"
    USER_LOGIN_SUCCESS    = "user.login.success"
    USER_LOGIN_FAILED     = "user.login.failed"
    USER_LOGOUT           = "user.logout"
    USER_LOGOUT_ALL       = "user.logout.all"
    USER_PASSWORD_CHANGED = "user.password.changed"
    USER_DEACTIVATED      = "user.deactivated"
    USER_ACTIVATED        = "user.activated"
    GOOGLE_CONNECTED      = "user.google.connected"

    # RBAC
    ROLE_ASSIGNED         = "user.role.assigned"
    ROLE_REVOKED          = "user.role.revoked"

    # API Keys
    API_KEY_CREATED       = "api_key.created"
    API_KEY_REVOKED       = "api_key.revoked"

    # Documents
    DOCUMENT_UPLOADED     = "document.uploaded"
    DOCUMENT_DELETED      = "document.deleted"
    DOCUMENT_SEARCHED     = "document.searched"

    # Conversations
    CONVERSATION_CREATED  = "conversation.created"
    CONVERSATION_DELETED  = "conversation.deleted"

    # Admin
    ADMIN_USER_UPDATED    = "admin.user.updated"
    ADMIN_ROLE_CREATED    = "admin.role.created"

    # Security
    SUSPICIOUS_ACTIVITY   = "security.suspicious_activity"
    RATE_LIMIT_HIT        = "security.rate_limit_hit"