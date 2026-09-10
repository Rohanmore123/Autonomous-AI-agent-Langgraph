"""
app/api/v1/endpoints/admin.py
==============================
Admin-only endpoints for platform management:
  GET  /admin/users             — list all users (paginated)
  GET  /admin/users/{id}        — get user details
  PATCH /admin/users/{id}       — update user status / roles
  GET  /admin/agent-runs        — view all agent run logs with filters
  GET  /admin/metrics           — platform-wide metrics summary
  POST /admin/roles             — create a new role
  POST /admin/roles/assign      — assign role to user
  GET  /health                  — system health check (public)
  GET  /metrics/summary         — high-level stats for dashboard

app/api/v1/endpoints/gmail.py
==============================
Gmail agent specific endpoints:
  GET  /gmail/search            — search emails
  GET  /gmail/email/{id}        — get full email
  POST /gmail/send              — send email
  GET  /gmail/summarise         — LLM-powered inbox summary
  POST /gmail/draft-reply/{id}  — draft a reply to an email
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.v1.dependencies.auth import CurrentUser, require_role
from app.core.config import get_settings
from app.core.database import check_db_health, get_db_session
from app.core.logging import get_logger
from app.core.redis_client import check_redis_health
from app.models.models import AgentRun, AuditLog, Role, User, UserRole
from app.schemas.schemas import (
    APIResponse,
    AgentRunResponse,
    AssignRoleRequest,
    HealthResponse,
    MetricsSummary,
    PaginatedResponse,
    RoleResponse,
    UserAdminUpdate,
    UserResponse,
)
from app.services.vector_db.weaviate_client import check_weaviate_health

settings = get_settings()
logger = get_logger(__name__)

# ============================================================================
# ADMIN ROUTER
# ============================================================================

admin_router = APIRouter(prefix="/admin", tags=["Admin"])


@admin_router.get(
    "/users",
    response_model=APIResponse[PaginatedResponse],
    summary="[Admin] List all users",
)
async def list_users(
    page: int = 1,
    page_size: int = 50,
    search: str | None = None,
    current_user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    offset = (page - 1) * page_size
    q = (
        select(User)
        .options(selectinload(User.user_roles).selectinload(UserRole.role))
        .where(User.deleted_at.is_(None))
        .order_by(User.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    if search:
        q = q.where(User.email.ilike(f"%{search}%") | User.username.ilike(f"%{search}%"))

    total_q = select(func.count()).select_from(User).where(User.deleted_at.is_(None))
    users = (await db.execute(q)).scalars().all()
    total = (await db.execute(total_q)).scalar_one()

    return APIResponse(
        data=PaginatedResponse(
            items=[
                UserResponse(
                    id=u.id, email=u.email, username=u.username,
                    full_name=u.full_name, is_active=u.is_active, is_verified=u.is_verified,
                    created_at=u.created_at, last_login_at=u.last_login_at,
                    roles=[ur.role.name for ur in u.user_roles],
                )
                for u in users
            ],
            total=total, page=page, page_size=page_size,
            pages=(total + page_size - 1) // page_size,
        )
    )


@admin_router.patch(
    "/users/{user_id}",
    response_model=APIResponse[UserResponse],
    summary="[Admin] Update user status or roles",
)
async def update_user(
    user_id: str,
    body: UserAdminUpdate,
    current_user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    result = await db.execute(
        select(User)
        .options(selectinload(User.user_roles).selectinload(UserRole.role))
        .where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if body.is_active is not None:
        user.is_active = body.is_active
        if not body.is_active:
            from app.core.security import revoke_all_user_tokens
            await revoke_all_user_tokens(user_id)

    if body.is_verified is not None:
        user.is_verified = body.is_verified

    if body.roles is not None:
        # Replace all roles
        for ur in user.user_roles:
            await db.delete(ur)
        await db.flush()

        for role_name in body.roles:
            role_result = await db.execute(select(Role).where(Role.name == role_name))
            role = role_result.scalar_one_or_none()
            if role:
                db.add(UserRole(user_id=user.id, role_id=role.id, granted_by=current_user.id))

    await db.commit()
    await db.refresh(user)

    logger.info("admin.user.updated", target_user=user_id, by=current_user.id)
    return APIResponse(
        data=UserResponse(
            id=user.id, email=user.email, username=user.username,
            full_name=user.full_name, is_active=user.is_active, is_verified=user.is_verified,
            created_at=user.created_at, last_login_at=user.last_login_at,
            roles=[ur.role.name for ur in user.user_roles],
        )
    )


@admin_router.get(
    "/agent-runs",
    response_model=APIResponse[PaginatedResponse],
    summary="[Admin] View all agent run logs",
)
async def list_agent_runs(
    page: int = 1,
    page_size: int = 50,
    agent_type: str | None = None,
    user_id: str | None = None,
    success: bool | None = None,
    current_user: User = Depends(require_role("admin", "operator")),
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    offset = (page - 1) * page_size
    q = select(AgentRun).order_by(desc(AgentRun.created_at)).offset(offset).limit(page_size)

    if agent_type:
        q = q.where(AgentRun.agent_type == agent_type)
    if user_id:
        q = q.where(AgentRun.user_id == user_id)
    if success is not None:
        q = q.where(AgentRun.success == success)

    total_q = select(func.count()).select_from(AgentRun)
    runs = (await db.execute(q)).scalars().all()
    total = (await db.execute(total_q)).scalar_one()

    return APIResponse(
        data=PaginatedResponse(
            items=[AgentRunResponse.model_validate(r) for r in runs],
            total=total, page=page, page_size=page_size,
            pages=(total + page_size - 1) // page_size,
        )
    )


@admin_router.get(
    "/metrics",
    response_model=APIResponse[MetricsSummary],
    summary="[Admin] Platform metrics summary",
)
async def get_metrics(
    current_user: User = Depends(require_role("admin", "operator")),
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    """Compute key platform metrics for the last 24 hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    total_requests = (
        await db.execute(
            select(func.count()).select_from(AgentRun).where(AgentRun.created_at >= since)
        )
    ).scalar_one()

    total_tokens_row = await db.execute(
        select(
            func.sum(AgentRun.input_tokens + AgentRun.output_tokens)
        ).where(AgentRun.created_at >= since)
    )
    total_tokens = total_tokens_row.scalar_one() or 0

    avg_latency_row = await db.execute(
        select(func.avg(AgentRun.latency_ms)).where(AgentRun.created_at >= since)
    )
    avg_latency = float(avg_latency_row.scalar_one() or 0.0)

    cached_count = (
        await db.execute(
            select(func.count()).select_from(AgentRun)
            .where(AgentRun.created_at >= since, AgentRun.was_cached == True)  # noqa
        )
    ).scalar_one()

    error_count = (
        await db.execute(
            select(func.count()).select_from(AgentRun)
            .where(AgentRun.created_at >= since, AgentRun.success == False)  # noqa
        )
    ).scalar_one()

    active_users = (
        await db.execute(
            select(func.count(AgentRun.user_id.distinct()))
            .where(AgentRun.created_at >= since)
        )
    ).scalar_one()

    # Agent breakdown
    agent_rows = await db.execute(
        select(AgentRun.agent_type, func.count().label("cnt"))
        .where(AgentRun.created_at >= since)
        .group_by(AgentRun.agent_type)
    )
    agent_breakdown = {row.agent_type: row.cnt for row in agent_rows}

    cache_hit_rate = (cached_count / total_requests) if total_requests > 0 else 0.0
    error_rate = (error_count / total_requests) if total_requests > 0 else 0.0

    return APIResponse(
        data=MetricsSummary(
            total_requests_today=total_requests,
            total_tokens_today=total_tokens,
            avg_latency_ms=round(avg_latency, 2),
            cache_hit_rate=round(cache_hit_rate, 4),
            error_rate=round(error_rate, 4),
            active_users=active_users,
            agent_breakdown=agent_breakdown,
        )
    )


@admin_router.post(
    "/roles",
    response_model=APIResponse[RoleResponse],
    status_code=201,
    summary="[Admin] Create a new role",
)
async def create_role(
    name: str,
    description: str = "",
    permissions: dict = {},
    current_user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    existing = await db.execute(select(Role).where(Role.name == name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Role '{name}' already exists")

    role = Role(name=name, description=description, permissions=permissions)
    db.add(role)
    await db.commit()
    return APIResponse(data=RoleResponse(
        id=role.id, name=role.name, description=role.description, permissions=role.permissions
    ))


@admin_router.post(
    "/roles/assign",
    response_model=APIResponse,
    summary="[Admin] Assign role to user",
)
async def assign_role(
    body: AssignRoleRequest,
    current_user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    role_result = await db.execute(select(Role).where(Role.name == body.role_name))
    role = role_result.scalar_one_or_none()
    if not role:
        raise HTTPException(status_code=404, detail=f"Role '{body.role_name}' not found")

    user_result = await db.execute(select(User).where(User.id == body.user_id))
    if not user_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="User not found")

    db.add(UserRole(user_id=body.user_id, role_id=role.id, granted_by=current_user.id))
    await db.commit()
    logger.info("admin.role.assigned", role=body.role_name, user=body.user_id)
    return APIResponse(message=f"Role '{body.role_name}' assigned to user {body.user_id}")


# ============================================================================
# HEALTH ROUTER (public)
# ============================================================================

health_router = APIRouter(tags=["Health"])
_start_time = datetime.now(timezone.utc)


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
async def health_check() -> HealthResponse:
    """
    Public health endpoint used by load balancers and monitoring.
    Checks connectivity to all external dependencies.
    Returns 200 if healthy, 503 if any critical component is down.
    """
    db_ok = await check_db_health()
    redis_ok = await check_redis_health()
    weaviate_ok = await check_weaviate_health()
    llm_ok = True  # LLM health is circuit-breaker based; not pinged here

    components = {
        "postgres": db_ok,
        "redis": redis_ok,
        "weaviate": weaviate_ok,
        "llm": llm_ok,
    }

    all_ok = all(components.values())
    critical_ok = db_ok and redis_ok  # LLM + Weaviate degraded is acceptable

    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()

    return HealthResponse(
        status="healthy" if all_ok else ("degraded" if critical_ok else "unhealthy"),
        version=settings.app_version,
        environment=settings.app_env,
        components=components,
        uptime_seconds=uptime,
    )


# ============================================================================
# GMAIL ROUTER
# ============================================================================

gmail_router = APIRouter(prefix="/gmail", tags=["Gmail Agent"])


@gmail_router.get(
    "/search",
    summary="Search emails using Gmail query syntax",
)
async def search_emails(
    query: str = Query("in:inbox is:unread", description="Gmail search query"),
    max_results: int = Query(10, ge=1, le=50),
    include_body: bool = False,
    current_user: CurrentUser = ...,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    from app.services.agents.gmail_agent_2 import GmailAgent
    agent = GmailAgent(current_user.id, db)
    emails = await agent.search_emails(query, max_results, include_body)
    return APIResponse(data={"emails": [e.model_dump() for e in emails], "count": len(emails)})


@gmail_router.get("/email/{message_id}", summary="Get full email by ID")
async def get_email(
    message_id: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    from app.services.agents.gmail_agent_2 import GmailAgent
    agent = GmailAgent(current_user.id, db)
    email = await agent.get_email(message_id)
    return APIResponse(data=email.model_dump())


@gmail_router.post("/send", summary="Send an email")
async def send_email(
    body: dict,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    from app.schemas.schemas import GmailSendRequest
    from app.services.agents.gmail_agent_2 import GmailAgent
    req = GmailSendRequest(**body)
    agent = GmailAgent(current_user.id, db)
    message_id = await agent.send_email(
        to=req.to, subject=req.subject, body=req.body, cc=req.cc, bcc=req.bcc
    )
    return APIResponse(data={"message_id": message_id}, message="Email sent")


@gmail_router.get("/summarise", summary="LLM-powered inbox summary")
async def summarise_inbox(
    query: str = "in:inbox is:unread",
    current_user: CurrentUser = ...,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    from app.services.agents.gmail_agent_2 import GmailAgent
    agent = GmailAgent(current_user.id, db)
    summary = await agent.summarise_inbox(query)
    return APIResponse(data={"summary": summary})


@gmail_router.post("/draft-reply/{message_id}", summary="Draft an AI-generated reply")
async def draft_reply(
    message_id: str,
    instructions: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    from app.services.agents.gmail_agent_2 import GmailAgent
    agent = GmailAgent(current_user.id, db)
    draft = await agent.draft_reply(message_id, instructions)
    return APIResponse(data={"draft": draft})