"""
app/main.py
============
FastAPI application factory and entrypoint.

STARTUP SEQUENCE (order matters):
  1. setup_logging()          — structured logging must come first
  2. setup_tracing()          — OTel provider registered before any spans created
  3. init_redis()             — connection pools for rate limiter + cache
  4. init_weaviate()          — create Weaviate collection schema if needed
  5. seed_initial_roles()     — ensure default roles exist (idempotent)
  6. FastAPI app created      — all middleware registered
  7. setup_metrics(app)       — Prometheus /metrics endpoint registered
  8. setup_fastapi_tracing()  — OTel FastAPI auto-instrumentation wired

SHUTDOWN SEQUENCE (lifespan exit):
  1. close_db()     — dispose SQLAlchemy connection pool
  2. close_redis()  — close Redis connection pools
  3. close_weaviate()

MIDDLEWARE STACK (applied bottom-up — last added = outermost):
  1. ErrorHandlerMiddleware   — catches all unhandled exceptions
  2. RequestContextMiddleware — injects request ID and structured logging
  3. RateLimitMiddleware      — sliding window rate limiter
  4. CORSMiddleware           — cross-origin request headers
  5. GZipMiddleware           — compress responses > 1KB

ROUTE PREFIXES:
  /api/v1/auth        — authentication (register, login, OAuth)
  /api/v1/chat        — chat + conversations (all agents)
  /api/v1/documents   — document upload + search
  /api/v1/gmail       — Gmail agent endpoints
  /api/v1/admin       — admin-only management
  /api/v1/tasks       — task management (CRUD via API)
  /health             — public health check
  /metrics            — Prometheus metrics
  /docs               — Swagger UI (disabled in production)
  /redoc              — ReDoc UI (disabled in production)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.endpoints.admin import admin_router, gmail_router, health_router
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.document import router as documents_router
from app.core.config import get_settings
from app.core.database import close_db, init_db
from app.core.logging import get_logger, setup_logging
from app.core.redis_client import close_redis, init_redis
from app.middleware.error_handler import (
    ErrorHandlerMiddleware,
    http_exception_handler,
    validation_exception_handler,
) 
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_context import RequestContextMiddleware
from app.middleware.telemetry import (
    setup_fastapi_tracing,
    setup_metrics,
    setup_tracing,
)
from app.services.vector_db.weaviate_client import close_weaviate, init_weaviate

# ---------------------------------------------------------------------------
# Setup logging BEFORE anything else
# ---------------------------------------------------------------------------
setup_logging()
settings = get_settings()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup + shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager.
    Code before yield → startup.
    Code after yield  → shutdown.
    """
    logger.info(
        "platform.starting",
        version=settings.app_version,
        env=settings.app_env,
        workers=settings.workers,
    )

    # ── Startup ──────────────────────────────────────────────────────────
    setup_tracing()

    await init_redis()
    logger.info("startup.redis.ready")

    await init_weaviate()
    logger.info("startup.weaviate.ready")

    # In development: auto-create tables. In production: use Alembic migrations.
    if settings.is_development:
        await init_db()
        logger.info("startup.db.tables_created")

    # Ensure default roles exist (idempotent)
    await _seed_roles_if_needed()
    logger.info("startup.roles.verified")

    logger.info("platform.ready", host=settings.host, port=settings.port)

    yield   # ← Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("platform.shutting_down")
    await close_db()
    await close_redis()
    await close_weaviate()
    logger.info("platform.stopped")


async def _seed_roles_if_needed() -> None:
    """
    Ensure default roles exist without running the full seed script.
    Idempotent — safe to call on every startup.
    """
    from sqlalchemy import select
    from app.core.database import AsyncSessionFactory
    from app.models.models import Role

    DEFAULT_ROLES = [
        {"name": "admin",    "description": "Full access", "permissions": {"*": ["*"]}},
        {"name": "operator", "description": "Ops access",  "permissions": {"metrics": ["read"]}},
        {"name": "user",     "description": "Standard user","permissions": {"chat": ["read", "write"]}},
        {"name": "viewer",   "description": "Read only",   "permissions": {"chat": ["read"]}},
    ]

    async with AsyncSessionFactory() as db:
        for role_def in DEFAULT_ROLES:
            result = await db.execute(select(Role).where(Role.name == role_def["name"]))
            if not result.scalar_one_or_none():
                db.add(Role(**role_def))
        await db.commit()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    Returns a fully configured app instance ready to serve requests.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## LLM Agentic Platform

Production-ready multi-agent AI platform with:
- **RAG**: Hybrid search (BM25 + vector) over your documents
- **Gmail Agent**: AI-powered email management
- **Task Agent**: Natural language task management  
- **Router Agent**: Automatic agent selection
- **RBAC**: Role-based access control (admin/operator/user/viewer)
- **Multi-provider**: Anthropic Claude primary, OpenAI GPT fallback
- **Observability**: OpenTelemetry traces + Prometheus metrics
        """,
        lifespan=lifespan,
        # Disable docs in production (security — don't expose API schema publicly)
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    # ── Middleware (order = outermost first) ──────────────────────────────
    # GZip compression — reduces response size by 60–80% for JSON
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # CORS — allow configured origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time", "X-RateLimit-Remaining"],
    )

    # Rate limiting — sliding window per user/IP
    app.add_middleware(RateLimitMiddleware)

    # Request context — injects request_id, access logs, latency
    app.add_middleware(RequestContextMiddleware)

    # Error handler — converts all exceptions to structured JSON
    app.add_middleware(ErrorHandlerMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # ── Routers ───────────────────────────────────────────────────────────
    API_PREFIX = "/api/v1"

    app.include_router(health_router)                                  # /health
    app.include_router(auth_router,      prefix=API_PREFIX)            # /api/v1/auth
    app.include_router(chat_router,      prefix=API_PREFIX)            # /api/v1/chat
    app.include_router(documents_router, prefix=API_PREFIX)            # /api/v1/documents
    app.include_router(gmail_router,     prefix=API_PREFIX)            # /api/v1/gmail
    app.include_router(admin_router,     prefix=API_PREFIX)            # /api/v1/admin

    # ── Prometheus metrics ────────────────────────────────────────────────
    setup_metrics(app)          # Exposes /metrics
    setup_fastapi_tracing(app)  # OTel auto-instrumentation

    logger.info(
        "app.configured",
        routes=len(app.routes),
        middleware_count=len(app.user_middleware),
    )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn/gunicorn)
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Development runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
        workers=1 if settings.is_development else settings.workers,
        # Production: use gunicorn with uvicorn workers instead
        # gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
    )