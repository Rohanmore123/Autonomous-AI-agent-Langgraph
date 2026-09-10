"""
app/core/database.py
====================
Async PostgreSQL engine and session management via SQLAlchemy 2.0.

KEY DESIGN DECISIONS:
  • AsyncEngine + AsyncSession — non-blocking DB I/O; critical for 10K concurrent users.
    Blocking DB calls would stall the event loop and tank throughput.
  • Connection pool (pool_size=20, max_overflow=40) — 60 connections per worker.
    With 4 workers → 240 total connections to Postgres (tune pg_max_connections).
  • pool_pre_ping=True — tests connections before use; prevents "connection closed" errors
    after network hiccups or Postgres restarts.
  • Session-per-request pattern via FastAPI Depends — each HTTP request gets its
    own session, automatically closed after the response.
  • expire_on_commit=False — avoids lazy-load errors after commit in async context.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_async_engine(
    settings.db.url,
    pool_size=settings.db.pool_size,
    max_overflow=settings.db.max_overflow,
    pool_timeout=settings.db.pool_timeout,
    pool_recycle=settings.db.pool_recycle,
    pool_pre_ping=True,           # Heartbeat check before handing connection to code
    echo=settings.db.echo,        # SQL logging (dev only)
    future=True,                  # SQLAlchemy 2.0 style
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionFactory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,    # Don't expire ORM objects after commit (safe in async)
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Base ORM class
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """
    All ORM models inherit from this.
    Provides type-aware column declarations (SQLAlchemy 2.0 Mapped[]).
    """
    pass


# ---------------------------------------------------------------------------
# Dependency — one session per request
# ---------------------------------------------------------------------------

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session scoped to a request.

    Pattern:
        async def my_endpoint(db: AsyncSession = Depends(get_db_session)):
            ...

    The session is:
      - Committed automatically if the handler returns without error.
      - Rolled back automatically on any exception.
      - Closed always (via finally), returning the connection to the pool.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Health-check helper
# ---------------------------------------------------------------------------

async def check_db_health() -> bool:
    """Ping the database; used by /health endpoint."""
    try:
        async with AsyncSessionFactory() as session:
            await session.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("db.health_check.failed", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Lifecycle helpers (called from app lifespan)
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create tables if they don't exist (dev/test only; use Alembic in prod)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db.initialized")


async def close_db() -> None:
    """Dispose the engine connection pool on shutdown."""
    await engine.dispose()
    logger.info("db.closed")