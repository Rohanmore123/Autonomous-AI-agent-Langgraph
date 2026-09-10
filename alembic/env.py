"""
alembic/env.py
===============
Alembic migration environment — wires the migration engine to our
async SQLAlchemy setup and ORM models.

WHY ASYNC ALEMBIC?
  Our app uses AsyncEngine (asyncpg). Alembic's default env.py uses sync
  connections. We use the async-compatible pattern here so both the app
  and migrations talk to the same database driver.

  Pattern: run_async_migrations() wraps the sync migration logic in
  asyncio.run() so Alembic (which is sync) can drive an async engine.

AUTOGENERATE:
  Alembic compares target_metadata (our ORM models) against the live DB.
  It generates ADD/DROP COLUMN, CREATE/DROP INDEX, etc. automatically.
  Always REVIEW generated migrations before applying — autogenerate is
  smart but not perfect (e.g. it can't detect column type changes well).

  What autogenerate detects:
    ✓ New tables
    ✓ Dropped tables
    ✓ New columns
    ✓ Dropped columns
    ✓ Column type changes (most cases)
    ✓ New indexes / unique constraints
    ✗ Server default changes (manual)
    ✗ Stored procedures (manual)
    ✗ Sequence changes (manual)

WORKFLOW:
  1. Change app/models/models.py (add/modify ORM class)
  2. alembic revision --autogenerate -m "descriptive_message"
  3. Review alembic/versions/<timestamp>_<hash>_<slug>.py
  4. alembic upgrade head
  5. Commit BOTH the model change AND the migration file
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ---------------------------------------------------------------------------
# Load our app config and models
# ---------------------------------------------------------------------------

# Add project root to sys.path so `from app.X import Y` works in this script
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all models so Alembic sees them via Base.metadata
# IMPORTANT: Every new model file must be imported here
from app.models.models import (  # noqa: F401 — imports register models with Base
    Base,
    Role,
    User,
    UserRole,
    Conversation,
    Message,
    Document,
    DocumentChunk,
    Task,
    AgentRun,
    APIKey,
    AuditLog,
    GoogleOAuthToken,
)

# ---------------------------------------------------------------------------
# Alembic config object
# ---------------------------------------------------------------------------

config = context.config

# Interpret alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata object Alembic compares against the live DB
target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Database URL — prefer environment variable over alembic.ini
# ---------------------------------------------------------------------------

def get_url() -> str:
    """
    Read the database URL from environment (preferred) or alembic.ini.

    We convert asyncpg URLs to regular psycopg2 for Alembic's sync connection,
    and back to asyncpg for the async migration runner.
    """
    url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url", "")
    # Alembic's async runner needs asyncpg; ensure the right driver is set
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    return url


# ---------------------------------------------------------------------------
# Offline mode — generates SQL without connecting to DB
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Generates SQL statements to stdout/file without connecting to DB.
    Useful for DBAs who review SQL before applying.

    Usage: alembic upgrade head --sql > migration.sql
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,              # Detect column type changes
        compare_server_default=True,    # Detect server default changes
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode — connects to DB and applies migrations
# ---------------------------------------------------------------------------

def do_run_migrations(connection: Connection) -> None:
    """
    Core migration runner — called from both sync and async paths.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        # Render AS BATCH mode for SQLite (no-op on PostgreSQL)
        render_as_batch=False,
        # Transaction per migration (safer than one big transaction)
        transaction_per_migration=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations using an async engine.
    asyncpg does not support sync connection — we use the async engine
    and run the sync migration logic via run_sync().
    """
    # Build the async engine from our config
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,    # Don't pool during migrations
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — wraps async runner."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Dispatch based on mode
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()