"""
tests/conftest.py
==================
Shared pytest fixtures for the entire test suite.

TESTING STRATEGY:
  Unit tests      → mock all external dependencies (DB, Redis, LLM, Weaviate)
  Integration tests → use real DB (test schema), mock LLM + Weaviate
  E2E tests       → full stack with all services running

FIXTURE SCOPES:
  session  — created once per test session (expensive: DB engine, event loop)
  module   — once per test file (moderate: seeded DB state)
  function — once per test function (cheap: per-test DB transactions)

DATABASE ISOLATION:
  Each test function runs inside a transaction that is ROLLED BACK at the end.
  This means tests never pollute each other's data and the DB is always clean.
  Technique: begin transaction → run test → rollback (never commit).

ASYNC TESTING:
  All FastAPI endpoints are async. We use pytest-asyncio with
  asyncio_mode="auto" so every async test function is automatically awaited.
  The event loop is shared at session scope for performance.

MOCKING STRATEGY:
  External services (LLM APIs, Redis, Weaviate) are mocked with pytest-mock
  and unittest.mock to avoid:
    • Cost (LLM API charges per token)
    • Flakiness (network failures break tests)
    • Speed (LLM calls take 2–10 seconds each)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.core.database import Base, get_db_session
from app.core.security import create_access_token, hash_password
from app.models.models import Role, User, UserRole
from app.main import create_app

settings = get_settings()

# ---------------------------------------------------------------------------
# Test database — SQLite in-memory for speed (no PostgreSQL required for unit tests)
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """
    Session-scoped event loop.
    pytest-asyncio requires this for session/module-scoped async fixtures.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """
    Create the test database engine once per session.
    SQLite in-memory is used for unit/integration tests — no external DB needed.
    For E2E tests, use a real PostgreSQL test database.
    """
    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def session_factory(test_engine):
    """Session-scoped async session factory."""
    return async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def db_session(session_factory) -> AsyncGenerator[AsyncSession, None]:
    """
    Per-test database session with automatic rollback.

    HOW TRANSACTION ISOLATION WORKS:
      1. Begin a "savepoint" transaction
      2. Run the test (all DB writes go into this transaction)
      3. Rollback to the savepoint → all changes vanish
      4. Next test starts with a clean slate

    This is 10x faster than truncating tables between tests.
    """
    async with session_factory() as session:
        async with session.begin():
            yield session
            await session.rollback()


# ---------------------------------------------------------------------------
# Seed data fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def default_roles(db_session: AsyncSession) -> dict[str, Role]:
    """Create default roles for tests that need RBAC."""
    roles = {}
    for name, perms in [
        ("admin",   {"*": ["*"]}),
        ("user",    {"chat": ["read", "write"]}),
        ("viewer",  {"chat": ["read"]}),
        ("operator",{"metrics": ["read"]}),
    ]:
        role = Role(name=name, description=f"{name} role", permissions=perms)
        db_session.add(role)
        roles[name] = role

    await db_session.flush()
    return roles


@pytest_asyncio.fixture
async def regular_user(db_session: AsyncSession, default_roles: dict) -> User:
    """A standard (non-admin) user for testing."""
    user = User(
        email="testuser@example.com",
        username="testuser",
        hashed_password=hash_password("TestPass@123"),
        full_name="Test User",
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.flush()

    db_session.add(UserRole(user_id=user.id, role_id=default_roles["user"].id))
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession, default_roles: dict) -> User:
    """An admin user for testing admin-only endpoints."""
    user = User(
        email="admin@example.com",
        username="admin",
        hashed_password=hash_password("AdminPass@123"),
        full_name="Admin User",
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.flush()

    db_session.add(UserRole(user_id=user.id, role_id=default_roles["admin"].id))
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def inactive_user(db_session: AsyncSession, default_roles: dict) -> User:
    """A deactivated user — should fail auth checks."""
    user = User(
        email="inactive@example.com",
        username="inactive",
        hashed_password=hash_password("InactivePass@123"),
        is_active=False,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.flush()
    db_session.add(UserRole(user_id=user.id, role_id=default_roles["user"].id))
    await db_session.flush()
    return user


# ---------------------------------------------------------------------------
# JWT token fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def user_token(regular_user: User) -> str:
    """Valid JWT access token for the regular_user fixture."""
    return create_access_token(
        user_id=regular_user.id,
        role="user",
    )


@pytest.fixture
def admin_token(admin_user: User) -> str:
    """Valid JWT access token for the admin_user fixture."""
    return create_access_token(
        user_id=admin_user.id,
        role="admin",
    )


@pytest.fixture
def auth_headers(user_token: str) -> dict[str, str]:
    """Authorization headers for regular user."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def admin_auth_headers(admin_token: str) -> dict[str, str]:
    """Authorization headers for admin user."""
    return {"Authorization": f"Bearer {admin_token}"}


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_app(db_session: AsyncSession) -> FastAPI:
    """
    FastAPI app with the DB dependency overridden to use the test session.
    This ensures every endpoint uses our isolated, rollback-safe DB session.
    """
    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db
    return app


@pytest_asyncio.fixture
async def client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP client for testing FastAPI endpoints.
    Uses ASGITransport — no real network calls, everything in-process.
    """
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://testserver",
        headers={"Content-Type": "application/json"},
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_client(client: AsyncClient, auth_headers: dict) -> AsyncClient:
    """Pre-authenticated client for the regular user."""
    client.headers.update(auth_headers)
    return client


@pytest_asyncio.fixture
async def admin_client(client: AsyncClient, admin_auth_headers: dict) -> AsyncClient:
    """Pre-authenticated client for the admin user."""
    client.headers.update(admin_auth_headers)
    return client


# ---------------------------------------------------------------------------
# External service mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_router():
    """
    Mock LLMRouter.complete() to return a fake response instantly.
    Prevents real LLM API calls (cost + latency) during tests.
    """
    from app.services.llm.router import LLMResponse

    fake_response = LLMResponse(
        content="This is a mocked LLM response for testing.",
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_tokens=50,
        output_tokens=20,
        latency_ms=150.0,
        was_cached=False,
        used_fallback=False,
    )

    with patch("app.services.llm.router.llm_router") as mock:
        mock.complete = AsyncMock(return_value=fake_response)
        mock.stream = AsyncMock(return_value=iter(["mocked ", "response"]))
        yield mock


@pytest.fixture
def mock_redis():
    """
    Mock Redis client — prevents real Redis calls in unit tests.
    Returns sensible defaults (cache miss, rate limit not exceeded).
    """
    with patch("app.core.redis_client._session_pool") as session_mock, \
         patch("app.core.redis_client._cache_pool") as cache_mock, \
         patch("app.core.redis_client._rate_limit_pool") as rate_mock:

        for mock in [session_mock, cache_mock, rate_mock]:
            mock.get = AsyncMock(return_value=None)      # Cache miss by default
            mock.set = AsyncMock(return_value=True)
            mock.setex = AsyncMock(return_value=True)
            mock.delete = AsyncMock(return_value=1)
            mock.exists = AsyncMock(return_value=0)
            mock.ping = AsyncMock(return_value=True)
            mock.zcard = AsyncMock(return_value=0)        # Rate limit: 0 requests
            mock.zremrangebyscore = AsyncMock(return_value=0)
            mock.zadd = AsyncMock(return_value=1)
            mock.expire = AsyncMock(return_value=True)
            mock.mget = AsyncMock(return_value=[None])
            mock.pipeline = MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=MagicMock(
                    execute=AsyncMock(return_value=[0, 0, 1, True])
                )),
                __aexit__=AsyncMock(return_value=False),
            ))

        yield {
            "session": session_mock,
            "cache": cache_mock,
            "rate_limit": rate_mock,
        }


@pytest.fixture
def mock_weaviate():
    """
    Mock Weaviate client — prevents real vector DB calls in unit tests.
    Returns empty search results by default.
    """
    with patch("app.services.vector_db.weaviate_client.get_weaviate_client") as mock:
        client = MagicMock()
        client.is_live.return_value = True
        client.collections.exists.return_value = True

        # Mock search results
        mock_result = MagicMock()
        mock_result.objects = []
        client.collections.get.return_value.query.hybrid.return_value = mock_result
        client.collections.get.return_value.data.insert_many.return_value = MagicMock(
            has_errors=False
        )

        mock.return_value = client
        yield client


@pytest.fixture
def mock_embedding():
    """Mock embedding model — returns a fixed 384-dim vector."""
    fake_vector = [0.1] * 384

    with patch("app.services.vector_db.weaviate_client.embed_text") as mock_text, \
         patch("app.services.vector_db.weaviate_client.embed_batch") as mock_batch, \
         patch("app.services.llm.embeddings.EmbeddingService._get_model") as mock_model:

        mock_text.return_value = fake_vector
        mock_batch.return_value = [fake_vector]

        model_instance = MagicMock()
        import numpy as np
        model_instance.encode.return_value = np.array([fake_vector])
        mock_model.return_value = model_instance

        yield {
            "vector": fake_vector,
            "embed_text": mock_text,
            "embed_batch": mock_batch,
        }


@pytest.fixture
def mock_gmail():
    """Mock Google Gmail API — prevents real email calls."""
    with patch("app.services.agents.gmail_agent._get_user_credentials") as mock_creds, \
         patch("app.services.agents.gmail_agent._build_gmail_service") as mock_service:

        mock_creds.return_value = MagicMock()
        service = MagicMock()
        service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": [{"id": "msg-001", "threadId": "thread-001"}]
        }
        service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
            "id": "msg-001",
            "threadId": "thread-001",
            "snippet": "Test email snippet",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "user@example.com"},
                    {"name": "Date", "value": "Mon, 01 Jan 2025 10:00:00 +0000"},
                ]
            },
        }
        mock_service.return_value = service
        yield service


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

class UserFactory:
    """Create test users programmatically."""

    @staticmethod
    async def create(
        db: AsyncSession,
        email: str = "test@example.com",
        username: str = "testuser",
        password: str = "TestPass@123",
        is_active: bool = True,
        is_verified: bool = True,
        role_name: str = "user",
    ) -> User:
        user = User(
            email=email,
            username=username,
            hashed_password=hash_password(password),
            is_active=is_active,
            is_verified=is_verified,
        )
        db.add(user)
        await db.flush()

        role_result = await db.execute(
            __import__("sqlalchemy").select(Role).where(Role.name == role_name)
        )
        role = role_result.scalar_one_or_none()
        if role:
            db.add(UserRole(user_id=user.id, role_id=role.id))
            await db.flush()

        return user


@pytest.fixture
def user_factory():
    return UserFactory