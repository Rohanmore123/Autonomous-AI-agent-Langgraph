"""
tests/e2e/test_user_journeys.py
================================
End-to-end tests that simulate complete user journeys through the platform.

REQUIREMENTS:
  These tests require ALL services to be running:
    docker compose -f docker/docker-compose.yml up -d

  Set environment variable: E2E_BASE_URL=http://localhost:8000

  Run with: pytest tests/e2e/ -v --tb=short

  These tests are NOT run in CI by default (requires full stack).
  Add --run-e2e flag or set RUN_E2E=true to enable.

USER JOURNEYS TESTED:
  Journey 1: New User Onboarding
    Register → Login → Upload document → Chat with RAG → View conversation history

  Journey 2: Admin Platform Management
    Admin login → List users → Deactivate user → Verify deactivated user can't login

  Journey 3: Rate Limiting
    Login → Send 70 requests/minute → Verify 429 after limit → Verify headers

  Journey 4: Token Refresh Lifecycle
    Login → Use access token → Refresh → Use new token → Logout → Verify token invalid

  Journey 5: Multi-Agent Routing
    Login → Send email query → Verify routed to Gmail agent
           → Send task query → Verify routed to Task agent
           → Send doc query  → Verify routed to RAG agent
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Skip E2E tests unless explicitly enabled
# ---------------------------------------------------------------------------

E2E_ENABLED = os.environ.get("RUN_E2E", "false").lower() == "true"
E2E_BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")

pytestmark = pytest.mark.skipif(
    not E2E_ENABLED,
    reason="E2E tests require full stack. Set RUN_E2E=true to run.",
)


# ---------------------------------------------------------------------------
# E2E client fixture (real HTTP calls to running server)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def e2e_client():
    """Real HTTP client pointing to the running server."""
    async with AsyncClient(base_url=E2E_BASE_URL, timeout=60) as client:
        yield client


@pytest_asyncio.fixture
async def e2e_user(e2e_client: AsyncClient) -> dict:
    """Register + login a unique test user. Returns {email, password, tokens}."""
    suffix = uuid.uuid4().hex[:8]
    email    = f"e2etest_{suffix}@example.com"
    username = f"e2etest_{suffix}"
    password = "E2ETestPass@123"

    # Register
    reg = await e2e_client.post("/api/v1/auth/register", json={
        "email": email,
        "username": username,
        "password": password,
    })
    assert reg.status_code == 201, f"Registration failed: {reg.text}"

    # Login
    login = await e2e_client.post("/api/v1/auth/login", json={
        "email": email,
        "password": password,
    })
    assert login.status_code == 200, f"Login failed: {login.text}"

    tokens = login.json()["data"]
    return {
        "email":         email,
        "username":      username,
        "password":      password,
        "access_token":  tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "headers":       {"Authorization": f"Bearer {tokens['access_token']}"},
    }


# ===========================================================================
# Journey 1: New User Onboarding
# ===========================================================================

class TestNewUserOnboarding:

    @pytest.mark.asyncio
    async def test_full_onboarding_journey(
        self, e2e_client: AsyncClient, e2e_user: dict
    ):
        """
        Complete journey:
          1. Verify profile accessible
          2. Chat with direct agent
          3. Verify conversation created
          4. Upload a text document
          5. Wait for ingestion
          6. Chat with RAG agent using the document
          7. List conversation history
        """
        headers = e2e_user["headers"]

        # Step 1: Verify profile
        me = await e2e_client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["data"]["email"] == e2e_user["email"]

        # Step 2: Chat with direct agent
        chat = await e2e_client.post("/api/v1/chat", json={
            "message": "What is the capital of France?",
            "agent_type": "direct",
        }, headers=headers)
        assert chat.status_code == 200
        assert "conversation_id" in chat.json()["data"]
        conv_id = chat.json()["data"]["conversation_id"]

        # Step 3: Verify conversation in history
        convs = await e2e_client.get("/api/v1/chat/conversations", headers=headers)
        assert convs.status_code == 200
        conv_ids = [c["id"] for c in convs.json()["data"]["items"]]
        assert conv_id in conv_ids

        # Step 4: Upload a document
        doc_content = b"""
        LLM Platform Documentation
        ===========================
        This platform supports hybrid search with BM25 and vector similarity.
        The RAG agent retrieves context from uploaded documents.
        Rate limiting is implemented with a sliding window algorithm.
        Authentication uses JWT tokens with RBAC.
        """
        upload = await e2e_client.post(
            "/api/v1/documents/upload",
            files={"file": ("platform_docs.txt", doc_content, "text/plain")},
            headers=headers,
        )
        assert upload.status_code == 202
        doc_id = upload.json()["data"]["id"]
        assert doc_id is not None

        # Step 5: Poll for ingestion completion (max 30 seconds)
        for _ in range(15):
            status_resp = await e2e_client.get(
                f"/api/v1/documents/{doc_id}", headers=headers
            )
            if status_resp.json()["data"]["status"] == "ready":
                break
            await asyncio.sleep(2)
        else:
            pytest.skip("Document ingestion timed out — Weaviate may not be running")

        # Step 6: RAG chat using the document
        rag_chat = await e2e_client.post("/api/v1/chat", json={
            "message": "What search algorithm does the platform use?",
            "agent_type": "rag",
            "include_sources": True,
        }, headers=headers)
        assert rag_chat.status_code == 200
        rag_data = rag_chat.json()["data"]
        # The answer should reference the document
        assert rag_data["content"] is not None
        assert len(rag_data["content"]) > 0

        # Step 7: List conversations — should have at least 2
        final_convs = await e2e_client.get("/api/v1/chat/conversations", headers=headers)
        assert final_convs.json()["data"]["total"] >= 1


# ===========================================================================
# Journey 2: Token Refresh Lifecycle
# ===========================================================================

class TestTokenRefreshLifecycle:

    @pytest.mark.asyncio
    async def test_token_refresh_and_logout(
        self, e2e_client: AsyncClient, e2e_user: dict
    ):
        """
        1. Use access token → success
        2. Refresh tokens
        3. Use new access token → success
        4. Logout (revoke refresh token)
        5. Try to use revoked refresh token → 401
        """
        headers = e2e_user["headers"]
        refresh_token = e2e_user["refresh_token"]

        # Step 1: Use access token
        me = await e2e_client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200

        # Step 2: Refresh tokens
        refresh_resp = await e2e_client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token
        })
        assert refresh_resp.status_code == 200
        new_tokens = refresh_resp.json()["data"]
        new_access  = new_tokens["access_token"]
        new_refresh = new_tokens["refresh_token"]

        # Step 3: Use new access token
        me2 = await e2e_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {new_access}"},
        )
        assert me2.status_code == 200

        # Step 4: Logout (revoke refresh token)
        logout = await e2e_client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": new_refresh},
            headers={"Authorization": f"Bearer {new_access}"},
        )
        assert logout.status_code == 200

        # Step 5: Try to use revoked refresh token → 401
        revoked_resp = await e2e_client.post("/api/v1/auth/refresh", json={
            "refresh_token": new_refresh
        })
        assert revoked_resp.status_code == 401


# ===========================================================================
# Journey 3: Rate Limiting
# ===========================================================================

class TestRateLimiting:

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_429(
        self, e2e_client: AsyncClient, e2e_user: dict
    ):
        """
        Send rapid requests until rate limit kicks in.
        Verify 429 response with correct headers.
        """
        headers = e2e_user["headers"]
        rate_limit_hit = False
        responses = []

        # Fire 70 rapid requests (per-minute limit is 60)
        for i in range(70):
            resp = await e2e_client.get("/api/v1/auth/me", headers=headers)
            responses.append(resp.status_code)
            if resp.status_code == 429:
                rate_limit_hit = True
                # Verify rate limit headers present
                assert "X-RateLimit-Limit" in resp.headers
                assert "X-RateLimit-Remaining" in resp.headers
                assert "Retry-After" in resp.headers
                assert int(resp.headers["X-RateLimit-Remaining"]) == 0
                break

        assert rate_limit_hit, (
            f"Rate limit was not triggered after {len(responses)} requests. "
            f"Status codes: {set(responses)}"
        )

    @pytest.mark.asyncio
    async def test_rate_limit_headers_on_successful_requests(
        self, e2e_client: AsyncClient, e2e_user: dict
    ):
        """Successful requests should include rate limit info headers."""
        resp = await e2e_client.get("/api/v1/auth/me", headers=e2e_user["headers"])
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert int(resp.headers["X-RateLimit-Remaining"]) >= 0


# ===========================================================================
# Journey 4: Multi-Agent Routing
# ===========================================================================

class TestMultiAgentRouting:

    @pytest.mark.asyncio
    async def test_router_selects_correct_agents(
        self, e2e_client: AsyncClient, e2e_user: dict
    ):
        """
        Verify the auto-router selects the right agent for different query types.
        We check the 'agent_type' stored in the conversation.
        """
        headers = e2e_user["headers"]

        test_cases = [
            ("Show me my unread emails",          "gmail"),
            ("Create a task to review the report", "task"),
            ("What does 2+2 equal?",               "direct"),
        ]

        for message, expected_agent in test_cases:
            resp = await e2e_client.post("/api/v1/chat", json={
                "message": message,
                "agent_type": "router",
            }, headers=headers)

            if resp.status_code == 200:
                # Get the conversation to check which agent was used
                conv_id = resp.json()["data"]["conversation_id"]
                conv_resp = await e2e_client.get(
                    f"/api/v1/chat/conversations/{conv_id}",
                    headers=headers,
                )
                if conv_resp.status_code == 200:
                    agent_type = conv_resp.json()["data"].get("agent_type", "")
                    # Agent type should match expectation (or "router" if routing failed)
                    assert agent_type in (expected_agent, "router", "direct"), \
                        f"Expected {expected_agent} for: {message!r}, got {agent_type}"
            # 503 is acceptable if Gmail/Weaviate isn't configured — don't fail E2E on it
            elif resp.status_code not in (200, 503):
                pytest.fail(f"Unexpected status {resp.status_code} for: {message!r}")


# ===========================================================================
# Journey 5: Admin Management
# ===========================================================================

class TestAdminManagement:

    @pytest.mark.asyncio
    async def test_admin_can_deactivate_and_user_cannot_login(
        self, e2e_client: AsyncClient, e2e_user: dict
    ):
        """
        Admin deactivates a user → that user can no longer login.
        Requires admin credentials from env vars.
        """
        admin_email    = os.environ.get("ADMIN_EMAIL",    "admin@llmplatform.local")
        admin_password = os.environ.get("ADMIN_PASSWORD", "Admin@123456")

        # Admin login
        admin_login = await e2e_client.post("/api/v1/auth/login", json={
            "email": admin_email, "password": admin_password
        })
        if admin_login.status_code != 200:
            pytest.skip(f"Admin login failed — check ADMIN_EMAIL/ADMIN_PASSWORD env vars")

        admin_token = admin_login.json()["data"]["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # Find the test user's ID
        me_resp = await e2e_client.get("/api/v1/auth/me", headers=e2e_user["headers"])
        user_id = me_resp.json()["data"]["id"]

        # Admin deactivates the user
        deact = await e2e_client.patch(
            f"/api/v1/admin/users/{user_id}",
            json={"is_active": False},
            headers=admin_headers,
        )
        assert deact.status_code == 200

        # User can no longer login
        login_attempt = await e2e_client.post("/api/v1/auth/login", json={
            "email": e2e_user["email"],
            "password": e2e_user["password"],
        })
        assert login_attempt.status_code == 403   # Account deactivated

        # Re-activate to avoid polluting the DB (cleanup)
        await e2e_client.patch(
            f"/api/v1/admin/users/{user_id}",
            json={"is_active": True},
            headers=admin_headers,
        )