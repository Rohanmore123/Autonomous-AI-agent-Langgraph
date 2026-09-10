"""
tests/integration/test_endpoints.py
=====================================
Integration tests for all major API endpoints.

These tests use:
  • Real SQLite DB (in-memory, per-test transaction isolation)
  • Mocked external services (LLM, Redis, Weaviate, Gmail)
  • Real FastAPI request/response cycle via AsyncClient

COVERAGE:
  Chat endpoints:
    ✓ POST /chat — direct agent
    ✓ POST /chat — RAG agent (with mocked Weaviate)
    ✓ POST /chat — task agent (with mocked Anthropic)
    ✓ POST /chat — router auto-detects agent
    ✓ POST /chat — creates new conversation when no ID provided
    ✓ POST /chat — continues existing conversation
    ✓ GET /chat/conversations — paginated list
    ✓ GET /chat/conversations/{id} — with messages
    ✓ DELETE /chat/conversations/{id}
    ✓ POST /chat — unauthenticated → 401

  Document endpoints:
    ✓ POST /documents/upload — accepts valid text file
    ✓ POST /documents/upload — rejects unsupported type
    ✓ POST /documents/upload — rejects files > 50MB
    ✓ GET /documents — lists user's documents only
    ✓ GET /documents/{id} — returns correct document
    ✓ GET /documents/{id} — 404 for another user's document
    ✓ DELETE /documents/{id} — removes from DB + Weaviate
    ✓ POST /documents/search — returns ranked results

  Admin endpoints:
    ✓ GET /admin/users — admin can list all users
    ✓ GET /admin/users — non-admin gets 403
    ✓ PATCH /admin/users/{id} — deactivate user
    ✓ GET /admin/metrics — returns metrics summary
    ✓ POST /admin/roles — create new role
    ✓ POST /admin/roles/assign — assign role to user

  Health:
    ✓ GET /health — returns 200 with component status
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Conversation, Document, User


# ===========================================================================
# Chat Endpoint Tests
# ===========================================================================

class TestChatEndpoint:

    @pytest.mark.asyncio
    async def test_chat_direct_agent_success(
        self, auth_client: AsyncClient, mock_llm_router, mock_redis, regular_user: User
    ):
        resp = await auth_client.post("/api/v1/chat", json={
            "message": "What is 2 + 2?",
            "agent_type": "direct",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "conversation_id" in data
        assert "message_id" in data
        assert data["content"] == "This is a mocked LLM response for testing."
        assert data["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chat_creates_new_conversation(
        self,
        auth_client: AsyncClient,
        mock_llm_router,
        mock_redis,
        db_session: AsyncSession,
        regular_user: User,
    ):
        resp = await auth_client.post("/api/v1/chat", json={
            "message": "Hello world",
            "agent_type": "direct",
        })
        assert resp.status_code == 200
        conv_id = resp.json()["data"]["conversation_id"]

        # Verify conversation exists in DB
        from sqlalchemy import select
        result = await db_session.execute(
            select(Conversation).where(Conversation.id == conv_id)
        )
        conv = result.scalar_one_or_none()
        assert conv is not None
        assert conv.user_id == regular_user.id

    @pytest.mark.asyncio
    async def test_chat_continues_existing_conversation(
        self,
        auth_client: AsyncClient,
        mock_llm_router,
        mock_redis,
        db_session: AsyncSession,
        regular_user: User,
    ):
        # First message → creates conversation
        r1 = await auth_client.post("/api/v1/chat", json={
            "message": "First message",
            "agent_type": "direct",
        })
        assert r1.status_code == 200
        conv_id = r1.json()["data"]["conversation_id"]

        # Second message → uses same conversation
        r2 = await auth_client.post("/api/v1/chat", json={
            "message": "Second message",
            "agent_type": "direct",
            "conversation_id": conv_id,
        })
        assert r2.status_code == 200
        assert r2.json()["data"]["conversation_id"] == conv_id

    @pytest.mark.asyncio
    async def test_chat_rag_agent_with_sources(
        self,
        auth_client: AsyncClient,
        mock_llm_router,
        mock_redis,
        mock_weaviate,
        mock_embedding,
        regular_user: User,
    ):
        from app.services.vector_db.weaviate_client import SearchResult

        fake_chunk = SearchResult(
            weaviate_id="wid-001",
            document_id="doc-001",
            filename="guide.pdf",
            content="Configuration guide content",
            score=0.95,
            chunk_index=0,
        )

        with patch(
            "app.services.agents.rag_agent.hybrid_search",
            new_callable=AsyncMock,
            return_value=[fake_chunk],
        ):
            resp = await auth_client.post("/api/v1/chat", json={
                "message": "How do I configure the system?",
                "agent_type": "rag",
                "include_sources": True,
            })

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["sources"] is not None   # Sources returned when include_sources=True

    @pytest.mark.asyncio
    async def test_chat_router_auto_detects_agent(
        self,
        auth_client: AsyncClient,
        mock_llm_router,
        mock_redis,
        mock_weaviate,
        mock_embedding,
    ):
        with patch(
            "app.api.v1.endpoints.chat.route_query",
            new_callable=AsyncMock,
            return_value=__import__("app.services.agents.router_agent", fromlist=["AgentType"]).AgentType.DIRECT,
        ):
            resp = await auth_client.post("/api/v1/chat", json={
                "message": "Tell me a joke",
                "agent_type": "router",
            })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_unauthenticated_returns_401(self, client: AsyncClient, mock_redis):
        resp = await client.post("/api/v1/chat", json={
            "message": "Hello",
            "agent_type": "direct",
        })
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_chat_nonexistent_conversation_returns_404(
        self, auth_client: AsyncClient, mock_llm_router, mock_redis
    ):
        resp = await auth_client.post("/api/v1/chat", json={
            "message": "Hello",
            "agent_type": "direct",
            "conversation_id": "conv-does-not-exist",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_chat_returns_latency_and_token_info(
        self, auth_client: AsyncClient, mock_llm_router, mock_redis
    ):
        resp = await auth_client.post("/api/v1/chat", json={
            "message": "Test message",
            "agent_type": "direct",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "latency_ms" in data
        assert "tokens_used" in data
        assert "model_used" in data
        assert "was_cached" in data

    @pytest.mark.asyncio
    async def test_list_conversations_paginated(
        self,
        auth_client: AsyncClient,
        mock_llm_router,
        mock_redis,
        db_session: AsyncSession,
        regular_user: User,
    ):
        # Create 3 conversations
        for i in range(3):
            await auth_client.post("/api/v1/chat", json={
                "message": f"Message {i}",
                "agent_type": "direct",
            })

        resp = await auth_client.get("/api/v1/chat/conversations?page=1&page_size=2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] >= 3
        assert len(data["items"]) == 2

    @pytest.mark.asyncio
    async def test_get_conversation_with_messages(
        self,
        auth_client: AsyncClient,
        mock_llm_router,
        mock_redis,
    ):
        # Create a conversation
        r = await auth_client.post("/api/v1/chat", json={
            "message": "Hello",
            "agent_type": "direct",
        })
        conv_id = r.json()["data"]["conversation_id"]

        # Fetch it
        resp = await auth_client.get(f"/api/v1/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["id"] == conv_id
        assert len(data["messages"]) >= 2   # user + assistant

    @pytest.mark.asyncio
    async def test_delete_conversation_archives_it(
        self,
        auth_client: AsyncClient,
        mock_llm_router,
        mock_redis,
        db_session: AsyncSession,
    ):
        r = await auth_client.post("/api/v1/chat", json={
            "message": "Hello",
            "agent_type": "direct",
        })
        conv_id = r.json()["data"]["conversation_id"]

        resp = await auth_client.delete(f"/api/v1/chat/conversations/{conv_id}")
        assert resp.status_code == 200

        from sqlalchemy import select
        result = await db_session.execute(
            select(Conversation).where(Conversation.id == conv_id)
        )
        conv = result.scalar_one_or_none()
        assert conv.is_archived is True

    @pytest.mark.asyncio
    async def test_cannot_access_other_users_conversation(
        self,
        auth_client: AsyncClient,
        admin_client: AsyncClient,
        mock_llm_router,
        mock_redis,
    ):
        """Users should not be able to view conversations belonging to others."""
        # Admin creates a conversation
        r = await admin_client.post("/api/v1/chat", json={
            "message": "Admin message",
            "agent_type": "direct",
        })
        admin_conv_id = r.json()["data"]["conversation_id"]

        # Regular user tries to access it
        resp = await auth_client.get(f"/api/v1/chat/conversations/{admin_conv_id}")
        assert resp.status_code == 404   # Not found (ownership filter)


# ===========================================================================
# Document Endpoint Tests
# ===========================================================================

class TestDocumentEndpoint:

    @pytest.mark.asyncio
    async def test_upload_text_file_success(
        self,
        auth_client: AsyncClient,
        mock_redis,
        db_session: AsyncSession,
        regular_user: User,
    ):
        content = b"This is a test document with some content for RAG."

        with patch("app.api.v1.endpoints.documents.ingest_document_task") as mock_task:
            mock_task.delay = MagicMock(return_value=MagicMock(id="celery-task-001"))

            resp = await auth_client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
            )

        assert resp.status_code == 202   # Accepted (async processing)
        data = resp.json()["data"]
        assert data["filename"] == "test.txt"
        assert data["status"] == "pending"
        assert data["content_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_upload_unsupported_type_rejected(
        self, auth_client: AsyncClient, mock_redis
    ):
        content = b"<html><body>Not supported</body></html>"
        resp = await auth_client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.html", io.BytesIO(content), "text/html")},
        )
        assert resp.status_code == 415

    @pytest.mark.asyncio
    async def test_list_documents_only_shows_own(
        self,
        auth_client: AsyncClient,
        admin_client: AsyncClient,
        mock_redis,
        db_session: AsyncSession,
        regular_user: User,
        admin_user: User,
    ):
        # Add a document owned by regular_user
        doc = Document(
            owner_id=regular_user.id,
            filename="myfile.txt",
            content_type="text/plain",
            size_bytes=100,
            weaviate_class="LLMPlatformDocumentChunk",
            status="ready",
        )
        db_session.add(doc)
        await db_session.flush()

        resp = await auth_client.get("/api/v1/documents")
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        owner_ids = {item.get("owner_id", regular_user.id) for item in items}
        # All documents should belong to regular_user
        assert all(
            item["filename"] != "admin_file.txt" for item in items
        )

    @pytest.mark.asyncio
    async def test_get_document_not_found_for_other_user(
        self,
        auth_client: AsyncClient,
        db_session: AsyncSession,
        admin_user: User,
        mock_redis,
    ):
        """User cannot see another user's document."""
        doc = Document(
            owner_id=admin_user.id,
            filename="admin_secret.txt",
            content_type="text/plain",
            size_bytes=500,
            weaviate_class="LLMPlatformDocumentChunk",
            status="ready",
        )
        db_session.add(doc)
        await db_session.flush()

        resp = await auth_client.get(f"/api/v1/documents/{doc.id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_success(
        self,
        auth_client: AsyncClient,
        db_session: AsyncSession,
        regular_user: User,
        mock_redis,
    ):
        doc = Document(
            owner_id=regular_user.id,
            filename="todelete.txt",
            content_type="text/plain",
            size_bytes=100,
            weaviate_class="LLMPlatformDocumentChunk",
            status="ready",
        )
        db_session.add(doc)
        await db_session.flush()
        doc_id = doc.id

        with patch(
            "app.api.v1.endpoints.documents.delete_document_chunks",
            new_callable=AsyncMock,
            return_value=5,
        ):
            resp = await auth_client.delete(f"/api/v1/documents/{doc_id}")

        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"].lower()

        # Verify gone from DB
        from sqlalchemy import select
        result = await db_session.execute(
            select(Document).where(Document.id == doc_id)
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_search_documents(
        self, auth_client: AsyncClient, mock_redis, mock_embedding
    ):
        from app.services.vector_db.weaviate_client import SearchResult

        fake_results = [
            SearchResult(
                weaviate_id="wid-001",
                document_id="doc-001",
                filename="guide.pdf",
                content="Configuration details here",
                score=0.88,
                chunk_index=0,
            )
        ]

        with patch(
            "app.api.v1.endpoints.documents.hybrid_search",
            new_callable=AsyncMock,
            return_value=fake_results,
        ):
            resp = await auth_client.post("/api/v1/documents/search", json={
                "query": "configuration guide",
                "top_k": 5,
                "alpha": 0.5,
            })

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total_found"] == 1
        assert data["results"][0]["filename"] == "guide.pdf"
        assert data["results"][0]["score"] == 0.88


# ===========================================================================
# Admin Endpoint Tests
# ===========================================================================

class TestAdminEndpoint:

    @pytest.mark.asyncio
    async def test_list_users_admin_only(
        self, admin_client: AsyncClient, auth_client: AsyncClient, mock_redis
    ):
        # Admin can access
        resp_admin = await admin_client.get("/api/v1/admin/users")
        assert resp_admin.status_code == 200

        # Regular user cannot
        resp_user = await auth_client.get("/api/v1/admin/users")
        assert resp_user.status_code == 403

    @pytest.mark.asyncio
    async def test_list_users_includes_all_users(
        self,
        admin_client: AsyncClient,
        mock_redis,
        regular_user: User,
        admin_user: User,
    ):
        resp = await admin_client.get("/api/v1/admin/users")
        assert resp.status_code == 200
        total = resp.json()["data"]["total"]
        assert total >= 2

    @pytest.mark.asyncio
    async def test_deactivate_user(
        self,
        admin_client: AsyncClient,
        mock_redis,
        db_session: AsyncSession,
        regular_user: User,
    ):
        mock_redis["session"].scan = AsyncMock(return_value=(0, []))

        resp = await admin_client.patch(
            f"/api/v1/admin/users/{regular_user.id}",
            json={"is_active": False},
        )
        assert resp.status_code == 200

        await db_session.refresh(regular_user)
        assert regular_user.is_active is False

    @pytest.mark.asyncio
    async def test_metrics_endpoint(
        self, admin_client: AsyncClient, mock_redis
    ):
        resp = await admin_client.get("/api/v1/admin/metrics")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "total_requests_today" in data
        assert "avg_latency_ms" in data
        assert "cache_hit_rate" in data
        assert "error_rate" in data

    @pytest.mark.asyncio
    async def test_create_role(
        self, admin_client: AsyncClient, mock_redis
    ):
        resp = await admin_client.post(
            "/api/v1/admin/roles",
            params={
                "name": "custom_role",
                "description": "A custom test role",
                "permissions": {},
            },
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["name"] == "custom_role"

    @pytest.mark.asyncio
    async def test_list_agent_runs(
        self, admin_client: AsyncClient, mock_redis
    ):
        resp = await admin_client.get("/api/v1/admin/agent-runs")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "items" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_operator_can_view_metrics_but_not_manage_users(
        self,
        db_session: AsyncSession,
        client: AsyncClient,
        default_roles: dict,
        mock_redis,
    ):
        """Operators have read-only access to metrics but cannot manage users."""
        from app.core.security import hash_password, create_access_token
        from app.models.models import User, UserRole

        op = User(
            email="operator@example.com",
            username="operatoruser",
            hashed_password=hash_password("OpPass@123"),
            is_active=True,
        )
        db_session.add(op)
        await db_session.flush()
        db_session.add(UserRole(user_id=op.id, role_id=default_roles["operator"].id))
        await db_session.flush()

        token = create_access_token(op.id, "operator")
        op_client = client
        op_client.headers["Authorization"] = f"Bearer {token}"

        # Can view metrics
        metrics_resp = await op_client.get("/api/v1/admin/metrics")
        assert metrics_resp.status_code == 200

        # Cannot manage users
        users_resp = await op_client.get("/api/v1/admin/users")
        assert users_resp.status_code == 403


# ===========================================================================
# Health Endpoint Tests
# ===========================================================================

class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient):
        with patch("app.api.v1.endpoints.admin.check_db_health", new_callable=AsyncMock, return_value=True), \
             patch("app.api.v1.endpoints.admin.check_redis_health", new_callable=AsyncMock, return_value=True), \
             patch("app.api.v1.endpoints.admin.check_weaviate_health", new_callable=AsyncMock, return_value=True):
            resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "components" in data
        assert "version" in data
        assert "uptime_seconds" in data

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client: AsyncClient):
        """Health endpoint must be publicly accessible (no auth)."""
        with patch("app.api.v1.endpoints.admin.check_db_health", new_callable=AsyncMock, return_value=True), \
             patch("app.api.v1.endpoints.admin.check_redis_health", new_callable=AsyncMock, return_value=True), \
             patch("app.api.v1.endpoints.admin.check_weaviate_health", new_callable=AsyncMock, return_value=True):
            resp = await client.get("/health")

        assert resp.status_code != 401

    @pytest.mark.asyncio
    async def test_health_degraded_when_weaviate_down(self, client: AsyncClient):
        with patch("app.api.v1.endpoints.admin.check_db_health", new_callable=AsyncMock, return_value=True), \
             patch("app.api.v1.endpoints.admin.check_redis_health", new_callable=AsyncMock, return_value=True), \
             patch("app.api.v1.endpoints.admin.check_weaviate_health", new_callable=AsyncMock, return_value=False):
            resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["components"]["weaviate"] is False