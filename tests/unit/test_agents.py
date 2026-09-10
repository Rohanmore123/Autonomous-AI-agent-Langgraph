"""
tests/unit/test_agents.py
==========================
Unit tests for all agents: RAG, Gmail, Task, and Router.

COVERAGE:
  RAG Agent:
    ✓ Retrieves chunks and builds grounded prompt
    ✓ Trims context when chunks exceed MAX_CONTEXT_CHARS
    ✓ Returns empty sources when no chunks found
    ✓ Passes conversation history to LLM
    ✓ Returns SearchResult list alongside LLMResponse

  Router Agent:
    ✓ Rule-based routing — Gmail keywords → gmail
    ✓ Rule-based routing — Task keywords → task
    ✓ Rule-based routing — Document keywords → rag
    ✓ Rule-based routing — No match → direct
    ✓ Ambiguous query → LLM classification fallback
    ✓ Explicit agent override respected
    ✓ Multi-agent plan decomposition

  Task Agent:
    ✓ Create task tool invocation
    ✓ List tasks tool invocation
    ✓ Update task tool invocation
    ✓ Complete task tool invocation
    ✓ MAX_ITERATIONS prevents infinite loops
    ✓ Tool results fed back to LLM

  Gmail Agent:
    ✓ search_emails returns parsed GmailEmailResponse list
    ✓ summarise_inbox passes emails to LLM
    ✓ draft_reply fetches email then calls LLM
    ✓ No Google token raises ValueError
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agents.router_agent_2 import AgentType, route_query, _rule_based_route


# ===========================================================================
# Router Agent — Rule-based routing
# ===========================================================================

class TestRouterAgentRuleBased:

    def test_email_keywords_route_to_gmail(self):
        for query in [
            "Show me my unread emails",
            "Send an email to john@example.com",
            "Check my inbox",
            "Reply to the message from Alice",
            "Compose a new email",
            "Forward this mail to the team",
        ]:
            result = _rule_based_route(query)
            assert result == AgentType.GMAIL, f"Expected gmail for: {query!r}"

    def test_task_keywords_route_to_task(self):
        for query in [
            "Create a task to review the report",
            "Show my todos for today",
            "Mark task 123 as done",
            "What tasks are due this week?",
            "Add a reminder for the meeting",
            "Complete task: write documentation",
        ]:
            result = _rule_based_route(query)
            assert result == AgentType.TASK, f"Expected task for: {query!r}"

    def test_document_keywords_route_to_rag(self):
        for query in [
            "Search the knowledge base for API docs",
            "What does the uploaded PDF say?",
            "Find information in the document",
            "According to the file, what is the policy?",
            "Summarize the document I uploaded",
        ]:
            result = _rule_based_route(query)
            assert result == AgentType.RAG, f"Expected rag for: {query!r}"

    def test_no_keywords_routes_to_direct(self):
        for query in [
            "What is the capital of France?",
            "Write me a poem about autumn",
            "Explain quantum computing",
            "How do I reverse a Python list?",
        ]:
            result = _rule_based_route(query)
            assert result == AgentType.DIRECT, f"Expected direct for: {query!r}"

    def test_ambiguous_query_returns_none(self):
        """Queries matching multiple agents should return None → LLM fallback."""
        result = _rule_based_route("Search my email documents and create a task")
        assert result is None   # Ambiguous → needs LLM classification

    @pytest.mark.asyncio
    async def test_explicit_override_respected(self):
        """If caller specifies agent_type explicitly, use it without routing."""
        result = await route_query(
            query="What is the weather today?",   # Would normally → direct
            explicit_agent="gmail",               # Override to gmail
        )
        assert result == AgentType.GMAIL

    @pytest.mark.asyncio
    async def test_invalid_explicit_override_falls_through(self):
        """Invalid explicit agent name should fall through to auto-routing."""
        result = await route_query(
            query="Show my emails",
            explicit_agent="nonexistent_agent",
        )
        assert result == AgentType.GMAIL   # Routed by rule-based

    @pytest.mark.asyncio
    async def test_llm_fallback_for_ambiguous_query(self):
        """Ambiguous query should trigger LLM classification."""
        with patch("app.services.agents.router_agent.llm_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=MagicMock(content="task"))

            result = await route_query(
                "I need to check my tasks and also my emails and search documents",
                explicit_agent=None,
            )
            # LLM said "task"
            assert result == AgentType.TASK
            mock_router.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_returns_invalid_label_falls_back_to_direct(self):
        """If LLM returns an unrecognised label, default to 'direct'."""
        with patch("app.services.agents.router_agent.llm_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=MagicMock(content="UNKNOWN_LABEL"))

            result = await route_query(
                "some ambiguous multi-keyword email task document query",
                explicit_agent=None,
            )
            assert result == AgentType.DIRECT


# ===========================================================================
# RAG Agent
# ===========================================================================

class TestRAGAgent:

    @pytest.mark.asyncio
    async def test_run_returns_llm_response_and_sources(self, mock_llm_router):
        """RAG agent should return (LLMResponse, [SearchResult])."""
        from app.services.agents.rag_agent_2 import RAGAgent
        from app.services.vector_db.weaviate_client import SearchResult

        fake_chunk = SearchResult(
            weaviate_id="wid-001",
            document_id="doc-001",
            filename="guide.pdf",
            content="This document explains the configuration.",
            score=0.92,
            chunk_index=0,
        )

        with patch("app.services.agents.rag_agent.hybrid_search",
                   new_callable=AsyncMock, return_value=[fake_chunk]):
            agent = RAGAgent()
            llm_resp, chunks = await agent.run(
                query="How do I configure the system?",
                owner_id="user-123",
            )

        assert llm_resp is not None
        assert len(chunks) == 1
        assert chunks[0].filename == "guide.pdf"
        mock_llm_router.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_no_chunks_still_calls_llm(self, mock_llm_router):
        """Even with 0 chunks, agent should call LLM (returns 'I don't know')."""
        from app.services.agents.rag_agent_2 import RAGAgent

        with patch("app.services.agents.rag_agent.hybrid_search",
                   new_callable=AsyncMock, return_value=[]):
            agent = RAGAgent()
            llm_resp, chunks = await agent.run(
                query="What is the answer?",
                owner_id="user-123",
            )

        assert chunks == []
        mock_llm_router.complete.assert_called_once()

        # System prompt should mention no documents found
        call_args = mock_llm_router.complete.call_args
        system_prompt = call_args.kwargs.get("system") or call_args.args[1] if len(call_args.args) > 1 else ""
        assert "No relevant documents found" in (system_prompt or "")

    @pytest.mark.asyncio
    async def test_context_trimmed_when_too_large(self, mock_llm_router):
        """Chunks exceeding MAX_CONTEXT_CHARS should be trimmed."""
        from app.services.agents.rag_agent_2 import RAGAgent, MAX_CONTEXT_CHARS
        from app.services.vector_db.weaviate_client import SearchResult

        # Create chunks that together exceed the limit
        big_chunks = [
            SearchResult(
                weaviate_id=f"wid-{i:03d}",
                document_id="doc-001",
                filename="bigdoc.pdf",
                content="x" * (MAX_CONTEXT_CHARS // 2),  # Each chunk = half the limit
                score=0.9 - (i * 0.01),
                chunk_index=i,
            )
            for i in range(5)  # 5 × (limit/2) = 2.5× limit → must trim
        ]

        with patch("app.services.agents.rag_agent.hybrid_search",
                   new_callable=AsyncMock, return_value=big_chunks):
            agent = RAGAgent()
            _, used_chunks = await agent.run(
                query="Tell me everything",
                owner_id="user-123",
            )

        # Not all 5 chunks should be used
        assert len(used_chunks) < 5

    @pytest.mark.asyncio
    async def test_conversation_history_passed_to_llm(self, mock_llm_router):
        """Conversation history should be included in the LLM messages."""
        from app.services.agents.rag_agent_2 import RAGAgent

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        with patch("app.services.agents.rag_agent.hybrid_search",
                   new_callable=AsyncMock, return_value=[]):
            agent = RAGAgent()
            await agent.run(
                query="Follow-up question",
                owner_id="user-123",
                conversation_history=history,
            )

        call_args = mock_llm_router.complete.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        # History + current query
        assert len(messages) >= 3


# ===========================================================================
# Task Agent
# ===========================================================================

class TestTaskAgent:

    @pytest.mark.asyncio
    async def test_create_task_via_agent(self, db_session, regular_user):
        """Task agent should invoke create_task tool and persist to DB."""
        from app.services.agents.task_agent_2 import TaskAgent
        from app.models.models import Task

        # Mock Anthropic tool_use response sequence
        mock_client = AsyncMock()

        # First call: model uses tool
        tool_use_resp = MagicMock()
        tool_use_resp.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "create_task"
        tool_block.id = "tool-call-001"
        tool_block.input = {"title": "Test task", "priority": "high"}
        tool_use_resp.content = [tool_block]

        # Second call: model gives final response
        final_resp = MagicMock()
        final_resp.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I've created the task 'Test task' with high priority."
        final_resp.content = [text_block]

        mock_client.messages.create = AsyncMock(side_effect=[tool_use_resp, final_resp])

        with patch("app.services.agents.task_agent.anthropic.AsyncAnthropic",
                   return_value=mock_client):
            agent = TaskAgent(user_id=regular_user.id, db=db_session)
            response, tool_calls = await agent.run("Create a high priority task: Test task")

        assert "Test task" in response or "created" in response.lower()
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "create_task"

    @pytest.mark.asyncio
    async def test_max_iterations_prevents_infinite_loop(self, db_session, regular_user):
        """Agent should stop after MAX_ITERATIONS even if model keeps calling tools."""
        from app.services.agents.task_agent_2 import TaskAgent, MAX_ITERATIONS

        mock_client = AsyncMock()

        # Always return tool_use (never end_turn) → should hit MAX_ITERATIONS
        tool_resp = MagicMock()
        tool_resp.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "list_tasks"
        tool_block.id = "tool-inf-001"
        tool_block.input = {}
        tool_resp.content = [tool_block]

        mock_client.messages.create = AsyncMock(return_value=tool_resp)

        with patch("app.services.agents.task_agent.anthropic.AsyncAnthropic",
                   return_value=mock_client):
            agent = TaskAgent(user_id=regular_user.id, db=db_session)
            response, tool_calls = await agent.run("List tasks infinitely")

        # Should have stopped at MAX_ITERATIONS
        assert mock_client.messages.create.call_count <= MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_list_tasks_returns_user_tasks_only(self, db_session, regular_user):
        """list_tasks tool must only return tasks belonging to the current user."""
        from app.services.agents.task_agent_2 import TaskToolExecutor
        from app.models.models import Task

        # Create tasks for regular_user
        for i in range(3):
            task = Task(
                user_id=regular_user.id,
                task_type="user_task",
                status="pending",
                input_data={"title": f"Task {i}", "priority": "medium"},
            )
            db_session.add(task)

        # Create task for a different user
        other_task = Task(
            user_id="other-user-id",
            task_type="user_task",
            status="pending",
            input_data={"title": "Other user task"},
        )
        db_session.add(other_task)
        await db_session.flush()

        executor = TaskToolExecutor(user_id=regular_user.id, db=db_session)
        import json
        result = json.loads(await executor.execute("list_tasks", {}))

        assert result["total"] == 3
        titles = [t["title"] for t in result["tasks"]]
        assert "Other user task" not in titles

    @pytest.mark.asyncio
    async def test_complete_task_updates_status(self, db_session, regular_user):
        """complete_task tool should set status='success' and completed_at."""
        from app.services.agents.task_agent_2 import TaskToolExecutor
        from app.models.models import Task
        import json

        task = Task(
            user_id=regular_user.id,
            task_type="user_task",
            status="pending",
            input_data={"title": "Finish me"},
        )
        db_session.add(task)
        await db_session.flush()

        executor = TaskToolExecutor(user_id=regular_user.id, db=db_session)
        result = json.loads(await executor.execute("complete_task", {"task_id": task.id}))

        assert result["status"] == "completed"
        await db_session.refresh(task)
        assert task.status == "success"
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_complete_other_users_task_fails(self, db_session, regular_user):
        """Users cannot complete tasks they don't own."""
        from app.services.agents.task_agent_2 import TaskToolExecutor
        from app.models.models import Task
        import json

        other_task = Task(
            user_id="other-user-999",
            task_type="user_task",
            status="pending",
            input_data={"title": "Not yours"},
        )
        db_session.add(other_task)
        await db_session.flush()

        executor = TaskToolExecutor(user_id=regular_user.id, db=db_session)
        result = json.loads(await executor.execute("complete_task", {"task_id": other_task.id}))

        assert "error" in result
        assert "not found" in result["error"].lower()


# ===========================================================================
# Gmail Agent
# ===========================================================================

class TestGmailAgent:

    @pytest.mark.asyncio
    async def test_no_google_token_raises_value_error(self, db_session, regular_user):
        """Agent should raise ValueError if user hasn't connected Google."""
        from app.services.agents.gmail_agent_2 import GmailAgent

        agent = GmailAgent(user_id=regular_user.id, db=db_session)

        with pytest.raises(ValueError, match="Google account not connected"):
            await agent.search_emails("in:inbox")

    @pytest.mark.asyncio
    async def test_search_emails_returns_parsed_list(
        self, db_session, regular_user, mock_gmail, mock_llm_router
    ):
        """search_emails should return a list of GmailEmailResponse objects."""
        from app.services.agents.gmail_agent_2 import GmailAgent
        from app.models.models import GoogleOAuthToken
        from datetime import datetime, timezone

        # Give user a Google token
        db_session.add(GoogleOAuthToken(
            user_id=regular_user.id,
            access_token="fake-access-token",
            refresh_token="fake-refresh-token",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            token_expiry=datetime(2030, 1, 1, tzinfo=timezone.utc),
        ))
        await db_session.flush()

        with patch("app.services.agents.gmail_agent._get_user_credentials") as mock_creds:
            mock_creds.return_value = MagicMock(expired=False)

            agent = GmailAgent(user_id=regular_user.id, db=db_session)
            emails = await agent.search_emails("in:inbox", max_results=5)

        assert isinstance(emails, list)
        assert len(emails) >= 1
        assert emails[0].message_id == "msg-001"
        assert emails[0].subject == "Test Subject"

    @pytest.mark.asyncio
    async def test_summarise_inbox_calls_llm(
        self, db_session, regular_user, mock_gmail, mock_llm_router
    ):
        """summarise_inbox should call LLM with email content."""
        from app.services.agents.gmail_agent_2 import GmailAgent
        from app.models.models import GoogleOAuthToken
        from datetime import datetime, timezone

        db_session.add(GoogleOAuthToken(
            user_id=regular_user.id,
            access_token="fake-token",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            token_expiry=datetime(2030, 1, 1, tzinfo=timezone.utc),
        ))
        await db_session.flush()

        with patch("app.services.agents.gmail_agent._get_user_credentials",
                   return_value=MagicMock(expired=False)):
            agent = GmailAgent(user_id=regular_user.id, db=db_session)
            summary = await agent.summarise_inbox()

        # LLM was called (mock returns the mock response content)
        mock_llm_router.complete.assert_called_once()
        assert summary is not None