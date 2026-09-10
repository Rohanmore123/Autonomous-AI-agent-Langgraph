"""Initial schema — all platform tables

Revision ID: 0001_initial_schema
Revises:
Create Date: 2025-01-01 00:00:00.000000

Creates all tables for the LLM Agentic Platform:
  roles, users, user_roles, conversations, messages,
  documents, document_chunks, tasks, agent_runs,
  api_keys, audit_logs, google_oauth_tokens
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from alembic import op

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── roles ──────────────────────────────────────────────────────────────
    op.create_table(
        "roles",
        sa.Column("id",          sa.String(36),  primary_key=True),
        sa.Column("name",        sa.String(50),  nullable=False, unique=True),
        sa.Column("description", sa.Text,        nullable=True),
        sa.Column("permissions", JSONB,           nullable=False, server_default="{}"),
        sa.Column("created_at",  sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )

    # ── users ──────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",              sa.String(36),  primary_key=True),
        sa.Column("email",           sa.String(255), nullable=False, unique=True),
        sa.Column("username",        sa.String(100), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name",       sa.String(255), nullable=True),
        sa.Column("is_active",       sa.Boolean,     nullable=False, server_default="true"),
        sa.Column("is_verified",     sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("metadata",        JSONB,           nullable=False, server_default="{}"),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at",      sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("deleted_at",      sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at",   sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email",      "users", ["email"])
    op.create_index("ix_users_created_at", "users", ["created_at"])

    # Trigram indexes for fast partial search
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE INDEX ix_users_email_trgm ON users USING GIN (email gin_trgm_ops)")
    op.execute("CREATE INDEX ix_users_username_trgm ON users USING GIN (username gin_trgm_ops)")

    # ── user_roles ─────────────────────────────────────────────────────────
    op.create_table(
        "user_roles",
        sa.Column("id",         sa.String(36), primary_key=True),
        sa.Column("user_id",    sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("role_id",    sa.String(36), sa.ForeignKey("roles.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("granted_by", sa.String(36), nullable=True),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "role_id", name="uq_user_roles"),
    )

    # ── conversations ──────────────────────────────────────────────────────
    op.create_table(
        "conversations",
        sa.Column("id",             sa.String(36), primary_key=True),
        sa.Column("user_id",        sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("title",          sa.String(255), nullable=True),
        sa.Column("agent_type",     sa.String(50),  nullable=False, server_default="default"),
        sa.Column("model_used",     sa.String(100), nullable=True),
        sa.Column("total_tokens",   sa.Integer,     nullable=False, server_default="0"),
        sa.Column("total_cost_usd", sa.Float,       nullable=False, server_default="0.0"),
        sa.Column("metadata",       JSONB,           nullable=False, server_default="{}"),
        sa.Column("is_archived",    sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("created_at",     sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at",     sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_conversations_user_id",   "conversations", ["user_id"])
    op.create_index("ix_conversations_created_at","conversations", ["created_at"])

    # ── messages ───────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id",              sa.String(36), primary_key=True),
        sa.Column("conversation_id", sa.String(36),
                  sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role",        sa.String(20),  nullable=False),
        sa.Column("content",     sa.Text,        nullable=False),
        sa.Column("tokens_used", sa.Integer,     nullable=False, server_default="0"),
        sa.Column("latency_ms",  sa.Float,       nullable=True),
        sa.Column("model",       sa.String(100), nullable=True),
        sa.Column("provider",    sa.String(50),  nullable=True),
        sa.Column("sources",     JSONB,           nullable=False, server_default="[]"),
        sa.Column("tool_calls",  JSONB,           nullable=False, server_default="[]"),
        sa.Column("metadata",    JSONB,           nullable=False, server_default="{}"),
        sa.Column("created_at",  sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])

    # ── documents ──────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id",            sa.String(36),  primary_key=True),
        sa.Column("owner_id",      sa.String(36),  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("filename",      sa.String(255), nullable=False),
        sa.Column("content_type",  sa.String(100), nullable=False),
        sa.Column("size_bytes",    sa.Integer,     nullable=False),
        sa.Column("weaviate_class",sa.String(100), nullable=False),
        sa.Column("chunk_count",   sa.Integer,     nullable=False, server_default="0"),
        sa.Column("status",        sa.String(20),  nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text,        nullable=True),
        sa.Column("metadata",      JSONB,           nullable=False, server_default="{}"),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at",    sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])

    # ── document_chunks ────────────────────────────────────────────────────
    op.create_table(
        "document_chunks",
        sa.Column("id",           sa.String(36), primary_key=True),
        sa.Column("document_id",  sa.String(36),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("weaviate_id",  sa.String(36), nullable=False, unique=True),
        sa.Column("chunk_index",  sa.Integer,    nullable=False),
        sa.Column("content",      sa.Text,       nullable=False),
        sa.Column("token_count",  sa.Integer,    nullable=False),
        sa.Column("metadata",     JSONB,          nullable=False, server_default="{}"),
    )
    op.create_index("ix_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_chunks_weaviate_id", "document_chunks", ["weaviate_id"])

    # ── tasks ──────────────────────────────────────────────────────────────
    op.create_table(
        "tasks",
        sa.Column("id",             sa.String(36),  primary_key=True),
        sa.Column("celery_task_id", sa.String(255), nullable=True, unique=True),
        sa.Column("user_id",        sa.String(36),  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("task_type",      sa.String(100), nullable=False),
        sa.Column("status",         sa.String(20),  nullable=False, server_default="pending"),
        sa.Column("input_data",     JSONB,           nullable=False, server_default="{}"),
        sa.Column("output_data",    JSONB,           nullable=True),
        sa.Column("error_message",  sa.Text,         nullable=True),
        sa.Column("retries",        sa.Integer,      nullable=False, server_default="0"),
        sa.Column("started_at",     sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at",     sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_tasks_user_id", "tasks", ["user_id"])
    op.create_index("ix_tasks_status",  "tasks", ["status"])

    # ── agent_runs ─────────────────────────────────────────────────────────
    op.create_table(
        "agent_runs",
        sa.Column("id",              sa.String(36),  primary_key=True),
        sa.Column("user_id",         sa.String(36),  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("conversation_id", sa.String(36),  nullable=True),
        sa.Column("agent_type",      sa.String(50),  nullable=False),
        sa.Column("input_tokens",    sa.Integer,     nullable=False, server_default="0"),
        sa.Column("output_tokens",   sa.Integer,     nullable=False, server_default="0"),
        sa.Column("latency_ms",      sa.Float,       nullable=False),
        sa.Column("provider_used",   sa.String(50),  nullable=False),
        sa.Column("model_used",      sa.String(100), nullable=False),
        sa.Column("was_cached",      sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("used_fallback",   sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("retry_count",     sa.Integer,     nullable=False, server_default="0"),
        sa.Column("success",         sa.Boolean,     nullable=False),
        sa.Column("error_type",      sa.String(100), nullable=True),
        sa.Column("tool_calls_made", JSONB,           nullable=False, server_default="[]"),
        sa.Column("cost_usd",        sa.Float,       nullable=False, server_default="0.0"),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_agent_runs_user_id",    "agent_runs", ["user_id"])
    op.create_index("ix_agent_runs_agent_type", "agent_runs", ["agent_type"])
    op.create_index("ix_agent_runs_created_at", "agent_runs", ["created_at"])

    # ── api_keys ───────────────────────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("id",           sa.String(36),  primary_key=True),
        sa.Column("user_id",      sa.String(36),  sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("name",         sa.String(100), nullable=False),
        sa.Column("key_hash",     sa.String(64),  nullable=False, unique=True),
        sa.Column("key_prefix",   sa.String(10),  nullable=False),
        sa.Column("scopes",       JSONB,           nullable=False, server_default="[]"),
        sa.Column("is_active",    sa.Boolean,     nullable=False, server_default="true"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at",   sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"])

    # ── audit_logs ─────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id",            sa.String(36),  primary_key=True),
        sa.Column("user_id",       sa.String(36),  nullable=True),
        sa.Column("action",        sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50),  nullable=True),
        sa.Column("resource_id",   sa.String(36),  nullable=True),
        sa.Column("ip_address",    sa.String(45),  nullable=True),
        sa.Column("user_agent",    sa.String(500), nullable=True),
        sa.Column("request_id",    sa.String(36),  nullable=True),
        sa.Column("details",       JSONB,           nullable=False, server_default="{}"),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_audit_user_id",   "audit_logs", ["user_id"])
    op.create_index("ix_audit_action",    "audit_logs", ["action"])
    op.create_index("ix_audit_created_at","audit_logs", ["created_at"])

    # ── google_oauth_tokens ────────────────────────────────────────────────
    op.create_table(
        "google_oauth_tokens",
        sa.Column("id",            sa.String(36), primary_key=True),
        sa.Column("user_id",       sa.String(36),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("access_token",  sa.Text,       nullable=False),
        sa.Column("refresh_token", sa.Text,       nullable=True),
        sa.Column("token_expiry",  sa.DateTime(timezone=True), nullable=True),
        sa.Column("scopes",        JSONB,          nullable=False, server_default="[]"),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at",    sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )


def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    op.drop_table("google_oauth_tokens")
    op.drop_table("audit_logs")
    op.drop_table("api_keys")
    op.drop_table("agent_runs")
    op.drop_table("tasks")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("user_roles")
    op.drop_table("users")
    op.drop_table("roles")
