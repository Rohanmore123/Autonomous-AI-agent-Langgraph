"""
app/models/models.py
====================
SQLAlchemy ORM models — the ground truth schema for PostgreSQL.

TABLE OVERVIEW:
  users               — platform accounts with bcrypt-hashed passwords
  roles               — RBAC role definitions (admin, operator, user, viewer)
  user_roles          — many-to-many join (one user can have multiple roles)
  conversations       — chat sessions; one user → many conversations
  messages            — individual turns within a conversation
  documents           — files ingested into the vector DB
  document_chunks     — sub-sections of documents (what's actually embedded)
  tasks               — background task records (Celery job tracking)
  agent_runs          — logs every agent invocation with inputs/outputs/latency
  api_keys            — per-user API keys for programmatic access
  audit_logs          — immutable security event log
  google_oauth_tokens — stored OAuth2 credentials per user (for Gmail agent)
  rate_limit_overrides— per-user rate limit customisation

DESIGN PRINCIPLES:
  • All PKs are UUID strings (not serial integers) — avoids enumeration attacks
    and works across distributed systems without coordination.
  • created_at / updated_at on every table for audit trails.
  • Soft delete (deleted_at IS NULL filter) on sensitive tables.
  • Index on every FK + frequently-filtered column.
  • JSONB columns for flexible metadata that doesn't need to be queried by key.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB as PostgreSQLJSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


JSONB = JSON().with_variant(PostgreSQLJSONB, "postgresql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------

class Role(Base):
    """
    Static role definitions.  Seeded on first boot.
    Roles: admin | operator | user | viewer
    """
    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    permissions: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user_roles: Mapped[list["UserRole"]] = relationship(back_populates="role")


class User(Base):
    """
    Platform user account.
    Password is NEVER stored in plaintext — only the bcrypt hash.
    """
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    user_roles: Mapped[list["UserRole"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")
    documents: Mapped[list["Document"]] = relationship(back_populates="owner")
    tasks: Mapped[list["Task"]] = relationship(back_populates="user")
    api_keys: Mapped[list["APIKey"]] = relationship(back_populates="user")
    google_token: Mapped["GoogleOAuthToken | None"] = relationship(
        back_populates="user", uselist=False
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


class UserRole(Base):
    """Many-to-many join between User and Role."""
    __tablename__ = "user_roles"
    __table_args__ = (
        UniqueConstraint("user_id", "role_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"))
    role_id: Mapped[str] = mapped_column(String(36), ForeignKey("roles.id", ondelete="CASCADE"))
    granted_by: Mapped[str | None] = mapped_column(String(36))  # admin user ID
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship(back_populates="user_roles")
    role: Mapped["Role"] = relationship(back_populates="user_roles")


# ---------------------------------------------------------------------------
# Conversations & Messages
# ---------------------------------------------------------------------------

class Conversation(Base):
    """
    A chat session.  Groups messages + tracks which agent/model was used.
    """
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_user_id", "user_id"),
        Index("ix_conversations_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(String(255))
    agent_type: Mapped[str] = mapped_column(String(50), default="default")
    model_used: Mapped[str | None] = mapped_column(String(100))
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        order_by="Message.created_at",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """Individual turn in a conversation (user or assistant)."""
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(20))          # "user" | "assistant" | "system"
    content: Mapped[str] = mapped_column(Text)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float | None] = mapped_column(Float)  # LLM response time
    model: Mapped[str | None] = mapped_column(String(100))
    provider: Mapped[str | None] = mapped_column(String(50))
    sources: Mapped[list] = mapped_column(JSONB, default=list)  # RAG source chunks
    tool_calls: Mapped[list] = mapped_column(JSONB, default=list)  # Agent tool invocations
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


# ---------------------------------------------------------------------------
# Documents & Chunks (RAG)
# ---------------------------------------------------------------------------

class Document(Base):
    """A file ingested into the vector database."""
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_owner_id", "owner_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    weaviate_class: Mapped[str] = mapped_column(String(100))  # Weaviate collection name
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending | processing | ready | failed
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    owner: Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document")


class DocumentChunk(Base):
    """Granular text chunk. The actual unit stored in Weaviate."""
    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_weaviate_id", "weaviate_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE")
    )
    weaviate_id: Mapped[str] = mapped_column(String(36), unique=True)  # UUID in Weaviate
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    document: Mapped["Document"] = relationship(back_populates="chunks")


# ---------------------------------------------------------------------------
# Background Tasks
# ---------------------------------------------------------------------------

class Task(Base):
    """Tracks every Celery background task submitted by a user."""
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_user_id", "user_id"),
        Index("ix_tasks_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    task_type: Mapped[str] = mapped_column(String(100))  # "ingest_document" | "send_email" etc.
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending | running | success | failed | cancelled
    input_data: Mapped[dict] = mapped_column(JSONB, default=dict)
    output_data: Mapped[dict | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(Text)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship(back_populates="tasks")

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# ---------------------------------------------------------------------------
# Agent Runs
# ---------------------------------------------------------------------------

class AgentRun(Base):
    """
    Immutable log of every agent invocation.
    Used for latency analysis, cost tracking, and debugging.
    """
    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("ix_agent_runs_user_id", "user_id"),
        Index("ix_agent_runs_agent_type", "agent_type"),
        Index("ix_agent_runs_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    conversation_id: Mapped[str | None] = mapped_column(String(36))
    agent_type: Mapped[str] = mapped_column(String(50))    # "gmail" | "task" | "rag" | "router"
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float)
    provider_used: Mapped[str] = mapped_column(String(50))  # "anthropic" | "openai" | "google"
    model_used: Mapped[str] = mapped_column(String(100))
    was_cached: Mapped[bool] = mapped_column(Boolean, default=False)
    used_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean)
    error_type: Mapped[str | None] = mapped_column(String(100))
    tool_calls_made: Mapped[list] = mapped_column(JSONB, default=list)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

class APIKey(Base):
    """Per-user API keys for programmatic access (not OAuth)."""
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_key_hash", "key_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100))             # Human label, e.g. "CI Bot"
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)  # SHA-256 of the raw key
    key_prefix: Mapped[str] = mapped_column(String(10))        # First 8 chars shown to user
    scopes: Mapped[list] = mapped_column(JSONB, default=list)  # ["chat", "documents"]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship(back_populates="api_keys")


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

class AuditLog(Base):
    """
    Append-only security audit trail.
    Never update or delete rows — immutability is the point.
    """
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_user_id", "user_id"),
        Index("ix_audit_action", "action"),
        Index("ix_audit_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(String(36))   # None for unauthenticated events
    action: Mapped[str] = mapped_column(String(100))           # "user.login" | "doc.delete" etc.
    resource_type: Mapped[str | None] = mapped_column(String(50))
    resource_id: Mapped[str | None] = mapped_column(String(36))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    request_id: Mapped[str | None] = mapped_column(String(36))
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ---------------------------------------------------------------------------
# Google OAuth Tokens
# ---------------------------------------------------------------------------

class GoogleOAuthToken(Base):
    """Encrypted storage of Google OAuth2 tokens per user."""
    __tablename__ = "google_oauth_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    access_token: Mapped[str] = mapped_column(Text)    # Encrypted at rest
    refresh_token: Mapped[str | None] = mapped_column(Text)  # Encrypted
    token_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scopes: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    user: Mapped["User"] = relationship(back_populates="google_token")