"""
app/schemas/schemas.py
======================
Pydantic v2 request/response schemas — the API contract layer.

WHY SEPARATE SCHEMAS FROM MODELS?
  ORM models reflect database structure.  Schemas reflect API shape.
  They're often different:
    • Passwords come IN but never go OUT.
    • Computed fields (e.g., duration_seconds) exist in responses but not in DB.
    • Different endpoints need different subsets of the same entity.

NAMING CONVENTION:
  <Entity>Create   — POST request body
  <Entity>Update   — PATCH request body (all fields Optional)
  <Entity>Response — what we return (never includes secrets)
  <Entity>InDB     — full DB representation (internal use only)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic wrappers
# ---------------------------------------------------------------------------

class PaginatedResponse(BaseModel, Generic[T]):
    """Standard paginated list response."""
    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int


class APIResponse(BaseModel, Generic[T]):
    """Consistent envelope for all API responses."""
    success: bool = True
    data: T | None = None
    message: str | None = None
    request_id: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    errors: list[ErrorDetail]
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class UserRegister(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str | None = Field(None, max_length=255)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int        # seconds


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    username: str
    full_name: str | None
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: datetime | None
    roles: list[str] = []


class UserUpdate(BaseModel):
    full_name: str | None = Field(None, max_length=255)
    username: str | None = Field(None, min_length=3, max_length=50)


class UserAdminUpdate(BaseModel):
    is_active: bool | None = None
    is_verified: bool | None = None
    roles: list[str] | None = None


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class RoleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None
    permissions: dict[str, Any]


class AssignRoleRequest(BaseModel):
    user_id: str
    role_name: str


# ---------------------------------------------------------------------------
# Chat / Conversations
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=32000)


class ChatRequest(BaseModel):
    """
    Unified chat request.  agent_type routes to the right agent.
    include_sources=True returns the RAG chunks used to build the answer.
    """
    conversation_id: str | None = None
    message: str = Field(..., min_length=1, max_length=32000)
    agent_type: str = Field(
        "rag",
        description="rag | gmail | task | router"
    )
    include_sources: bool = False
    stream: bool = False
    model_override: str | None = None     # Admin can force a specific model


class SourceChunk(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    content: str
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    content: str
    role: str = "assistant"
    model_used: str
    provider_used: str
    tokens_used: int
    latency_ms: float
    was_cached: bool
    used_fallback: bool
    sources: list[SourceChunk] = []
    tool_calls: list[dict] = []
    created_at: datetime


# ---------------------------------------------------------------------------
# Documents (RAG)
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    status: str
    created_at: datetime


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Hybrid search weight: 0=BM25 only, 1=vector only, 0.5=balanced",
    )
    document_ids: list[str] | None = None   # Filter to specific docs


class DocumentSearchResponse(BaseModel):
    results: list[SourceChunk]
    total_found: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class TaskCreate(BaseModel):
    task_type: str
    input_data: dict[str, Any] = {}


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    celery_task_id: str | None
    task_type: str
    status: str
    input_data: dict
    output_data: dict | None
    error_message: str | None
    retries: int
    duration_seconds: float | None
    created_at: datetime
    completed_at: datetime | None


# ---------------------------------------------------------------------------
# Agent-specific
# ---------------------------------------------------------------------------

class GmailSearchRequest(BaseModel):
    query: str = Field(..., description="Gmail search query, e.g. 'from:boss@corp.com'")
    max_results: int = Field(10, ge=1, le=50)
    include_body: bool = False


class GmailSendRequest(BaseModel):
    to: list[EmailStr]
    subject: str = Field(..., max_length=500)
    body: str = Field(..., max_length=50000)
    cc: list[EmailStr] = []
    bcc: list[EmailStr] = []


class GmailEmailResponse(BaseModel):
    message_id: str
    thread_id: str
    subject: str
    sender: str
    recipients: list[str]
    date: str
    snippet: str
    body: str | None = None


class TaskAgentRequest(BaseModel):
    """Natural language task management via the Task Agent."""
    instruction: str = Field(..., min_length=1, max_length=5000)
    context: dict[str, Any] = {}


class AgentRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_type: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider_used: str
    model_used: str
    was_cached: bool
    used_fallback: bool
    retry_count: int
    success: bool
    cost_usd: float
    created_at: datetime


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = ["chat"]
    expires_in_days: int | None = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    is_active: bool
    created_at: datetime
    expires_at: datetime | None


class APIKeyCreatedResponse(APIKeyResponse):
    """Returned ONCE at creation — raw_key is never stored."""
    raw_key: str


# ---------------------------------------------------------------------------
# Health & Metrics
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str                        # "healthy" | "degraded" | "unhealthy"
    version: str
    environment: str
    components: dict[str, bool]        # db, redis, weaviate, llm
    uptime_seconds: float


class MetricsSummary(BaseModel):
    total_requests_today: int
    total_tokens_today: int
    avg_latency_ms: float
    cache_hit_rate: float
    error_rate: float
    active_users: int
    agent_breakdown: dict[str, int]    # agent_type → call count