"""
app/core/config.py
==================
Centralised configuration using Pydantic-Settings.

WHY PYDANTIC SETTINGS?
• Type-safe env parsing — wrong types crash at startup, not at 3 AM in prod.
• Nested models keep related settings grouped.
• .env file support means zero code changes between environments.
• Validators allow cross-field logic (e.g., derive DATABASE_URL if not set).

PRODUCTION PRINCIPLE:
  "Fail fast at startup, not slowly at runtime."
  All config is validated once; the rest of the app trusts Config.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection pool tuning."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    url: str = Field(..., alias="DATABASE_URL")
    pool_size: int = 20          # Number of persistent connections per worker
    max_overflow: int = 40       # Extra connections allowed under high load
    pool_timeout: int = 30       # Seconds to wait for a connection before error
    pool_recycle: int = 3600     # Recycle connections every hour (prevents stale)
    echo: bool = False           # SQL query logging — True only in DEBUG

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", populate_by_name=True
    )


class RedisSettings(BaseSettings):
    """Redis URLs for each logical database (0–3 keep concerns separated)."""

    url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    cache_url: str = Field("redis://localhost:6379/1", alias="REDIS_CACHE_URL")
    celery_url: str = Field("redis://localhost:6379/2", alias="REDIS_CELERY_URL")
    rate_limit_url: str = Field("redis://localhost:6379/3", alias="REDIS_RATE_LIMIT_URL")
    max_connections: int = Field(50, alias="REDIS_MAX_CONNECTIONS")

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", populate_by_name=True
    )


class WeaviateSettings(BaseSettings):
    """Weaviate vector database configuration."""

    url: str = Field("http://localhost:8080", alias="WEAVIATE_URL")
    api_key: str | None = Field(None, alias="WEAVIATE_API_KEY")
    class_prefix: str = Field("LLMPlatform", alias="WEAVIATE_CLASS_PREFIX")

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", populate_by_name=True
    )


class LLMSettings(BaseSettings):
    """
    LLM provider routing and fallback configuration.

    ROUTING STRATEGY:
      Primary → if fails → Fallback → if fails → raise HTTPException(503)
    Each provider has its own API key; the router picks the right client.
    """

    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    google_ai_api_key: str = Field("", alias="GOOGLE_AI_API_KEY")

    primary_provider: Literal["openai", "anthropic", "google"] = Field(
        "openai", alias="PRIMARY_LLM_PROVIDER"
    )
    fallback_provider: Literal["openai", "anthropic", "google"] = Field(
        "google", alias="FALLBACK_LLM_PROVIDER"
    )
    primary_model: str = Field("gpt-4o", alias="PRIMARY_LLM_MODEL")
    fallback_model: str = Field("gemini-2.5-flash", alias="FALLBACK_LLM_MODEL")

    max_retries: int = Field(3, alias="LLM_MAX_RETRIES")
    timeout_seconds: int = Field(60, alias="LLM_TIMEOUT_SECONDS")

    # Embedding model (local, no API cost)
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(384, alias="EMBEDDING_DIMENSION")

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", populate_by_name=True
    )


class JWTSettings(BaseSettings):
    """JWT token configuration."""

    secret_key: str = Field(..., alias="SECRET_KEY")
    algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(7, alias="REFRESH_TOKEN_EXPIRE_DAYS")

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", populate_by_name=True
    )


class GoogleOAuthSettings(BaseSettings):
    """Google OAuth2 for Gmail / Calendar agent."""

    client_id: str = Field("", alias="GOOGLE_CLIENT_ID")
    client_secret: str = Field("", alias="GOOGLE_CLIENT_SECRET")
    redirect_uri: str = Field("", alias="GOOGLE_REDIRECT_URI")
    scopes: list[str] = Field(
        default=[
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/calendar",
        ],
        alias="GOOGLE_SCOPES",
    )

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", populate_by_name=True
    )

    @field_validator("scopes", mode="before")
    @classmethod
    def parse_scopes(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v


class RateLimitSettings(BaseSettings):
    per_minute: int = Field(60, alias="RATE_LIMIT_PER_MINUTE")
    per_hour: int = Field(1000, alias="RATE_LIMIT_PER_HOUR")
    burst: int = Field(10, alias="RATE_LIMIT_BURST")

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", populate_by_name=True
    )


class ObservabilitySettings(BaseSettings):
    """OpenTelemetry + Prometheus configuration."""

    otel_endpoint: str = Field("http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = Field("llm-platform", alias="OTEL_SERVICE_NAME")
    otel_service_version: str = Field("1.0.0", alias="OTEL_SERVICE_VERSION")
    otel_environment: str = Field("production", alias="OTEL_ENVIRONMENT")
    prometheus_enabled: bool = Field(True, alias="PROMETHEUS_ENABLED")

    model_config = SettingsConfigDict(env_prefix="", populate_by_name=True)


class Settings(BaseSettings):
    """
    Master settings object — compose all sub-settings here.
    Loaded once at startup via get_settings() (lru_cache).
    """

    # Application identity
    app_name: str = Field("LLM Agentic Platform", alias="APP_NAME")
    app_env: Literal["development", "staging", "production"] = Field(
        "production", alias="APP_ENV"
    )
    app_version: str = Field("1.0.0", alias="APP_VERSION")
    debug: bool = Field(False, alias="DEBUG")

    # Server
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8000, alias="PORT")
    workers: int = Field(4, alias="WORKERS")

    # CORS
    cors_origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "console"] = Field("json", alias="LOG_FORMAT")

    # Feature flags
    enable_hybrid_search: bool = Field(True, alias="ENABLE_HYBRID_SEARCH")
    enable_gmail_agent: bool = Field(True, alias="ENABLE_GMAIL_AGENT")
    enable_task_agent: bool = Field(True, alias="ENABLE_TASK_AGENT")
    enable_circuit_breaker: bool = Field(True, alias="ENABLE_CIRCUIT_BREAKER")
    enable_cache: bool = Field(True, alias="ENABLE_CACHE")

    # Sub-settings (populated from the same .env via model_validator)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    weaviate: WeaviateSettings = Field(default_factory=WeaviateSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    google_oauth: GoogleOAuthSettings = Field(default_factory=GoogleOAuthSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",        # Ignore unknown env vars — prevents crashes from unrelated vars
        populate_by_name=True,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton settings loader.
    lru_cache ensures .env is read exactly once — cheap on every request.
    """
    return Settings()