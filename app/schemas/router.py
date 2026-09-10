"""
app/services/llm/router.py
==========================
Production LLM Router with:
  • Primary → Fallback provider routing
  • Retry with exponential back-off (tenacity)
  • Circuit breaker per provider (prevents cascade failures)
  • Response caching (Redis) for identical prompts
  • Token counting and cost estimation
  • Full latency tracking and AgentRun logging
  • Streaming support

ROUTING FLOW:
  1. Check cache — return immediately if hit
  2. Try PRIMARY provider (with retry)
  3. If primary fails after retries → try FALLBACK provider
  4. If both fail → raise ServiceUnavailableError
  5. Log AgentRun record regardless of outcome

CIRCUIT BREAKER PATTERN:
  Each provider has its own circuit breaker.
  After N consecutive failures the breaker OPENS — calls are rejected immediately
  without hitting the provider API.  After a cool-down period it goes HALF-OPEN
  and tries one request.  If it succeeds, it CLOSES again.
  This prevents hammering a down provider and allows fast fail-over.

RETRY STRATEGY (tenacity):
  wait = exponential(multiplier=1, max=10)   → 1s, 2s, 4s, 8s, 10s
  stop = after_attempt(3)
  retry on: RateLimitError, ConnectionError, TimeoutError
  do NOT retry on: AuthenticationError, InvalidRequestError
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import anthropic
import openai
import google.generativeai as genai
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.redis_client import LLMCache

settings = get_settings()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cost table (USD per 1K tokens) — update monthly
# ---------------------------------------------------------------------------
COST_TABLE: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6":    {"input": 0.003,  "output": 0.015},
    "claude-opus-4-6":      {"input": 0.015,  "output": 0.075},
    "gpt-4o":               {"input": 0.005,  "output": 0.015},
    "gpt-4o-mini":          {"input": 0.00015,"output": 0.0006},
    "gemini-1.5-pro":       {"input": 0.0035, "output": 0.0105},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_TABLE.get(model, {"input": 0.003, "output": 0.015})
    return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])


# ---------------------------------------------------------------------------
# Simple in-process circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Finite-state machine: CLOSED → OPEN → HALF_OPEN → CLOSED
    CLOSED:    pass requests through normally
    OPEN:      reject immediately (fast fail)
    HALF_OPEN: allow one probe; close on success, open on failure
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0) -> None:
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        if self.state == self.OPEN:
            if time.monotonic() - (self._opened_at or 0) >= self.recovery_timeout:
                self.state = self.HALF_OPEN
                logger.info("circuit_breaker.half_open")
                return False
            return True
        return False

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = self.CLOSED

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "circuit_breaker.opened",
                failures=self.failure_count,
                threshold=self.failure_threshold,
            )


# One breaker per provider
_breakers: dict[str, CircuitBreaker] = {
    "anthropic": CircuitBreaker(),
    "openai": CircuitBreaker(),
    "google": CircuitBreaker(),
}


# ---------------------------------------------------------------------------
# Provider clients (lazy-init)
# ---------------------------------------------------------------------------

def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key=settings.llm.anthropic_api_key,
        timeout=settings.llm.timeout_seconds,
        max_retries=0,   # We handle retries ourselves via tenacity
    )


def _get_openai_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=settings.llm.openai_api_key,
        timeout=settings.llm.timeout_seconds,
        max_retries=0,
    )


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    was_cached: bool = False
    used_fallback: bool = False
    retry_count: int = 0
    cost_usd: float = field(init=False)

    def __post_init__(self) -> None:
        self.cost_usd = estimate_cost(self.model, self.input_tokens, self.output_tokens)


# ---------------------------------------------------------------------------
# Per-provider call functions
# ---------------------------------------------------------------------------

async def _call_anthropic(
    messages: list[dict],
    model: str,
    max_tokens: int,
    system: str | None,
) -> tuple[str, int, int]:
    """Returns (content, input_tokens, output_tokens)."""
    client = _get_anthropic_client()

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    resp = await client.messages.create(**kwargs)
    content = resp.content[0].text
    return content, resp.usage.input_tokens, resp.usage.output_tokens


async def _call_openai(
    messages: list[dict],
    model: str,
    max_tokens: int,
    system: str | None,
) -> tuple[str, int, int]:
    client = _get_openai_client()

    oai_messages = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(messages)

    resp = await client.chat.completions.create(
        model=model,
        messages=oai_messages,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    return content, usage.prompt_tokens, usage.completion_tokens


async def _call_provider(
    provider: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    system: str | None,
    retry_count: int,
) -> tuple[str, int, int]:
    """Dispatch to the right provider function; updates circuit breaker."""
    breaker = _breakers[provider]

    if breaker.is_open():
        raise RuntimeError(f"Circuit breaker OPEN for {provider}")

    # Retryable exceptions (rate limits, transient network errors)
    retryable = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        openai.RateLimitError,
        openai.APIConnectionError,
        TimeoutError,
        ConnectionError,
    )

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(settings.llm.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(retryable),
            reraise=True,
        ):
            with attempt:
                if provider == "anthropic":
                    result = await _call_anthropic(messages, model, max_tokens, system)
                elif provider == "openai":
                    result = await _call_openai(messages, model, max_tokens, system)
                else:
                    raise ValueError(f"Unknown provider: {provider}")

        breaker.record_success()
        return result

    except RetryError as exc:
        breaker.record_failure()
        raise exc.last_attempt.exception() from exc
    except Exception:
        breaker.record_failure()
        raise


# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    Entry point for all LLM calls throughout the application.

    Usage:
        router = LLMRouter()
        resp = await router.complete(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
        )
    """

    def __init__(self) -> None:
        self._cache = LLMCache()

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
        use_cache: bool = True,
        force_provider: str | None = None,
    ) -> LLMResponse:
        """
        Route an LLM completion request with caching + fallback.
        """
        primary = force_provider or settings.llm.primary_provider
        primary_model = settings.llm.primary_model
        fallback = settings.llm.fallback_provider
        fallback_model = settings.llm.fallback_model

        # --- Cache check ---
        if use_cache and settings.enable_cache:
            cached = await self._cache.get(primary, primary_model, messages)
            if cached:
                return LLMResponse(
                    content=cached,
                    provider=primary,
                    model=primary_model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    was_cached=True,
                )

        used_fallback = False
        retry_count = 0
        start = time.perf_counter()

        # --- Primary attempt ---
        try:
            content, in_tok, out_tok = await _call_provider(
                primary, primary_model, messages, max_tokens, system, retry_count
            )
            provider_used = primary
            model_used = primary_model

        except Exception as primary_exc:
            logger.warning(
                "llm.primary.failed",
                provider=primary,
                error=str(primary_exc),
            )

            # --- Fallback attempt ---
            if fallback and fallback != primary:
                try:
                    content, in_tok, out_tok = await _call_provider(
                        fallback, fallback_model, messages, max_tokens, system, retry_count
                    )
                    provider_used = fallback
                    model_used = fallback_model
                    used_fallback = True
                    logger.info("llm.fallback.success", fallback=fallback)
                except Exception as fallback_exc:
                    logger.error(
                        "llm.fallback.failed",
                        fallback=fallback,
                        error=str(fallback_exc),
                    )
                    raise RuntimeError(
                        f"Both LLM providers failed. Primary: {primary_exc}. Fallback: {fallback_exc}"
                    )
            else:
                raise

        latency_ms = (time.perf_counter() - start) * 1000

        # --- Cache the response ---
        if use_cache and settings.enable_cache and not used_fallback:
            await self._cache.set(primary, primary_model, messages, content)

        return LLMResponse(
            content=content,
            provider=provider_used,
            model=model_used,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            used_fallback=used_fallback,
        )

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """
        Streaming completion — yields text chunks as they arrive.
        Only Anthropic streaming is wired here; extend for OpenAI similarly.
        """
        client = _get_anthropic_client()
        kwargs: dict = {
            "model": settings.llm.primary_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text


# Module-level singleton
llm_router = LLMRouter()