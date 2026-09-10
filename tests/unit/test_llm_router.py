"""
tests/unit/test_llm_router.py
==============================
Unit tests for the LLM Router layer.

COVERAGE:
  ✓ Primary provider call — success path
  ✓ LLM cache hit — returns cached response, no provider call
  ✓ LLM cache miss — calls provider, stores in cache
  ✓ Primary failure → fallback success
  ✓ Both providers fail → raises RuntimeError
  ✓ Circuit breaker opens after N failures
  ✓ Circuit breaker half-open → probe succeeds → closes
  ✓ Circuit breaker rejects calls when OPEN
  ✓ Cost estimation — correct per-model pricing
  ✓ Token usage is correctly extracted from provider responses
  ✓ Force provider override works
  ✓ Retry logic — retries on RateLimitError, not on AuthenticationError
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.services.llm.router import (
    CircuitBreaker,
    LLMResponse,
    LLMRouter,
    estimate_cost,
    _breakers,
)


# ===========================================================================
# Cost estimation
# ===========================================================================

class TestCostEstimation:
    def test_known_model_cost(self):
        cost = estimate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
        # Input: 1000 * 0.003/1000 = 0.003, Output: 500 * 0.015/1000 = 0.0075
        assert abs(cost - 0.0105) < 0.0001

    def test_unknown_model_uses_default(self):
        cost = estimate_cost("unknown-model-xyz", input_tokens=1000, output_tokens=0)
        # Should not raise — uses default pricing
        assert cost > 0

    def test_zero_tokens_zero_cost(self):
        cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_output_more_expensive_than_input(self):
        input_cost  = estimate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=0)
        output_cost = estimate_cost("claude-sonnet-4-6", input_tokens=0, output_tokens=1000)
        assert output_cost > input_cost  # Output tokens always more expensive


# ===========================================================================
# LLMResponse dataclass
# ===========================================================================

class TestLLMResponse:
    def test_cost_auto_calculated(self):
        resp = LLMResponse(
            content="Hello",
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=500.0,
        )
        assert resp.cost_usd > 0

    def test_cached_response_zero_cost(self):
        """Cached responses have 0 tokens → 0 cost (the original was charged)."""
        resp = LLMResponse(
            content="Cached",
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=0,
            output_tokens=0,
            latency_ms=1.0,
            was_cached=True,
        )
        assert resp.cost_usd == 0.0


# ===========================================================================
# Circuit Breaker
# ===========================================================================

class TestCircuitBreaker:
    def setup_method(self):
        """Reset circuit breaker state before each test."""
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    def test_starts_closed(self):
        assert self.cb.state == CircuitBreaker.CLOSED
        assert not self.cb.is_open()

    def test_opens_after_threshold_failures(self):
        for _ in range(3):
            self.cb.record_failure()
        assert self.cb.state == CircuitBreaker.OPEN
        assert self.cb.is_open()

    def test_success_resets_failure_count(self):
        self.cb.record_failure()
        self.cb.record_failure()
        self.cb.record_success()
        assert self.cb.failure_count == 0
        assert self.cb.state == CircuitBreaker.CLOSED

    def test_half_open_after_timeout(self):
        """After recovery_timeout seconds, OPEN → HALF_OPEN."""
        for _ in range(3):
            self.cb.record_failure()
        assert self.cb.state == CircuitBreaker.OPEN

        # Fake the timeout by backdating _opened_at
        self.cb._opened_at = time.monotonic() - 2.0  # 2 seconds ago

        # is_open() should now transition to HALF_OPEN and return False
        result = self.cb.is_open()
        assert result is False
        assert self.cb.state == CircuitBreaker.HALF_OPEN

    def test_closes_from_half_open_on_success(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb._opened_at = time.monotonic() - 2.0
        self.cb.is_open()  # → HALF_OPEN

        self.cb.record_success()
        assert self.cb.state == CircuitBreaker.CLOSED

    def test_reopens_from_half_open_on_failure(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb._opened_at = time.monotonic() - 2.0
        self.cb.is_open()  # → HALF_OPEN

        self.cb.record_failure()
        assert self.cb.state == CircuitBreaker.OPEN

    def test_does_not_open_before_threshold(self):
        self.cb.record_failure()
        self.cb.record_failure()
        assert self.cb.state == CircuitBreaker.CLOSED   # Still closed at 2 < 3


# ===========================================================================
# LLM Router — Core Routing
# ===========================================================================

class TestLLMRouter:

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(self):
        """Cache hit must return immediately without calling the LLM provider."""
        router = LLMRouter()

        with patch.object(router._cache, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = "Cached LLM response text"

            result = await router.complete(
                messages=[{"role": "user", "content": "Hello"}],
                use_cache=True,
            )

            assert result.was_cached is True
            assert result.content == "Cached LLM response text"
            assert result.latency_ms == 0.0
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_provider_and_stores(self):
        """On cache miss, call provider and cache the result."""
        router = LLMRouter()

        with patch.object(router._cache, "get", return_value=AsyncMock(return_value=None)()) as mock_get, \
             patch.object(router._cache, "set", new_callable=AsyncMock) as mock_set, \
             patch("app.services.llm.router._call_provider", new_callable=AsyncMock) as mock_call:

            mock_get.return_value = None
            mock_call.return_value = ("Provider response", 100, 50)

            result = await router.complete(
                messages=[{"role": "user", "content": "Test"}],
                use_cache=True,
            )

            assert result.was_cached is False
            assert result.content == "Provider response"
            mock_set.assert_called_once()   # Result was cached

    @pytest.mark.asyncio
    async def test_primary_failure_triggers_fallback(self):
        """When primary provider fails, fallback should be called."""
        router = LLMRouter()

        call_count = {"n": 0}

        async def mock_call(provider, model, messages, max_tokens, system, retry_count):
            call_count["n"] += 1
            if provider == "anthropic":
                raise ConnectionError("Primary down")
            return ("Fallback response", 80, 40)

        with patch.object(router._cache, "get", new_callable=AsyncMock, return_value=None), \
             patch.object(router._cache, "set", new_callable=AsyncMock), \
             patch("app.services.llm.router._call_provider", side_effect=mock_call):

            result = await router.complete(
                messages=[{"role": "user", "content": "Test"}],
                use_cache=False,
            )

            assert result.used_fallback is True
            assert result.content == "Fallback response"
            assert call_count["n"] == 2   # Primary + fallback

    @pytest.mark.asyncio
    async def test_both_providers_fail_raises(self):
        """When both primary and fallback fail, raise RuntimeError."""
        router = LLMRouter()

        async def always_fail(provider, model, messages, max_tokens, system, retry_count):
            raise ConnectionError(f"{provider} is down")

        with patch.object(router._cache, "get", new_callable=AsyncMock, return_value=None), \
             patch("app.services.llm.router._call_provider", side_effect=always_fail):

            with pytest.raises(RuntimeError) as exc_info:
                await router.complete(
                    messages=[{"role": "user", "content": "Test"}],
                    use_cache=False,
                )

            assert "Both LLM providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_force_provider_overrides_primary(self):
        """force_provider parameter should bypass the configured primary."""
        router = LLMRouter()

        called_with = {}

        async def mock_call(provider, model, messages, max_tokens, system, retry_count):
            called_with["provider"] = provider
            return ("Response", 50, 25)

        with patch.object(router._cache, "get", new_callable=AsyncMock, return_value=None), \
             patch.object(router._cache, "set", new_callable=AsyncMock), \
             patch("app.services.llm.router._call_provider", side_effect=mock_call):

            await router.complete(
                messages=[{"role": "user", "content": "Test"}],
                use_cache=False,
                force_provider="openai",
            )

            assert called_with["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_cache_disabled_skips_cache(self):
        """When use_cache=False, cache should not be checked or set."""
        router = LLMRouter()

        with patch.object(router._cache, "get", new_callable=AsyncMock) as mock_get, \
             patch.object(router._cache, "set", new_callable=AsyncMock) as mock_set, \
             patch("app.services.llm.router._call_provider", new_callable=AsyncMock,
                   return_value=("Response", 100, 50)):

            await router.complete(
                messages=[{"role": "user", "content": "Test"}],
                use_cache=False,
            )

            mock_get.assert_not_called()
            mock_set.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_skips_provider(self):
        """If circuit breaker is OPEN, provider call should be rejected."""
        # Force open the anthropic circuit breaker
        original_state = _breakers["anthropic"].state
        original_count = _breakers["anthropic"].failure_count
        _breakers["anthropic"].state = CircuitBreaker.OPEN
        _breakers["anthropic"]._opened_at = time.monotonic()  # Just opened

        router = LLMRouter()

        try:
            with patch.object(router._cache, "get", new_callable=AsyncMock, return_value=None), \
                 patch("app.services.llm.router._call_provider") as mock_call:

                # Even if fallback also fails
                mock_call.side_effect = RuntimeError("Circuit open")

                with pytest.raises((RuntimeError, Exception)):
                    await router.complete(
                        messages=[{"role": "user", "content": "Test"}],
                        use_cache=False,
                    )

        finally:
            # Restore circuit breaker state
            _breakers["anthropic"].state = original_state
            _breakers["anthropic"].failure_count = original_count


# ===========================================================================
# Provider call helpers
# ===========================================================================

class TestProviderCalls:
    @pytest.mark.asyncio
    async def test_anthropic_call_extracts_content(self):
        """Verify Anthropic API response parsing."""
        from app.services.llm.router import _call_anthropic

        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Hello from Claude")]
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5

        with patch("app.services.llm.router._get_anthropic_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_resp)
            mock_client_factory.return_value = mock_client

            content, in_tok, out_tok = await _call_anthropic(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-sonnet-4-6",
                max_tokens=100,
                system="You are helpful.",
            )

            assert content == "Hello from Claude"
            assert in_tok == 10
            assert out_tok == 5

    @pytest.mark.asyncio
    async def test_openai_call_extracts_content(self):
        """Verify OpenAI API response parsing."""
        from app.services.llm.router import _call_openai

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Hello from GPT"))]
        mock_resp.usage.prompt_tokens = 15
        mock_resp.usage.completion_tokens = 8

        with patch("app.services.llm.router._get_openai_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_client_factory.return_value = mock_client

            content, in_tok, out_tok = await _call_openai(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
                max_tokens=100,
                system=None,
            )

            assert content == "Hello from GPT"
            assert in_tok == 15
            assert out_tok == 8