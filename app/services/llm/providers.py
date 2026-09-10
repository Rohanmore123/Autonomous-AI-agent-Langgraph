"""
app/services/llm/providers.py
==============================
Unified provider abstraction layer.

WHY A PROVIDER ABSTRACTION?
  Each LLM provider has a different API shape:
    • Anthropic:  messages=[...], system="..."  (system is top-level)
    • OpenAI:     messages=[{"role":"system",...}, ...]  (system is first message)
    • Google:     GenerativeModel.generate_content(parts=[...])  (completely different)

  The abstraction normalises all three into:
    complete(messages, system, max_tokens) → (content, input_tokens, output_tokens)

  The LLMRouter calls this interface — it doesn't care which provider is underneath.
  Adding a new provider (Mistral, Cohere, etc.) = write one new class, zero router changes.

DESIGN PATTERN: Strategy Pattern
  LLMRouter holds a reference to the current provider strategy.
  Strategies are swapped at runtime based on circuit breaker state.

PROVIDER CAPABILITIES TABLE:
  ┌───────────────┬─────────────┬───────────┬──────────┬──────────────┐
  │ Provider      │ Tool Use    │ Vision    │ Streaming│ Max Output   │
  ├───────────────┼─────────────┼───────────┼──────────┼──────────────┤
  │ Anthropic     │ ✓ (native)  │ ✓         │ ✓        │ 8192 tokens  │
  │ OpenAI        │ ✓ (native)  │ ✓         │ ✓        │ 16384 tokens │
  │ Google        │ ✓ (native)  │ ✓         │ ✓        │ 8192 tokens  │
  └───────────────┴─────────────┴───────────┴──────────┴──────────────┘
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import AsyncIterator, Any

import anthropic
import openai

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base provider interface
# ---------------------------------------------------------------------------

@dataclass
class ProviderResponse:
    """Normalised response from any LLM provider."""
    content: str
    input_tokens: int
    output_tokens: int
    raw_response: Any = None          # Original provider response object for debugging
    finish_reason: str = "stop"       # stop | length | tool_use | content_filter
    tool_calls: list[dict] = None     # Populated when model uses tools

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


class BaseLLMProvider(abc.ABC):
    """
    Abstract base class for all LLM providers.

    Every provider must implement:
      complete()  — single-turn or multi-turn completion
      stream()    — streaming completion (yields str chunks)
      complete_with_tools()  — tool use / function calling
    """

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str | None,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> ProviderResponse:
        ...

    @abc.abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        ...

    @abc.abstractmethod
    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None,
        max_tokens: int,
    ) -> ProviderResponse:
        ...


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.

    Key differences from OpenAI:
      • System prompt is a top-level parameter (not a message with role="system")
      • Tool results are sent as user messages with content type "tool_result"
      • Streaming uses context manager pattern
      • Usage is returned as resp.usage.input_tokens / output_tokens
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.llm.primary_model
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.llm.anthropic_api_key,
            timeout=settings.llm.timeout_seconds,
            max_retries=0,   # We handle retries in the router
        )

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        kwargs_to_send: dict = {
            "model":      self.model,
            "max_tokens": max_tokens,
            "messages":   messages,
            "temperature": temperature,
        }
        if system:
            kwargs_to_send["system"] = system

        start = time.perf_counter()
        resp = await self._client.messages.create(**kwargs_to_send)
        latency_ms = (time.perf_counter() - start) * 1000

        content = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content += block.text

        logger.debug(
            "anthropic.complete",
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=round(latency_ms, 2),
            finish_reason=resp.stop_reason,
        )

        return ProviderResponse(
            content=content,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw_response=resp,
            finish_reason=resp.stop_reason or "stop",
        )

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        kwargs: dict = {
            "model":       self.model,
            "max_tokens":  max_tokens,
            "messages":    messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream_ctx:
            async for text_chunk in stream_ctx.text_stream:
                yield text_chunk

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
    ) -> ProviderResponse:
        """
        Tool use (function calling) for Anthropic.
        Tools are defined in Anthropic's JSON schema format.
        """
        kwargs: dict = {
            "model":      self.model,
            "max_tokens": max_tokens,
            "messages":   messages,
            "tools":      tools,
        }
        if system:
            kwargs["system"] = system

        resp = await self._client.messages.create(**kwargs)

        tool_calls = []
        text_content = ""

        for block in resp.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id":    block.id,
                    "name":  block.name,
                    "input": block.input,
                })

        return ProviderResponse(
            content=text_content,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw_response=resp,
            finish_reason=resp.stop_reason or "tool_use",
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT provider (also compatible with Azure OpenAI via base_url).

    Key differences from Anthropic:
      • System prompt is the first message with role="system"
      • Tool results sent as messages with role="tool"
      • Usage: resp.usage.prompt_tokens / completion_tokens
      • Streaming uses async generator pattern
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.llm.fallback_model
        self._client = openai.AsyncOpenAI(
            api_key=settings.llm.openai_api_key,
            timeout=settings.llm.timeout_seconds,
            max_retries=0,
        )

    def _build_messages(
        self, messages: list[dict], system: str | None
    ) -> list[dict]:
        """Prepend system message if provided (OpenAI style)."""
        if system:
            return [{"role": "system", "content": system}, *messages]
        return messages

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        oai_messages = self._build_messages(messages, system)

        start = time.perf_counter()
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        content = resp.choices[0].message.content or ""
        usage   = resp.usage

        logger.debug(
            "openai.complete",
            model=self.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=round(latency_ms, 2),
        )

        return ProviderResponse(
            content=content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            raw_response=resp,
            finish_reason=resp.choices[0].finish_reason or "stop",
        )

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        oai_messages = self._build_messages(messages, system)

        async with await self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ) as stream_ctx:
            async for chunk in stream_ctx:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
    ) -> ProviderResponse:
        """
        OpenAI function calling.
        Tools are in OpenAI's function schema format — different from Anthropic's.
        """
        oai_messages = self._build_messages(messages, system)

        # Convert Anthropic-style tool schema to OpenAI function schema
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            tools=oai_tools,
            max_tokens=max_tokens,
        )

        choice = resp.choices[0]
        tool_calls = []

        if choice.message.tool_calls:
            import json
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id":    tc.id,
                    "name":  tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        return ProviderResponse(
            content=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            raw_response=resp,
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# Google Gemini Provider
# ---------------------------------------------------------------------------

class GoogleProvider(BaseLLMProvider):
    """
    Google Gemini provider via google-generativeai SDK.

    Key differences:
      • Messages are called "contents" with "parts"
      • System instruction is a separate parameter
      • Streaming uses response.resolve() pattern
      • Token counting uses model.count_tokens()
    """

    def __init__(self, model: str = "gemini-1.5-pro") -> None:
        import google.generativeai as genai
        genai.configure(api_key=settings.llm.google_ai_api_key)
        self.model_name = model
        self._genai = genai

    def _build_contents(
        self, messages: list[dict], system: str | None
    ) -> tuple[list[dict], dict | None]:
        """Convert OpenAI-style messages to Gemini contents format."""
        system_instruction = None
        if system:
            system_instruction = {"parts": [{"text": system}]}

        # Map roles: "user" → "user", "assistant" → "model"
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        return contents, system_instruction

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        import asyncio
        contents, system_instruction = self._build_contents(messages, system)

        model_kwargs: dict = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            **model_kwargs,
        )

        config = self._genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # google-generativeai is sync — run in executor
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: model.generate_content(contents, generation_config=config),
        )

        content = resp.text or ""
        usage   = resp.usage_metadata

        return ProviderResponse(
            content=content,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            raw_response=resp,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        import asyncio
        contents, system_instruction = self._build_contents(messages, system)
        model_kwargs: dict = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = self._genai.GenerativeModel(self.model_name, **model_kwargs)
        config = self._genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(contents, generation_config=config, stream=True),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 2048,
    ) -> ProviderResponse:
        # Simplified — Google tool use has a different schema format
        # In production, implement full Google FunctionDeclaration conversion
        logger.warning("google.tool_use.not_fully_implemented")
        return await self.complete(messages, system, max_tokens)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_provider(provider_name: str, model: str | None = None) -> BaseLLMProvider:
    """
    Factory function: return the right provider instance by name.

    Usage:
        provider = get_provider("anthropic")
        resp = await provider.complete(messages=[...], system="...")
    """
    providers: dict[str, type[BaseLLMProvider]] = {
        "anthropic": AnthropicProvider,
        "openai":    OpenAIProvider,
        "google":    GoogleProvider,
    }
    cls = providers.get(provider_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Available: {list(providers.keys())}"
        )
    return cls(model=model) if model else cls()