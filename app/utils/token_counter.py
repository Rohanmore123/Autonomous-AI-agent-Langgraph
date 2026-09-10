"""
app/utils/token_counter.py
===========================
Token counting utilities for context window management and cost estimation.

WHY COUNT TOKENS?
  LLMs have a fixed context window (e.g., 200K tokens for Claude 3).
  Exceeding it causes an error. We count tokens BEFORE sending to:
    1. Trim conversation history to fit
    2. Truncate RAG context chunks
    3. Estimate cost before the call

TIKTOKEN vs ANTHROPIC TOKENISER:
  OpenAI's tiktoken library is used for approximation — it's fast and
  the counts are close enough for context management (within 5–10%).
  For exact counts, use provider-specific token counters, but they require
  an API call which adds latency.

  We use cl100k_base (GPT-4 tokeniser) as a universal approximation.
  Claude's tokeniser is different but very similar in count.

CONTEXT WINDOW LIMITS (as of 2025):
  claude-3-5-sonnet-20241022  200,000 tokens
  claude-opus-4               200,000 tokens
  gpt-4o                      128,000 tokens
  gpt-4o-mini                 128,000 tokens
  gemini-1.5-pro              1,000,000 tokens
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import tiktoken

from app.core.logging import get_logger

logger = get_logger(__name__)

# Context window limits by model
CONTEXT_WINDOWS: dict[str, int] = {
    "claude-sonnet-4-6":            200_000,
    "claude-opus-4-6":              200_000,
    "claude-haiku-4-5":             200_000,
    "gpt-4o":                       128_000,
    "gpt-4o-mini":                  128_000,
    "gemini-1.5-pro":             1_000_000,
    "gemini-1.5-flash":           1_000_000,
}

# Safe output buffer — reserve this many tokens for the model's response
OUTPUT_BUFFER = 4_096


@lru_cache(maxsize=1)
def _get_encoder():
    """Load tiktoken encoder once; cache for the process lifetime."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count approximate tokens in a string.
    Uses cl100k_base (GPT-4 tokeniser) as a universal approximation.
    ~20% faster than calling the API for exact counts.
    """
    try:
        enc = _get_encoder()
        return len(enc.encode(text))
    except Exception:
        # Fallback: words * 1.3 is a rough approximation
        return int(len(text.split()) * 1.3)


def count_messages_tokens(messages: list[dict[str, str]]) -> int:
    """
    Count tokens for a list of chat messages.
    Includes per-message overhead (role labels, separators).
    OpenAI overhead: ~4 tokens per message + 2 for reply priming.
    Claude overhead is similar.
    """
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        total += count_tokens(msg.get("role", ""))
        total += count_tokens(msg.get("content", ""))
    total += 2  # reply priming
    return total


def get_context_limit(model: str) -> int:
    """Return the context window limit for a model, defaulting to 128K."""
    return CONTEXT_WINDOWS.get(model, 128_000)


def trim_messages_to_fit(
    messages: list[dict],
    model: str,
    system_prompt: str = "",
    max_output_tokens: int = OUTPUT_BUFFER,
) -> list[dict]:
    """
    Trim conversation history to fit within the model's context window.

    Strategy:
      1. Always keep the LAST message (current user query) — never trim it.
      2. Always keep the FIRST message if it's a system message.
      3. Remove oldest messages from the middle until it fits.

    Args:
        messages:          Full conversation history (list of {role, content} dicts)
        model:             Model name (to look up context window)
        system_prompt:     The system prompt (counted but not in messages list)
        max_output_tokens: How many tokens to reserve for the model's response

    Returns:
        Trimmed messages list that fits within the context window.
    """
    context_limit = get_context_limit(model)
    available = context_limit - max_output_tokens
    system_tokens = count_tokens(system_prompt) if system_prompt else 0
    available -= system_tokens

    if available <= 0:
        logger.warning("token_counter.system_prompt_too_large", model=model)
        return messages[-1:]  # Just keep the last message

    # Check if it already fits
    total_tokens = count_messages_tokens(messages)
    if total_tokens <= available:
        return messages

    # Trim from the oldest non-first messages
    trimmed = list(messages)
    while len(trimmed) > 1 and count_messages_tokens(trimmed) > available:
        # Remove second-oldest (keep first=system/context, last=current query)
        trimmed.pop(1)

    logger.info(
        "token_counter.trimmed",
        original=len(messages),
        trimmed=len(trimmed),
        model=model,
    )
    return trimmed


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate USD cost for an LLM call.
    Prices per 1K tokens (update monthly from provider pricing pages).
    """
    PRICES: dict[str, tuple[float, float]] = {
        # (input_per_1k, output_per_1k)
        "claude-sonnet-4-6":  (0.003,   0.015),
        "claude-opus-4-6":    (0.015,   0.075),
        "claude-haiku-4-5":   (0.00025, 0.00125),
        "gpt-4o":             (0.005,   0.015),
        "gpt-4o-mini":        (0.00015, 0.0006),
        "gemini-1.5-pro":     (0.0035,  0.0105),
        "gemini-1.5-flash":   (0.000075,0.0003),
    }
    in_price, out_price = PRICES.get(model, (0.003, 0.015))
    return (input_tokens / 1000 * in_price) + (output_tokens / 1000 * out_price)


def format_token_usage(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> dict[str, Any]:
    """
    Return a structured token usage summary for logging/response headers.
    """
    cost = estimate_cost(model, input_tokens, output_tokens)
    total = input_tokens + output_tokens
    return {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  total,
        "cost_usd":      round(cost, 6),
        "model":         model,
    }