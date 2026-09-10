"""
app/services/llm/__init__.py
=============================
LLM service package exports.

PACKAGE STRUCTURE:
  router.py      — Primary/Fallback routing, circuit breaker, retry, caching
  providers.py   — Individual provider wrappers (Anthropic, OpenAI, Google)
  embeddings.py  — Embedding model management and batch processing
  prompts.py     — Prompt templates and system prompt library
  streaming.py   — SSE streaming helpers for all providers
"""

from app.services.llm.router import LLMResponse, LLMRouter, llm_router
from app.services.llm.providers import AnthropicProvider, OpenAIProvider, GoogleProvider
from app.services.llm.embeddings import EmbeddingService, embedding_service

__all__ = [
    "LLMResponse",
    "LLMRouter",
    "llm_router",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "EmbeddingService",
    "embedding_service",
]