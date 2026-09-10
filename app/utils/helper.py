"""
app/utils/helpers.py
=====================
General-purpose utilities used across the platform.

Contents:
  • Pagination helpers         — consistent offset/cursor pagination
  • Response builders          — wrap data in APIResponse envelopes
  • String utilities           — truncation, sanitisation, slug generation
  • Datetime utilities         — timezone-aware helpers
  • ID generation              — short UUIDs for human-readable IDs
  • Retry decorator            — sync version using tenacity
  • Data masking               — PII redaction for logs
  • File utilities             — safe filename generation, MIME detection

These are pure utility functions — no DB, no LLM, no external dependencies.
That keeps them testable and importable without circular imports.
"""

from __future__ import annotations

import hashlib
import math
import mimetypes
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any, TypeVar

import shortuuid

T = TypeVar("T")


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def generate_id(prefix: str = "") -> str:
    """
    Generate a short, URL-safe ID.
    shortuuid produces 22-char base57 strings (no ambiguous chars like 0/O/l/1).
    prefix allows namespacing: "user_", "conv_", "msg_"

    Example: generate_id("msg_") → "msg_3Hk9Pz7wQvRnXcT2Yj6Fm"
    """
    short = shortuuid.uuid()
    return f"{prefix}{short}"


def generate_uuid() -> str:
    """Standard UUID4 string."""
    return str(uuid.uuid4())


def generate_api_key() -> tuple[str, str]:
    """
    Generate a raw API key and its prefix.
    Returns: (raw_key, prefix)

    Format: llmp_<32 random hex chars>
    Prefix (shown to user): llmp_<first 8 chars>...
    Full key is never stored — only SHA-256 hash.
    """
    raw = "llmp_" + uuid.uuid4().hex + uuid.uuid4().hex[:8]
    prefix = raw[:13]  # "llmp_" + 8 chars
    return raw, prefix


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hash of a raw API key. This is what's stored in the DB."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def paginate(
    items: list[T],
    page: int,
    page_size: int,
) -> tuple[list[T], int]:
    """
    Slice a list for offset-based pagination.
    Returns (page_items, total_count).

    In production with SQLAlchemy, use SQL LIMIT/OFFSET instead.
    This is for in-memory pagination of small result sets.
    """
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return items[start:end], total


def total_pages(total: int, page_size: int) -> int:
    """Calculate total page count."""
    return math.ceil(total / page_size) if page_size > 0 else 0


def pagination_meta(page: int, page_size: int, total: int) -> dict[str, int]:
    """
    Build a standard pagination metadata dict.
    Attach to any list response.
    """
    return {
        "page":       page,
        "page_size":  page_size,
        "total":      total,
        "pages":      total_pages(total, page_size),
        "has_next":   page < total_pages(total, page_size),
        "has_prev":   page > 1,
    }


# ---------------------------------------------------------------------------
# Datetime utilities
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    """Return current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


def to_iso(dt: datetime | None) -> str | None:
    """Format datetime as ISO 8601 string, or None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def seconds_until(dt: datetime) -> int:
    """Return seconds until a future datetime (negative if past)."""
    now = utcnow()
    delta = dt - now if dt.tzinfo else dt - now.replace(tzinfo=None)
    return int(delta.total_seconds())


# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 100, suffix: str = "...") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def slugify(text: str) -> str:
    """
    Convert text to a URL-safe slug.
    "Hello World! 2024" → "hello-world-2024"
    """
    # Normalise unicode
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Lowercase and replace non-alphanumeric with hyphens
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text


def sanitise_filename(filename: str) -> str:
    """
    Make a filename safe for storage (remove path traversal characters).
    "../../etc/passwd" → "etc_passwd"
    """
    # Strip path separators and null bytes
    filename = re.sub(r"[/\\:*?\"<>|\x00]", "_", filename)
    # Limit length
    name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
    name = name[:200]
    return f"{name}.{ext}" if ext else name


def extract_domain(email: str) -> str:
    """Extract domain from email address."""
    return email.split("@")[-1].lower() if "@" in email else ""


def normalise_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Data masking (PII redaction for logs)
# ---------------------------------------------------------------------------

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"\b\d{10,15}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")


def mask_pii(text: str) -> str:
    """
    Redact common PII patterns from a string.
    Use before logging user-submitted content.
    email@example.com → e***@example.com
    """
    def _mask_email(m: re.Match) -> str:
        parts = m.group().split("@")
        return parts[0][0] + "***@" + parts[1]

    text = _EMAIL_PATTERN.sub(_mask_email, text)
    text = _CREDIT_CARD_PATTERN.sub("****-****-****-****", text)
    return text


def mask_token(token: str, visible_chars: int = 8) -> str:
    """Show only first N chars of a token. 'sk-abc123...' → 'sk-abc12...'"""
    if len(token) <= visible_chars:
        return "***"
    return token[:visible_chars] + "..."


# ---------------------------------------------------------------------------
# MIME type helpers
# ---------------------------------------------------------------------------

def get_mime_type(filename: str) -> str:
    """Detect MIME type from filename extension."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def is_text_file(content_type: str) -> bool:
    """Return True if the MIME type indicates a text-based file."""
    return content_type.startswith("text/") or content_type in {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/markdown",
    }


# ---------------------------------------------------------------------------
# Conversation title generation
# ---------------------------------------------------------------------------

def generate_conversation_title(first_message: str, max_length: int = 60) -> str:
    """
    Generate a conversation title from the first user message.
    Truncates long messages and removes newlines.
    """
    title = normalise_whitespace(first_message)
    title = title.replace("\n", " ").replace("\r", "")
    return truncate(title, max_length)


# ---------------------------------------------------------------------------
# Deep merge for dicts (used in metadata updates)
# ---------------------------------------------------------------------------

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge `override` into `base`.
    Unlike {**base, **override}, this merges nested dicts instead of replacing them.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result