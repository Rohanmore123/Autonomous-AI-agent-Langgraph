"""
app/core/logging.py
===================
Production-grade structured logging with:
  • structlog — for context-rich, key=value JSON logs
  • OpenTelemetry trace/span injection — every log line includes trace_id + span_id
    so you can jump from a log entry directly to its trace in Jaeger/Tempo.
  • Request-ID injection — correlation across microservices.
  • Log levels controlled by LOG_LEVEL env var.

WHY STRUCTLOG?
  Standard Python `logging` produces flat strings.  structlog produces
  JSON objects that log aggregators (Loki, Elastic, Datadog) can index
  field-by-field, enabling fast queries like:
      {user_id="u-123", level="error"}
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from opentelemetry import trace

from app.core.config import get_settings

settings = get_settings()


def _add_otel_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    structlog processor — injects current OpenTelemetry trace context.
    Runs on every log call so every line is traceable.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def _add_app_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Inject static app metadata into every log entry."""
    event_dict["service"] = settings.observability.otel_service_name
    event_dict["version"] = settings.app_version
    event_dict["env"] = settings.app_env
    return event_dict


def setup_logging() -> None:
    """
    Configure structlog and standard library logging together.
    Call once at application startup.

    PROCESSOR CHAIN (runs in order):
      1. stdlib integration — captures logs from third-party libs (SQLAlchemy, etc.)
      2. Log level filter
      3. OTel trace context injection
      4. App context injection
      5. ISO timestamp
      6. Stack info for exceptions
      7. Renderer: JSON (production) or colored console (development)
    """
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = log_level_map.get(settings.log_level.upper(), logging.INFO)
    
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_otel_context,
        _add_app_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,  # Performance: bind once per logger instance
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    # Quiet noisy libraries in production
    if settings.is_production:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Factory for named loggers.
    Usage:
        logger = get_logger(__name__)
        logger.info("user.created", user_id=user.id, email=user.email)
    """
    return structlog.get_logger(name)