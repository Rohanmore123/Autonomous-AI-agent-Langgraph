"""
app/middleware/request_context.py
===================================
Request lifecycle middleware that does three things per request:

1. REQUEST ID INJECTION
   Generates a UUID for every request (or reads X-Request-ID if supplied).
   Attaches it to:
     • Response header:  X-Request-ID  (client can correlate log lines)
     • structlog context: request_id   (appears in every log line for this request)
     • OpenTelemetry span attribute     (appears in every trace span)

   WHY REQUEST IDs?
   When 10K users are hammering the system, logs interleave.
   Without request IDs, debugging a specific failed request is nearly impossible.
   With them: grep 'request_id=abc-123' in Kibana → see every log line for that request.

2. STRUCTURED ACCESS LOGGING
   Logs method, path, status, latency, user_id, request_id as structured JSON.
   This feeds log aggregators (Loki, Elastic) for dashboards and alerts.

3. LATENCY TRACKING
   Measures wall-clock time from request receipt to response sent.
   Attaches X-Process-Time header for client-side performance monitoring.
   Emits a Prometheus histogram metric (handled by prometheus_fastapi_instrumentator,
   but we add a custom LLM-specific histogram here).
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Must be the FIRST middleware added so request_id is available
    to all downstream middleware and endpoints.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # ---- 1. Request ID ----
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store on request state so endpoints can access it
        request.state.request_id = request_id
        request.state.start_time = time.perf_counter()

        # Bind to structlog context for this coroutine chain
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        # ---- 2. Start OTel span ----
        with tracer.start_as_current_span(
            name=f"{request.method} {request.url.path}",
            attributes={
                "http.method":     request.method,
                "http.url":        str(request.url),
                "http.request_id": request_id,
            },
        ) as span:
            # ---- 3. Process request ----
            try:
                response = await call_next(request)
            except Exception as exc:
                span.record_exception(exc)
                span.set_attribute("http.status_code", 500)
                logger.error(
                    "request.unhandled_exception",
                    error=str(exc),
                    exc_info=True,
                )
                raise

            # ---- 4. Compute latency ----
            elapsed_ms = (time.perf_counter() - request.state.start_time) * 1000

            # ---- 5. Attach response headers ----
            response.headers["X-Request-ID"]    = request_id
            response.headers["X-Process-Time"]  = f"{elapsed_ms:.2f}ms"

            # ---- 6. OTel span attributes ----
            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.latency_ms",  round(elapsed_ms, 2))

            # ---- 7. Access log ----
            # Extract user_id if available (set by auth dependency)
            user_id = getattr(request.state, "user_id", None)

            log_fn = logger.warning if response.status_code >= 400 else logger.info
            log_fn(
                "http.request",
                status=response.status_code,
                latency_ms=round(elapsed_ms, 2),
                user_id=user_id,
                ip=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent", "")[:100],
            )

            return response