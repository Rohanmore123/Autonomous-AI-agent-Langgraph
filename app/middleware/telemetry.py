"""
app/middleware/telemetry.py
============================
Observability bootstrap — OpenTelemetry + Prometheus.

OPENTELEMETRY (Distributed Tracing):
  Every request generates a trace with spans for:
    • HTTP handler (via FastAPI auto-instrumentation)
    • SQLAlchemy queries (via SQLAlchemy auto-instrumentation)
    • Redis operations (via Redis auto-instrumentation)
    • Custom spans for LLM calls, embedding, Weaviate search

  Traces are exported to Jaeger (via OTLP gRPC).
  Jaeger UI: http://localhost:16686 — view full request traces,
  identify slow DB queries, see LLM latency breakdown.

  Trace context propagates across services via W3C TraceContext headers
  (traceparent / tracestate) — standard, works with any OTEL-compatible service.

PROMETHEUS (Metrics):
  prometheus_fastapi_instrumentator auto-exposes:
    • http_requests_total{method, path, status}
    • http_request_duration_seconds{method, path}

  We add CUSTOM metrics:
    • llm_requests_total{provider, model, cached, fallback}
    • llm_request_duration_seconds{provider, model}
    • llm_tokens_total{provider, model, type}   (type = input|output)
    • agent_runs_total{agent_type, success}
    • vector_search_duration_seconds
    • active_connections (gauge)

  Prometheus scrapes /metrics every 15s.
  Grafana reads Prometheus and renders dashboards.

SETUP ORDER (must match main.py):
  1. setup_tracing()   — before app starts accepting requests
  2. setup_metrics(app) — after app is created
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Prometheus metrics
# ---------------------------------------------------------------------------

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM API calls",
    labelnames=["provider", "model", "cached", "used_fallback"],
)

LLM_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM API call latency",
    labelnames=["provider", "model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total LLM tokens processed",
    labelnames=["provider", "model", "token_type"],  # token_type: input|output
)

LLM_COST = Counter(
    "llm_cost_usd_total",
    "Total LLM cost in USD",
    labelnames=["provider", "model"],
)

AGENT_RUNS = Counter(
    "agent_runs_total",
    "Total agent invocations",
    labelnames=["agent_type", "success"],
)

VECTOR_SEARCH_LATENCY = Histogram(
    "vector_search_duration_seconds",
    "Weaviate hybrid search latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

ACTIVE_CONNECTIONS = Gauge(
    "active_http_connections",
    "Number of currently active HTTP connections",
)

CACHE_HIT = Counter(
    "cache_hits_total",
    "Cache hit/miss counts",
    labelnames=["cache_type", "hit"],  # cache_type: llm|embedding
)

DOCUMENT_INGESTIONS = Counter(
    "document_ingestions_total",
    "Document ingestion attempts",
    labelnames=["status"],  # pending|success|failed
)

RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total",
    "Rate limit rejections",
    labelnames=["window", "identifier_type"],  # window: per_min|per_hour
)


# ---------------------------------------------------------------------------
# Convenience functions called from application code
# ---------------------------------------------------------------------------

def record_llm_call(
    provider: str,
    model: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    cached: bool,
    used_fallback: bool,
) -> None:
    """Call this after every LLM response to update all related metrics."""
    cached_str = str(cached).lower()
    fallback_str = str(used_fallback).lower()

    LLM_REQUESTS.labels(
        provider=provider, model=model,
        cached=cached_str, used_fallback=fallback_str,
    ).inc()

    LLM_LATENCY.labels(provider=provider, model=model).observe(latency_ms / 1000)
    LLM_TOKENS.labels(provider=provider, model=model, token_type="input").inc(input_tokens)
    LLM_TOKENS.labels(provider=provider, model=model, token_type="output").inc(output_tokens)
    LLM_COST.labels(provider=provider, model=model).inc(cost_usd)

    if cached:
        CACHE_HIT.labels(cache_type="llm", hit="true").inc()
    else:
        CACHE_HIT.labels(cache_type="llm", hit="false").inc()


def record_agent_run(agent_type: str, success: bool) -> None:
    AGENT_RUNS.labels(agent_type=agent_type, success=str(success).lower()).inc()


def record_vector_search(latency_ms: float) -> None:
    VECTOR_SEARCH_LATENCY.observe(latency_ms / 1000)


def record_rate_limit_hit(window: str, identifier_type: str) -> None:
    RATE_LIMIT_HITS.labels(window=window, identifier_type=identifier_type).inc()


# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------

def setup_tracing() -> None:
    """
    Initialise OpenTelemetry with OTLP gRPC exporter.
    Call ONCE before the FastAPI app starts.

    Trace pipeline:
      Application → BatchSpanProcessor → OTLPSpanExporter → Jaeger
      (BatchSpanProcessor buffers spans and exports in batches — low overhead)
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    import os
    if os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        logger.info("tracing.disabled", reason="OTEL_SDK_DISABLED=true")
        return
    # Service metadata attached to every span
    resource = Resource.create({
        "service.name":    settings.observability.otel_service_name,
        "service.version": settings.observability.otel_service_version,
        "deployment.environment": settings.observability.otel_environment,
    })

    provider = TracerProvider(resource=resource)

    # OTLP exporter → Jaeger (or any OTLP-compatible collector)
    exporter = OTLPSpanExporter(
        endpoint=settings.observability.otel_endpoint,
        insecure=True,   # Use TLS in production with proper certs
    )

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument libraries (zero code change in business logic)
    SQLAlchemyInstrumentor().instrument()
    RedisInstrumentor().instrument()

    logger.info(
        "tracing.initialized",
        endpoint=settings.observability.otel_endpoint,
        service=settings.observability.otel_service_name,
    )


def setup_metrics(app: FastAPI) -> None:
    """
    Attach Prometheus metrics to the FastAPI app.
    Exposes /metrics endpoint consumed by Prometheus scraper.
    Call AFTER the FastAPI app is created.
    """
    if not settings.observability.prometheus_enabled:
        return

    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/health", "/metrics"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics", tags=["Monitoring"])

    logger.info("prometheus.metrics.exposed", endpoint="/metrics")


def setup_fastapi_tracing(app: FastAPI) -> None:
    """Wire FastAPI auto-instrumentation. Call after app creation."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,metrics",
    )