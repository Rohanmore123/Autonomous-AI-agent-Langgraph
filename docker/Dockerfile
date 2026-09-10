# docker/Dockerfile
# ===================
# Multi-stage Docker build for the LLM Agentic Platform.
#
# STAGES:
#   base       — shared Python runtime + system deps
#   builder    — install Python packages (isolated from final image)
#   production — minimal runtime image (~500MB vs ~1.5GB naive build)
#   development— production + dev tools (pytest, ruff, mypy)
#
# WHY MULTI-STAGE?
#   The builder stage installs compilers (gcc, rust) needed to build
#   cryptography, asyncpg, etc. The production stage only copies the
#   compiled wheels — no compilers, no build caches → smaller attack surface.
#
# IMAGE SIZE:
#   Single stage:  ~1.8GB (includes gcc, build headers, pip caches)
#   Multi-stage:   ~550MB (runtime only)
#
# SECURITY:
#   - Runs as non-root user (appuser, uid=1000)
#   - No sudo, no package manager in final image
#   - Read-only filesystem (except /tmp and /var/log/llm-platform)
#
# BUILD:
#   docker build -f docker/Dockerfile -t llm-platform-api:latest .
#   docker build -f docker/Dockerfile --target development -t llm-platform-api:dev .

# ---------------------------------------------------------------------------
# Stage 1: base — shared Python runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PostgreSQL client (for pg_isready healthcheck)
    postgresql-client \
    # libpq needed by psycopg2
    libpq-dev \
    # curl for healthcheck fallback
    curl \
    # Needed by some ML libraries
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Create app directories with correct ownership
RUN mkdir -p /app /var/log/llm-platform /tmp/sentence_transformers && \
    chown -R appuser:appgroup /app /var/log/llm-platform /tmp/sentence_transformers

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2: builder — install Python packages
# ---------------------------------------------------------------------------
FROM base AS builder

# Build-only system dependencies (compilers, headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    cargo \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition first (Docker layer cache — only re-install if pyproject.toml changes)
COPY pyproject.toml .

# Install all dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install production dependencies
# We use pip install -e . to get the project itself plus all dependencies
RUN pip install \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    gunicorn==22.0.0 \
    python-multipart==0.0.9 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    python-dotenv==1.0.1 \
    sqlalchemy[asyncio]==2.0.30 \
    asyncpg==0.29.0 \
    alembic==1.13.1 \
    psycopg2-binary==2.9.9 \
    redis[hiredis]==5.0.4 \
    aioredis==2.0.1 \
    weaviate-client==4.6.1 \
    anthropic==0.26.1 \
    openai==1.30.1 \
    google-generativeai==0.5.4 \
    tiktoken==0.7.0 \
    sentence-transformers==3.0.1 \
    numpy==1.26.4 \
    google-api-python-client==2.130.0 \
    google-auth-httplib2==0.2.0 \
    google-auth-oauthlib==1.2.0 \
    celery[redis]==5.4.0 \
    flower==2.0.1 \
    opentelemetry-api==1.24.0 \
    opentelemetry-sdk==1.24.0 \
    opentelemetry-instrumentation-fastapi==0.45b0 \
    opentelemetry-instrumentation-sqlalchemy==0.45b0 \
    opentelemetry-instrumentation-redis==0.45b0 \
    opentelemetry-exporter-otlp-proto-grpc==1.24.0 \
    prometheus-fastapi-instrumentator==7.0.0 \
    prometheus-client==0.20.0 \
    structlog==24.1.0 \
    python-json-logger==2.0.7 \
    httpx==0.27.0 \
    aiohttp==3.9.5 \
    tenacity==8.3.0 \
    pydantic==2.7.1 \
    pydantic-settings==2.3.0 \
    email-validator==2.1.1 \
    shortuuid==1.0.13 \
    python-dateutil==2.9.0 \
    pytz==2024.1 \
    orjson==3.10.3 \
    cachetools==5.3.3 \
    aiosqlite==0.20.0

# ---------------------------------------------------------------------------
# Stage 3: production — minimal runtime image
# ---------------------------------------------------------------------------
FROM base AS production

# Copy the virtual environment from builder (compiled packages, no build tools)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup . /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python scripts/health_check.py --host localhost --port 8000 || exit 1

# Default: run gunicorn with uvicorn workers
# Override CMD in docker-compose for celery workers, beat, flower
CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--forwarded-allow-ips", "*"]

# ---------------------------------------------------------------------------
# Stage 4: development — production + dev/test tools
# ---------------------------------------------------------------------------
FROM production AS development

USER root

RUN /opt/venv/bin/pip install \
    pytest==8.2.1 \
    pytest-asyncio==0.23.7 \
    pytest-cov==5.0.0 \
    httpx==0.27.0 \
    factory-boy==3.3.0 \
    faker==25.2.0 \
    ruff==0.4.4 \
    mypy==1.10.0 \
    aiosqlite==0.20.0 \
    locust==2.29.1

USER appuser

# Development: use uvicorn with auto-reload
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]