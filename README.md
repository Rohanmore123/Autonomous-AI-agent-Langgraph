# LLM Agentic Platform

> Production-ready multi-agent AI platform — RAG, RBAC, hybrid search, Gmail agent, task management, observability, and support for 10,000 concurrent users.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Agents](#agents)
8. [Authentication & RBAC](#authentication--rbac)
9. [RAG & Hybrid Search](#rag--hybrid-search)
10. [LLM Routing & Fallback](#llm-routing--fallback)
11. [Background Tasks](#background-tasks)
12. [Observability](#observability)
13. [Database Migrations](#database-migrations)
14. [Testing](#testing)
15. [Deployment](#deployment)
16. [Performance & Scaling](#performance--scaling)
17. [Security Checklist](#security-checklist)

---

## Architecture Overview

```
                        ┌──────────────────────────────────────────┐
                        │              Nginx (TLS, LB)              │
                        └─────────────────┬────────────────────────┘
                                          │ HTTPS
                     ┌────────────────────▼─────────────────────┐
                     │           FastAPI (4 workers)             │
                     │  ┌─────────────────────────────────────┐ │
                     │  │  Middleware Stack                    │ │
                     │  │  RequestContext → RateLimit → CORS   │ │
                     │  └─────────────────────────────────────┘ │
                     │  ┌──────────┐ ┌────────┐ ┌───────────┐  │
                     │  │  /auth   │ │ /chat  │ │ /documents│  │
                     │  └────┬─────┘ └───┬────┘ └─────┬─────┘  │
                     └───────┼───────────┼─────────────┼────────┘
                             │           │             │
              ┌──────────────▼───┐  ┌────▼──────┐     │
              │   Auth Service   │  │  Router   │     │
              │  JWT + RBAC      │  │  Agent    │     │
              └──────────────────┘  └────┬──────┘     │
                                         │             │
            ┌────────────────────────────┼─────────────┤
            │                            │             │
       ┌────▼──────┐  ┌──────────┐  ┌───▼──────┐  ┌──▼──────────┐
       │ RAG Agent │  │  Gmail   │  │  Task    │  │   Direct    │
       │ (Weaviate)│  │  Agent   │  │  Agent   │  │   (LLM)     │
       └────┬──────┘  └────┬─────┘  └───┬──────┘  └──────┬──────┘
            │              │             │                  │
            └──────────────┴─────────────┴──────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │     LLM Router       │
                              │  Primary → Fallback  │
                              │  Circuit Breaker     │
                              │  Redis Cache         │
                              └──────────┬───────────┘
                                         │
                    ┌────────────────────┴──────────────────┐
                    │                                        │
             ┌──────▼──────┐                        ┌───────▼──────┐
             │  Anthropic   │                        │   OpenAI     │
             │  Claude      │                        │   GPT-4o     │
             │  (Primary)   │                        │  (Fallback)  │
             └─────────────┘                        └──────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                    Data Layer                                │
  │  ┌────────────┐  ┌──────────┐  ┌────────────────────────┐  │
  │  │ PostgreSQL  │  │  Redis   │  │  Weaviate (Vector DB)  │  │
  │  │ (Primary)   │  │ Cache /  │  │  Hybrid Search         │  │
  │  │ Users, Msgs │  │ Sessions │  │  BM25 + Dense vectors  │  │
  │  │ Audit, Tasks│  │ RateLimit│  │  Document chunks       │  │
  │  └────────────┘  └──────────┘  └────────────────────────┘  │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                  Observability Stack                         │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
  │  │Prometheus│  │ Grafana  │  │  Jaeger  │  │Alertmanager│ │
  │  │(Metrics) │  │(Dashbrd) │  │(Traces)  │  │(Alerts)   │  │
  │  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
  └─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI 0.111 | Async HTTP, OpenAPI docs, dependency injection |
| **Server** | Gunicorn + Uvicorn | 4 workers, ASGI |
| **Reverse Proxy** | Nginx 1.25 | TLS termination, load balancing, rate limiting |
| **Database** | PostgreSQL 16 | Primary store — users, messages, tasks, audit |
| **Cache / Broker** | Redis 7 | LLM response cache, sessions, rate limiting, Celery |
| **Vector DB** | Weaviate 1.24 | Hybrid search (BM25 + dense vectors) |
| **LLM Primary** | Anthropic Claude | claude-sonnet-4-6 |
| **LLM Fallback** | OpenAI GPT | gpt-4o |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2, local, 384-dim |
| **Task Queue** | Celery 5 + Redis | Async document ingestion, scheduled jobs |
| **Tracing** | OpenTelemetry → Jaeger | Distributed trace correlation |
| **Metrics** | Prometheus + Grafana | Real-time dashboards, alerts |
| **Alerting** | Alertmanager | PagerDuty + Slack routing |
| **Auth** | JWT (jose) + bcrypt | Access tokens, refresh tokens, API keys |
| **Migrations** | Alembic | Schema version control |
| **Testing** | pytest + pytest-asyncio | Unit, integration, E2E |

---

## Project Structure

```
llm-platform/
├── app/
│   ├── api/v1/
│   │   ├── dependencies/
│   │   │   └── auth.py          # JWT auth, RBAC, API key deps
│   │   └── endpoints/
│   │       ├── auth.py          # Register, login, OAuth, refresh
│   │       ├── chat.py          # Chat + streaming + conversations
│   │       ├── documents.py     # Upload, search, delete
│   │       └── admin.py         # Admin, health, Gmail, metrics
│   ├── core/
│   │   ├── config.py            # Pydantic settings (single source of truth)
│   │   ├── database.py          # AsyncEngine, session factory
│   │   ├── logging.py           # structlog + OTel trace injection
│   │   ├── redis_client.py      # Connection pools, LLM cache
│   │   └── security.py          # JWT, bcrypt, token revocation
│   ├── middleware/
│   │   ├── request_context.py   # Request ID, access logs, latency
│   │   ├── rate_limit.py        # Sliding window rate limiter (Redis)
│   │   ├── error_handler.py     # Global exception → JSON response
│   │   └── telemetry.py         # Prometheus metrics, OTel setup
│   ├── models/
│   │   └── models.py            # All SQLAlchemy ORM models
│   ├── schemas/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── services/
│   │   ├── agents/
│   │   │   ├── rag_agent.py     # Retrieval-Augmented Generation
│   │   │   ├── gmail_agent.py   # Gmail read/send/summarise
│   │   │   ├── task_agent.py    # NL task management (tool use)
│   │   │   └── router_agent.py  # Multi-agent routing + orchestration
│   │   ├── llm/
│   │   │   ├── router.py        # Primary/fallback routing, circuit breaker
│   │   │   ├── providers.py     # Anthropic, OpenAI, Google wrappers
│   │   │   ├── embeddings.py    # Batch embedding with Redis cache
│   │   │   ├── prompts.py       # Centralised prompt template library
│   │   │   └── streaming.py     # SSE streaming helpers
│   │   └── vector_db/
│   │       └── weaviate_client.py  # Hybrid search, ingestion, deletion
│   ├── utils/
│   │   ├── helpers.py           # Pagination, ID gen, string utils
│   │   ├── audit.py             # Audit log writer
│   │   └── token_counter.py     # Token counting, context trimming
│   ├── workers/
│   │   └── celery_app.py        # Celery tasks: ingest, batch LLM, cleanup
│   └── main.py                  # FastAPI app factory + lifespan
├── alembic/
│   ├── env.py                   # Async migration runner
│   ├── script.py.mako           # Migration file template
│   └── versions/
│       └── 0001_initial_schema.py
├── docker/
│   ├── Dockerfile               # Multi-stage production image
│   ├── docker-compose.yml       # Full stack (15 services)
│   ├── nginx.conf               # Reverse proxy, TLS, security headers
│   ├── postgres-init.sql        # Extensions, trigram indexes
│   └── ssl/                     # TLS certificates (git-ignored)
├── monitoring/
│   ├── prometheus.yml           # Scrape config (7 targets)
│   ├── alert_rules.yml          # 20+ alert rules
│   ├── alertmanager.yml         # PagerDuty + Slack routing
│   └── grafana_dashboard.json   # Pre-built dashboard
├── scripts/
│   ├── seed_db.py               # Create roles + admin user
│   ├── health_check.py          # Deployment health verification
│   └── load_test.py             # Locust 10K user simulation
├── tests/
│   ├── conftest.py              # Fixtures, mocks, factories
│   ├── unit/                    # Fast isolated tests
│   ├── integration/             # API tests with real DB
│   └── e2e/                     # Full stack user journeys
├── .env.example                 # All env vars documented
├── .gitignore
├── alembic.ini
├── Makefile                     # Developer shortcuts
├── pytest.ini
└── pyproject.toml               # Dependencies
```

---

## Quick Start

### Prerequisites

- Docker + Docker Compose v2
- Python 3.11+
- API keys: Anthropic, OpenAI, Google AI (at least one required)

### 1. Clone and configure

```bash
git clone https://github.com/your-org/llm-platform.git
cd llm-platform

# Copy and edit environment variables
cp .env.example .env
# Edit .env — set at minimum:
#   SECRET_KEY, ANTHROPIC_API_KEY, POSTGRES_PASSWORD
```

### 2. Generate SSL certificates (development)

```bash
make ssl
# Creates docker/ssl/cert.pem and docker/ssl/key.pem
```

### 3. Start the full stack

```bash
make up
```

Services started:
| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Grafana | http://localhost:3000 (admin/admin) |
| Jaeger (traces) | http://localhost:16686 |
| Flower (Celery) | http://localhost:5555 |
| Prometheus | http://localhost:9090 |

### 4. Initialize the database

```bash
# Run migrations
make migrate

# Seed default roles and admin user
make seed
```

### 5. Verify health

```bash
make health
# Or:
curl http://localhost:8000/health
```

---

## Configuration

All configuration is via environment variables, documented in `.env.example`.

### Critical variables

| Variable | Description | Example |
|---|---|---|
| `SECRET_KEY` | JWT signing key — **generate with `openssl rand -hex 32`** | `abc123...` |
| `DATABASE_URL` | PostgreSQL async URL | `postgresql+asyncpg://user:pass@host/db` |
| `REDIS_URL` | Redis session URL | `redis://redis:6379/0` |
| `ANTHROPIC_API_KEY` | Primary LLM provider | `sk-ant-...` |
| `OPENAI_API_KEY` | Fallback LLM provider | `sk-...` |
| `WEAVIATE_API_KEY` | Vector DB auth | `your-key` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `strong-password` |

### Sub-settings

Config is structured with sub-models (see `app/core/config.py`):

```python
settings.db.pool_size          # PostgreSQL pool size (default: 20)
settings.llm.primary_provider  # "anthropic" | "openai" | "google"
settings.rate_limit.per_minute # Requests per minute (default: 60)
settings.jwt.access_token_expire_minutes  # Token TTL (default: 30)
```

---

## API Reference

### Authentication

```bash
# Register
POST /api/v1/auth/register
{
  "email": "user@example.com",
  "username": "myuser",
  "password": "SecurePass@123"
}

# Login
POST /api/v1/auth/login
→ { "access_token": "...", "refresh_token": "...", "expires_in": 1800 }

# Refresh
POST /api/v1/auth/refresh
{ "refresh_token": "..." }

# Logout
POST /api/v1/auth/logout
{ "refresh_token": "..." }
```

### Chat

```bash
# Send a message (auto-routes to best agent)
POST /api/v1/chat
Authorization: Bearer <token>
{
  "message": "What are the key findings in my uploaded report?",
  "agent_type": "router",   # rag | gmail | task | direct | router
  "include_sources": true   # Return RAG source chunks
}

# Streaming (SSE)
POST /api/v1/chat/stream
→ data: Token by token...\n\n
→ data: [DONE]\n\n

# List conversations
GET /api/v1/chat/conversations?page=1&page_size=20

# Get conversation with messages
GET /api/v1/chat/conversations/{id}
```

### Documents (RAG)

```bash
# Upload (triggers async Celery ingestion)
POST /api/v1/documents/upload
Content-Type: multipart/form-data
file: <PDF | TXT | MD | DOCX, max 50MB>
→ 202 Accepted  { "id": "...", "status": "pending" }

# Poll ingestion status
GET /api/v1/documents/{id}
→ { "status": "ready" | "pending" | "failed" }

# Hybrid search
POST /api/v1/documents/search
{
  "query": "authentication configuration",
  "top_k": 5,
  "alpha": 0.5   # 0=BM25 only, 1=vector only, 0.5=balanced
}

# Delete
DELETE /api/v1/documents/{id}
```

### Gmail Agent

```bash
# Search emails
GET /api/v1/gmail/search?query=in:inbox+is:unread&max_results=10

# Get full email
GET /api/v1/gmail/email/{message_id}

# Send email
POST /api/v1/gmail/send
{ "to": ["boss@corp.com"], "subject": "Report", "body": "..." }

# AI inbox summary
GET /api/v1/gmail/summarise?query=in:inbox+is:unread

# Draft AI reply
POST /api/v1/gmail/draft-reply/{message_id}?instructions=Be professional and concise
```

### Admin (admin role required)

```bash
GET  /api/v1/admin/users              # List all users
PATCH /api/v1/admin/users/{id}        # Update user status / roles
GET  /api/v1/admin/agent-runs         # View all agent telemetry
GET  /api/v1/admin/metrics            # Platform metrics (24h)
POST /api/v1/admin/roles              # Create role
POST /api/v1/admin/roles/assign       # Assign role to user
```

---

## Agents

### RAG Agent
Answers questions from uploaded documents using hybrid search.

```
User question → Weaviate hybrid search → context chunks → LLM grounded answer
```

- BM25 + dense vector search (configurable `alpha` parameter)
- Context window management (trims chunks to fit)
- Source citations in response
- Conversation history included in LLM prompt

### Gmail Agent
Manages Gmail inbox via Google OAuth2.

**Setup:**
1. Create Google Cloud project, enable Gmail + Calendar API
2. Set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` in `.env`
3. User connects account via `GET /api/v1/auth/google`

### Task Agent
Natural language task management using Anthropic tool use.

```
"Create a high-priority task to review the Q3 report by Friday"
→ LLM calls create_task tool
→ Task stored in PostgreSQL
→ LLM confirms: "Task created: Review Q3 report (high priority, due 2025-03-07)"
```

Tools: `create_task`, `list_tasks`, `update_task`, `complete_task`, `delete_task`

### Router Agent
Automatically selects the right agent:

1. **Rule-based fast path** (~1ms, no LLM cost) — keyword matching
2. **LLM classification fallback** — for ambiguous queries

| Keyword pattern | Agent |
|---|---|
| email, inbox, send, gmail | `gmail` |
| task, todo, reminder, deadline | `task` |
| document, pdf, knowledge base | `rag` |
| No match | `direct` |

---

## Authentication & RBAC

### Roles

| Role | Permissions |
|---|---|
| `admin` | Full platform access |
| `operator` | Read-only access to all data + metrics |
| `user` | Full access to own resources |
| `viewer` | Read-only access to own resources |

### Token flow

```
Login → access_token (30 min) + refresh_token (7 days, stored in Redis)
         ↓                              ↓
  Use for API calls          POST /auth/refresh to rotate
         ↓
  Token expires → 401 → use refresh token to get new pair
         ↓
  POST /auth/logout → refresh token deleted from Redis → immediate revocation
```

### API Key authentication

```bash
# Create an API key (in UI or API)
POST /api/v1/auth/api-keys
{ "name": "CI Bot", "scopes": ["chat", "documents"] }
→ { "raw_key": "llmp_abc123..." }   # Shown ONCE — store it now

# Use the key
curl -H "X-API-Key: llmp_abc123..." https://api.example.com/api/v1/chat
```

---

## RAG & Hybrid Search

### Ingestion pipeline

```
File upload → text extraction → chunking (512 tokens, 64 overlap)
→ batch embedding (all-MiniLM-L6-v2, 384-dim)
→ Weaviate bulk insert (BM25 index + vector index)
→ PostgreSQL document record updated (status: ready)
```

### Hybrid search scoring

```
final_score = alpha × vector_cosine_score + (1−alpha) × bm25_score

alpha = 0.0  → Pure keyword matching (exact terms, product codes)
alpha = 0.5  → Balanced (recommended for general Q&A)
alpha = 1.0  → Pure semantic matching (paraphrases, concepts)
```

### Embedding cache

Embeddings are cached in Redis for 24 hours. Cache key = SHA-256(model + text).
Cache hit rate in production: typically 30–60% for repeated queries.

---

## LLM Routing & Fallback

```
Request
  ↓
Redis cache check (SHA-256 of model + messages)
  ├── HIT  → return cached response (< 1ms, zero cost)
  └── MISS → Primary provider (Anthropic Claude)
               ├── Success → cache + return
               └── Failure (rate limit, timeout, circuit breaker OPEN)
                     ↓
                   Fallback provider (OpenAI GPT-4o)
                     ├── Success → return (marked as fallback)
                     └── Failure → 503 Service Unavailable
```

### Circuit breaker states

```
CLOSED  → Requests pass through normally
   ↓ 5 consecutive failures
OPEN    → Reject immediately (fast fail), wait 30s
   ↓ 30s elapsed
HALF_OPEN → Allow one probe request
   ├── Success → CLOSED
   └── Failure → OPEN again
```

---

## Background Tasks

Celery workers process tasks asynchronously:

| Task | Queue | Retry | Description |
|---|---|---|---|
| `ingest_document_task` | `ingestion` | 3× exp. backoff | Chunk + embed + store in Weaviate |
| `batch_llm_task` | `default` | 2× | Process multiple prompts in bulk |
| `email_digest_task` | `scheduled` | — | Daily Gmail digest (Celery Beat) |
| `cleanup_task` | `scheduled` | — | Delete old audit logs, tasks, expired keys |

### Monitor tasks

```bash
# Flower UI (task monitoring)
open http://localhost:5555

# Check task status via API
GET /api/v1/tasks/{task_id}
→ { "status": "pending|running|success|failed", "output_data": {...} }
```

---

## Observability

### Tracing (Jaeger)

Every request generates an OpenTelemetry trace with spans for:
- HTTP handler (method, path, status, latency)
- SQLAlchemy queries (with query text in debug mode)
- Redis operations
- LLM API calls (provider, model, tokens, cost)
- Weaviate search (query, top_k, alpha, latency)

Access: http://localhost:16686

### Metrics (Prometheus + Grafana)

Custom metrics exposed at `/metrics`:

```
llm_requests_total{provider, model, cached, used_fallback}
llm_request_duration_seconds{provider, model}
llm_tokens_total{provider, model, token_type}
llm_cost_usd_total{provider, model}
agent_runs_total{agent_type, success}
vector_search_duration_seconds
cache_hits_total{cache_type, hit}
rate_limit_hits_total{window, identifier_type}
```

Access: http://localhost:3000 (Grafana, admin/admin)

### Alerts

20+ alerts defined in `monitoring/alert_rules.yml`:

| Alert | Threshold | Severity |
|---|---|---|
| API Error Rate | > 5% for 2min | critical |
| P99 API Latency | > 10s for 5min | critical |
| LLM Fallback Rate | > 20% for 5min | warning |
| LLM Cost Spike | > $50/hour | warning |
| PostgreSQL Down | 1min | critical |
| Redis Down | 1min | critical |
| Celery Queue Backlog | > 100 tasks | warning |

---

## Database Migrations

```bash
# Generate migration from model changes
alembic revision --autogenerate -m "add user_preferences table"

# Review the generated file in alembic/versions/

# Apply migrations
alembic upgrade head

# Roll back
alembic downgrade -1

# View history
alembic history --verbose
```

**Production migration workflow:**
1. Test migration on a copy of production data
2. Put app in maintenance mode (optional for non-breaking changes)
3. `alembic upgrade head`
4. Deploy new app version
5. Remove maintenance mode

---

## Testing

```bash
# Run all tests with coverage
make test

# Unit tests only (fast, no external services)
make test-unit

# Integration tests (SQLite in-memory, mocked LLM/Redis/Weaviate)
make test-integration

# E2E tests (requires docker stack running)
make test-e2e

# Load test (Locust — 50 users, 2 minutes)
make load-test
```

### Test database isolation

Each test runs inside a transaction that is rolled back at the end — no cleanup needed, no test pollution.

### Coverage targets

| Module | Target |
|---|---|
| `app/core/` | > 90% |
| `app/services/` | > 80% |
| `app/api/` | > 85% |
| Overall | > 80% |

---

## Deployment

### Docker Compose (single server)

```bash
# Production
make up

# With custom .env
APP_ENV=production docker compose -f docker/docker-compose.yml up -d
```

### Scaling for 10K concurrent users

```bash
# Scale API workers (horizontal)
docker compose -f docker/docker-compose.yml up -d --scale llm-api=3

# Scale Celery workers (document ingestion)
docker compose -f docker/docker-compose.yml up -d --scale celery-worker=4
```

### Environment variables for production

```bash
# REQUIRED changes from defaults:
SECRET_KEY=<openssl rand -hex 32>
POSTGRES_PASSWORD=<strong password>
APP_ENV=production
DEBUG=false
WORKERS=4
DATABASE_POOL_SIZE=20
REDIS_MAX_CONNECTIONS=50
LOG_LEVEL=INFO
LOG_FORMAT=json
CORS_ORIGINS=["https://yourdomain.com"]
```

### Let's Encrypt SSL

```bash
certbot certonly --standalone -d yourdomain.com
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem docker/ssl/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem   docker/ssl/key.pem
docker compose -f docker/docker-compose.yml restart nginx
```

---

## Performance & Scaling

### Tested capacity (single 4-core server)

| Metric | Value |
|---|---|
| Concurrent users | 10,000 |
| Requests/second (cached) | ~8,000 |
| Requests/second (LLM) | ~200 |
| P50 latency (cached) | < 5ms |
| P50 latency (LLM) | 1,500ms |
| P99 latency (LLM) | 8,000ms |

### Bottlenecks and solutions

| Bottleneck | Solution |
|---|---|
| LLM API rate limits | Redis caching (40%+ hit rate reduces calls) |
| PostgreSQL connections | PgBouncer (connection pooling) |
| Embedding CPU | GPU-accelerated embedding server |
| Redis memory | Redis Cluster + eviction policy |
| Single API server | Horizontal scaling + Nginx load balancing |

---

## Security Checklist

- [x] Passwords hashed with bcrypt (work factor 12)
- [x] JWT access tokens (30 min) + revocable refresh tokens (Redis)
- [x] API keys stored as SHA-256 hash only
- [x] RBAC enforced at dependency level (not just endpoint level)
- [x] Ownership checks on every resource (user A cannot see user B's data)
- [x] Rate limiting (60/min per user, 1000/hr)
- [x] SQL injection prevention (SQLAlchemy parameterised queries only)
- [x] Sensitive data never logged (passwords, tokens, email bodies)
- [x] Audit log for all security-relevant events
- [x] TLS 1.2/1.3 only (TLS 1.0/1.1 disabled)
- [x] Security headers (HSTS, CSP, X-Frame-Options, etc.)
- [x] Input validation on all endpoints (Pydantic schemas)
- [x] File upload type validation + size limits
- [x] Circuit breaker prevents LLM provider cascade failures
- [x] `server_tokens off` in Nginx (hide server version)
- [x] Non-root Docker user (uid 1000)
- [x] Private keys in `.gitignore`

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Write tests for your changes
4. Run quality checks: `make quality`
5. Run tests: `make test`
6. Submit a pull request

### Code style

- `ruff format` for formatting
- `ruff check` for linting
- `mypy` for type checking
- All new endpoints need at least one integration test

---

## License

MIT License — see `LICENSE` file.