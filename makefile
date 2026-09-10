# Makefile — LLM Agentic Platform developer shortcuts
# =======================================================
# Usage: make <target>
# Run `make help` to see all targets.

.PHONY: help install dev up down logs test test-unit test-integration test-e2e \
        lint format typecheck migrate seed shell clean rebuild docs

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN  := \033[0;32m
YELLOW := \033[1;33m
RESET  := \033[0m

help: ## Show this help message
	@echo ""
	@echo "$(GREEN)LLM Agentic Platform — Makefile$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Install all Python dependencies (dev mode)
	pip install -e ".[dev]"
	pip install aiosqlite

ssl: ## Generate self-signed SSL certs for development
	chmod +x docker/ssl/gen_dev_certs.sh
	./docker/ssl/gen_dev_certs.sh

env: ## Copy .env.example to .env (edit before running)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN).env created — edit it with your API keys$(RESET)"; \
	else \
		echo "$(YELLOW).env already exists$(RESET)"; \
	fi

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

up: ## Start the full stack (all services)
	docker compose -f docker/docker-compose.yml up -d
	@echo "$(GREEN)Stack started. API: http://localhost:8000$(RESET)"
	@echo "  Grafana:    http://localhost:3000  (admin/admin)"
	@echo "  Jaeger:     http://localhost:16686"
	@echo "  Flower:     http://localhost:5555"
	@echo "  Prometheus: http://localhost:9090"

up-dev: ## Start in development mode (with hot reload)
	docker compose -f docker/docker-compose.yml up -d postgres redis weaviate jaeger
	APP_ENV=development uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

down: ## Stop all services
	docker compose -f docker/docker-compose.yml down

down-v: ## Stop all services AND remove volumes (wipes data!)
	docker compose -f docker/docker-compose.yml down -v
	@echo "$(YELLOW)All data volumes removed$(RESET)"

rebuild: ## Rebuild Docker image and restart
	docker compose -f docker/docker-compose.yml build --no-cache llm-api
	docker compose -f docker/docker-compose.yml up -d llm-api

logs: ## Tail API server logs
	docker compose -f docker/docker-compose.yml logs -f llm-api

logs-all: ## Tail all service logs
	docker compose -f docker/docker-compose.yml logs -f

ps: ## Show running containers
	docker compose -f docker/docker-compose.yml ps

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

migrate: ## Run Alembic migrations (upgrade to head)
	alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new MSG="add user pref")
	alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Rollback last migration
	alembic downgrade -1

migrate-history: ## Show migration history
	alembic history --verbose

seed: ## Seed database with default roles and admin user
	python scripts/seed_db.py

db-shell: ## Open psql shell in the postgres container
	docker compose -f docker/docker-compose.yml exec postgres \
		psql -U llm_user -d llm_platform

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run all tests (unit + integration, skip e2e)
	pytest tests/unit tests/integration -v --tb=short \
		--cov=app --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/unit -v --tb=short -m "not e2e"

test-integration: ## Run integration tests only
	pytest tests/integration -v --tb=short

test-e2e: ## Run end-to-end tests (requires full stack)
	RUN_E2E=true pytest tests/e2e -v --tb=short

test-fast: ## Run tests without coverage (faster)
	pytest tests/unit tests/integration -q

test-watch: ## Re-run tests on file change (requires pytest-watch)
	ptw tests/unit tests/integration -- -q

load-test: ## Run Locust load test against local server
	locust -f scripts/load_test.py \
		--headless --users 50 --spawn-rate 5 --run-time 2m \
		--host http://localhost:8000 --html load_report.html

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint: ## Run ruff linter
	ruff check app tests

format: ## Auto-format code with ruff
	ruff format app tests

format-check: ## Check formatting without modifying files
	ruff format --check app tests

typecheck: ## Run mypy type checker
	mypy app --ignore-missing-imports --no-strict-optional

quality: lint format-check typecheck ## Run all quality checks

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

run: ## Run the API server locally (development)
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

run-prod: ## Run with gunicorn (production-like, 4 workers)
	gunicorn app.main:app \
		--workers 4 \
		--worker-class uvicorn.workers.UvicornWorker \
		--bind 0.0.0.0:8000 \
		--timeout 120

worker: ## Start a Celery worker
	celery -A app.workers.celery_app worker \
		--loglevel=info --concurrency=2 --queues=default,ingestion

beat: ## Start the Celery beat scheduler
	celery -A app.workers.celery_app beat --loglevel=info

flower: ## Start Celery Flower monitoring UI
	celery -A app.workers.celery_app flower --port=5555

health: ## Check platform health
	python scripts/health_check.py --verbose

shell: ## Open an iPython shell with app context
	python -c "import asyncio; from app.core.database import AsyncSessionFactory; print('DB ready')"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove cache files and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	@echo "$(GREEN)Cleaned$(RESET)"

clean-all: clean down-v ## Clean everything including Docker volumes