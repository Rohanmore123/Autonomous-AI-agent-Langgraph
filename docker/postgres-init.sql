-- docker/postgres-init.sql
-- ===========================
-- Runs ONCE when the PostgreSQL container is first created.
-- Sets up extensions, performance settings, and any DB-level config
-- that cannot be done via Alembic (which runs per-application).
--
-- EXECUTION ORDER:
--   1. Docker entrypoint runs this script as superuser (postgres)
--   2. Then Alembic migrations create tables
--   3. Then seed_db.py inserts initial data
--
-- EXTENSIONS INSTALLED:
--   uuid-ossp   — uuid_generate_v4() for UUID primary keys (legacy; we use gen_random_uuid())
--   pgcrypto    — gen_random_uuid(), pgp_sym_encrypt for token encryption at rest
--   pg_trgm     — Trigram indexes for fast ILIKE / fuzzy text search on email/username
--   btree_gin   — GIN indexes on JSONB + btree columns (for compound queries)
--   pg_stat_statements — Track slow queries (visible via pg_stat_statements view)

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------

-- All tables go in the public schema (default).
-- In multi-tenant setups, use a schema-per-tenant pattern.

-- ---------------------------------------------------------------------------
-- Performance settings (session-level; global settings are in docker-compose CMD)
-- ---------------------------------------------------------------------------

-- Ensure UTF-8 is used for all client connections
SET client_encoding = 'UTF8';

-- ---------------------------------------------------------------------------
-- Trigram indexes (created AFTER Alembic runs tables — done here as reference)
-- The actual CREATE INDEX statements should be in an Alembic migration.
-- This section documents what indexes are needed for the pg_trgm extension.
-- ---------------------------------------------------------------------------

-- NOTE: These are created by Alembic migration, documented here for clarity:
--
-- CREATE INDEX CONCURRENTLY ix_users_email_trgm
--   ON users USING GIN (email gin_trgm_ops);
--
-- CREATE INDEX CONCURRENTLY ix_users_username_trgm
--   ON users USING GIN (username gin_trgm_ops);
--
-- CREATE INDEX CONCURRENTLY ix_messages_content_trgm
--   ON messages USING GIN (content gin_trgm_ops);
--
-- CREATE INDEX CONCURRENTLY ix_audit_logs_action_trgm
--   ON audit_logs USING GIN (action gin_trgm_ops);

-- ---------------------------------------------------------------------------
-- Connection pooling advisory
-- ---------------------------------------------------------------------------

-- This comment documents the pg_hba.conf equivalent settings assumed:
-- The application connects as 'llm_user' with password auth (md5/scram-sha-256).
-- PgBouncer should sit between the app and PostgreSQL in production for
-- connection pooling beyond what SQLAlchemy's pool_size provides.
--
-- Recommended PgBouncer config (not automated here):
--   pool_mode = transaction
--   max_client_conn = 10000
--   default_pool_size = 100
--   reserve_pool_size = 20

-- ---------------------------------------------------------------------------
-- Read replica routing hint (for future use)
-- ---------------------------------------------------------------------------

-- When read replicas are added, analytics queries (AgentRun stats, audit logs)
-- should be routed to replicas via SQLAlchemy's engine routing:
--   engine = create_engine(..., execution_options={"postgresql_readonly": True})
-- This file documents that intention.

-- ---------------------------------------------------------------------------
-- Verification
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL init complete. Extensions: uuid-ossp, pgcrypto, pg_trgm, btree_gin, pg_stat_statements';
END
$$;