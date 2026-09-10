"""
app/workers/celery_app.py
==========================
Celery task queue — background processing engine.

WHY CELERY?
  LLM-powered tasks (document ingestion, batch summarisation, email digest)
  can take 10–120 seconds. HTTP requests must return in <30s or proxies timeout.
  Celery moves long work to background workers:
    API server  → submits task → Redis broker
    Celery worker → picks task → executes → stores result in Redis backend
    API server  → client polls GET /tasks/{id} for status

WORKER ARCHITECTURE:
  ┌──────────┐    submit    ┌───────┐    pick    ┌────────────────┐
  │ FastAPI  │ ──────────→ │ Redis │ ─────────→ │ Celery Worker  │
  │ Server   │ ←────────── │ Broker│ ←───────── │ (separate proc)│
  └──────────┘   task_id   └───────┘   result   └────────────────┘

TASK CATEGORIES:
  document:ingest    — chunk + embed + write to Weaviate
  document:reindex   — re-embed all chunks (after model change)
  email:digest       — scheduled Gmail summary
  llm:batch          — process a batch of prompts (bulk API use case)
  maintenance:cleanup— delete expired data, vacuum old audit logs

RETRY STRATEGY:
  max_retries=3, countdown=exponential backoff (60s, 120s, 240s)
  On permanent failure: task status → "failed", error stored in Task row.

ROUTING:
  Different queues for different task types allow independent scaling:
    default    → general tasks (1 worker)
    ingestion  → document processing (CPU-bound, 2+ workers)
    scheduled  → cron jobs (1 worker)
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone

from celery import Celery
from celery.signals import task_failure, task_postrun, task_prerun
from celery.utils.log import get_task_logger

from app.core.config import get_settings

settings = get_settings()
logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

celery_app = Celery(
    "llm_platform",
    broker=settings.redis.celery_url,
    backend=settings.redis.celery_url,
)

celery_app.conf.update(
    # Serialisation — JSON is safe across Python versions; never use pickle
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Time limits
    task_soft_time_limit=300,   # 5 min — task receives SoftTimeLimitExceeded, can clean up
    task_time_limit=600,        # 10 min — hard kill

    # Result expiry (don't fill Redis forever)
    result_expires=86400,       # 24 hours

    # Reliability
    task_acks_late=True,        # Ack AFTER task completes (not before) → no lost tasks on crash
    task_reject_on_worker_lost=True,  # Re-queue if worker dies mid-task

    # Queues
    task_routes={
        "app.workers.celery_app.ingest_document_task": {"queue": "ingestion"},
        "app.workers.celery_app.batch_llm_task":       {"queue": "default"},
        "app.workers.celery_app.email_digest_task":    {"queue": "scheduled"},
        "app.workers.celery_app.cleanup_task":         {"queue": "scheduled"},
    },
    task_default_queue="default",

    # Concurrency
    worker_concurrency=4,
    worker_prefetch_multiplier=1,  # Fetch 1 task at a time — prevents queue starvation

    # Beat scheduler (cron jobs)
    beat_schedule={
        "daily-cleanup": {
            "task": "app.workers.celery_app.cleanup_task",
            "schedule": 86400,   # every 24 hours
        },
    },

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)


# ---------------------------------------------------------------------------
# Signals — update Task DB record on state changes
# ---------------------------------------------------------------------------

def _get_sync_db_session():
    """
    Synchronous DB session for use in Celery tasks.
    Celery workers run in their own process/thread — no async event loop.
    We use the synchronous SQLAlchemy engine here.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Convert asyncpg URL to psycopg2 URL for sync use
    sync_url = settings.db.url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    engine = create_engine(sync_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    return Session()


def _update_task_status(task_id: str, status: str, **kwargs) -> None:
    """Update Task row in PostgreSQL synchronously."""
    try:
        from app.models.models import Task
        db = _get_sync_db_session()
        task = db.query(Task).filter(Task.celery_task_id == task_id).first()
        if task:
            task.status = status
            for k, v in kwargs.items():
                setattr(task, k, v)
            db.commit()
        db.close()
    except Exception as exc:
        logger.error(f"Failed to update task status: {exc}")


@task_prerun.connect
def task_prerun_handler(task_id: str, task, *args, **kwargs) -> None:
    _update_task_status(
        task_id,
        "running",
        started_at=datetime.now(timezone.utc),
    )


@task_postrun.connect
def task_postrun_handler(task_id: str, task, state: str, *args, **kwargs) -> None:
    if state == "SUCCESS":
        _update_task_status(
            task_id,
            "success",
            completed_at=datetime.now(timezone.utc),
        )


@task_failure.connect
def task_failure_handler(task_id: str, exception, *args, **kwargs) -> None:
    _update_task_status(
        task_id,
        "failed",
        completed_at=datetime.now(timezone.utc),
        error_message=str(exception)[:2000],
    )


# ---------------------------------------------------------------------------
# Task: Document Ingestion
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.celery_app.ingest_document_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,      # seconds before first retry
    queue="ingestion",
)
def ingest_document_task(
    self,
    document_id: str,
    owner_id: str,
    filename: str,
    text_content: str,
    metadata: dict | None = None,
) -> dict:
    """
    Background task: chunk + embed + store document in Weaviate.
    Runs in a separate worker process — no async needed.

    Retry logic:
      On transient failures (network, Weaviate unavailable) → retry with backoff.
      On permanent failures (invalid content) → mark as failed, don't retry.
    """
    import asyncio

    logger.info(f"Starting ingestion for document {document_id}")

    try:
        # Run async ingestion code in a new event loop (worker is sync process)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _ingest():
            from app.services.vector_db.weaviate_client import ingest_document
            return await ingest_document(
                document_id=document_id,
                owner_id=owner_id,
                filename=filename,
                content=text_content,
                metadata=metadata,
            )

        weaviate_ids = loop.run_until_complete(_ingest())
        loop.close()

        # Update document record with chunk count and status
        db = _get_sync_db_session()
        from app.models.models import Document
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "ready"
            doc.chunk_count = len(weaviate_ids)
            db.commit()

            # Store chunk records
            from app.models.models import DocumentChunk
            for idx, wid in enumerate(weaviate_ids):
                chunk = DocumentChunk(
                    document_id=document_id,
                    weaviate_id=wid,
                    chunk_index=idx,
                    content="",       # Content is in Weaviate; don't duplicate in PG
                    token_count=0,
                )
                db.add(chunk)
            db.commit()
        db.close()

        logger.info(f"Ingestion complete: {document_id}, {len(weaviate_ids)} chunks")
        return {"document_id": document_id, "chunks": len(weaviate_ids), "status": "ready"}

    except Exception as exc:
        logger.error(f"Ingestion failed for {document_id}: {exc}\n{traceback.format_exc()}")

        # Update document as failed
        try:
            db = _get_sync_db_session()
            from app.models.models import Document
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.status = "failed"
                doc.error_message = str(exc)[:1000]
                db.commit()
            db.close()
        except Exception:
            pass

        # Retry for transient errors, give up for permanent ones
        if isinstance(exc, (ConnectionError, TimeoutError)):
            retry_delay = 60 * (2 ** self.request.retries)  # exponential backoff
            raise self.retry(exc=exc, countdown=retry_delay)

        raise


# ---------------------------------------------------------------------------
# Task: Batch LLM Processing
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.celery_app.batch_llm_task",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    queue="default",
)
def batch_llm_task(
    self,
    prompts: list[str],
    system_prompt: str,
    user_id: str,
    task_db_id: str,
) -> dict:
    """
    Process a batch of LLM prompts.
    Useful for bulk operations: "summarise these 50 documents overnight".
    Returns list of responses in same order as prompts.
    """
    import asyncio

    results = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _process_all():
        from app.services.llm.router import llm_router
        responses = []
        for i, prompt in enumerate(prompts):
            try:
                resp = await llm_router.complete(
                    messages=[{"role": "user", "content": prompt}],
                    system=system_prompt,
                    use_cache=True,
                )
                responses.append({"index": i, "content": resp.content, "success": True})
            except Exception as e:
                responses.append({"index": i, "error": str(e), "success": False})
        return responses

    try:
        results = loop.run_until_complete(_process_all())
        loop.close()

        # Store results in Task.output_data
        db = _get_sync_db_session()
        from app.models.models import Task
        task = db.query(Task).filter(Task.id == task_db_id).first()
        if task:
            task.output_data = {"results": results, "total": len(results)}
            db.commit()
        db.close()

        return {"task_id": task_db_id, "processed": len(results)}

    except Exception as exc:
        loop.close()
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: Email Digest (scheduled)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.celery_app.email_digest_task",
    queue="scheduled",
)
def email_digest_task() -> dict:
    """
    Scheduled task: generate Gmail digest for all users with Google connected.
    Runs daily via Celery Beat.
    Stores digest in the user's conversation history for later retrieval.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        from app.core.database import AsyncSessionFactory
        from app.models.models import GoogleOAuthToken
        from sqlalchemy import select

        processed = 0
        async with AsyncSessionFactory() as db:
            result = await db.execute(select(GoogleOAuthToken))
            tokens = result.scalars().all()

            for token in tokens:
                try:
                    from app.services.agents.gmail_agent_2 import GmailAgent
                    agent = GmailAgent(token.user_id, db)
                    summary = await agent.summarise_inbox("in:inbox is:unread newer_than:1d")
                    logger.info(f"Email digest generated for user {token.user_id}")
                    processed += 1
                except Exception as exc:
                    logger.error(f"Email digest failed for {token.user_id}: {exc}")

        return processed

    try:
        count = loop.run_until_complete(_run())
        loop.close()
        return {"users_processed": count}
    except Exception as exc:
        loop.close()
        logger.error(f"Email digest task failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Task: Maintenance Cleanup
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.celery_app.cleanup_task",
    queue="scheduled",
)
def cleanup_task() -> dict:
    """
    Daily cleanup:
      • Delete audit_logs older than 90 days
      • Delete completed tasks older than 30 days
      • Purge expired API keys
    """
    from datetime import timedelta

    db = _get_sync_db_session()
    now = datetime.now(timezone.utc)

    # Audit logs: 90-day retention
    from app.models.models import AuditLog, Task as TaskModel, APIKey
    from sqlalchemy import delete

    audit_cutoff = now - timedelta(days=90)
    deleted_audits = db.execute(
        delete(AuditLog).where(AuditLog.created_at < audit_cutoff)
    ).rowcount

    # Old completed tasks: 30-day retention
    task_cutoff = now - timedelta(days=30)
    deleted_tasks = db.execute(
        delete(TaskModel).where(
            TaskModel.status.in_(["success", "failed"]),
            TaskModel.created_at < task_cutoff,
        )
    ).rowcount

    # Expired API keys
    expired_keys = db.execute(
        delete(APIKey).where(
            APIKey.expires_at < now,
            APIKey.is_active == True,  # noqa
        )
    ).rowcount

    db.commit()
    db.close()

    logger.info(
        f"Cleanup: {deleted_audits} audit logs, {deleted_tasks} tasks, {expired_keys} API keys"
    )
    return {
        "deleted_audit_logs": deleted_audits,
        "deleted_tasks": deleted_tasks,
        "expired_api_keys": expired_keys,
    }