"""
app/api/v1/endpoints/documents.py
==================================
Document management endpoints for RAG:

  POST   /documents/upload   — upload a file; triggers async Celery ingestion job
  GET    /documents          — list user's documents
  GET    /documents/{id}     — document details + ingestion status
  DELETE /documents/{id}     — delete document from DB + Weaviate
  POST   /documents/search   — hybrid search across user's documents

ASYNC INGESTION PATTERN:
  File uploads return immediately (202 Accepted) with a task_id.
  The actual chunking + embedding + Weaviate write runs in a Celery worker.
  This prevents large files from timing out the HTTP request.
  Clients poll GET /documents/{id} until status == "ready".

  Why Celery for document ingestion?
    • Files can be 100MB+ PDFs — processing takes 10–120 seconds.
    • HTTP requests time out after 30–60s in most reverse proxies.
    • Celery workers scale independently from API servers.
    • Failed ingestions are retried automatically.

OWNERSHIP ENFORCEMENT:
  Every document query filters on owner_id = current_user.id.
  No cross-user document access is possible at the SQL level.
"""

from __future__ import annotations

import io
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.dependencies.auth import CurrentUser
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.models.models import Document, DocumentChunk
from app.schemas.schemas import (
    APIResponse,
    DocumentResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    PaginatedResponse,
    SourceChunk,
)
from app.services.vector_db.weaviate_client import delete_document_chunks, hybrid_search

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_FILE_SIZE_MB = 50


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=APIResponse[DocumentResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for RAG ingestion",
)
async def upload_document(
    current_user: CurrentUser,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse[DocumentResponse]:
    """
    Upload a document file. Returns immediately with status='pending'.
    Ingestion (chunking + embedding + Weaviate write) runs in background.

    Supported types: PDF, TXT, Markdown, DOCX
    Max size: 50 MB
    """
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    # Read file and check size
    content_bytes = await file.read()
    size_mb = len(content_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {size_mb:.1f} MB. Max: {MAX_FILE_SIZE_MB} MB",
        )

    # Extract text content based on file type
    try:
        if file.content_type == "application/pdf":
            text_content = _extract_pdf_text(content_bytes)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text_content = _extract_docx_text(content_bytes)
        else:
            text_content = content_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract text from file: {exc}",
        )

    # Create document record
    doc = Document(
        owner_id=current_user.id,
        filename=file.filename or "unnamed",
        content_type=file.content_type,
        size_bytes=len(content_bytes),
        weaviate_class="LLMPlatformDocumentChunk",
        status="pending",
    )
    db.add(doc)
    await db.flush()

    # Submit Celery background task
    try:
        from app.workers.celery_app import ingest_document_task
        celery_task = ingest_document_task.delay(
            document_id=doc.id,
            owner_id=current_user.id,
            filename=doc.filename,
            text_content=text_content,
            metadata={"content_type": file.content_type, "size_bytes": len(content_bytes)},
        )
        doc.metadata_ = {
            **doc.metadata_,
            "celery_task_id": celery_task.id,
        }
    except Exception as exc:
        logger.error("document.celery.submit_failed", error=str(exc))
        doc.status = "failed"
        doc.error_message = "Failed to submit ingestion job"

    await db.commit()
    logger.info(
        "document.uploaded",
        doc_id=doc.id,
        filename=doc.filename,
        size_mb=round(size_mb, 2),
        user_id=current_user.id,
    )

    return APIResponse(
        data=DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            content_type=doc.content_type,
            size_bytes=doc.size_bytes,
            chunk_count=doc.chunk_count,
            status=doc.status,
            created_at=doc.created_at,
        ),
        message="Document uploaded. Ingestion is processing in the background.",
    )


def _extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n\n".join(page.get_text() for page in doc)
    except ImportError:
        # Fallback if fitz not installed
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n\n".join(
                page.extract_text() or "" for page in pdf.pages
            )


def _extract_docx_text(data: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    import docx
    doc = docx.Document(io.BytesIO(data))
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=APIResponse[PaginatedResponse],
    summary="List uploaded documents",
)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
    current_user: CurrentUser = ...,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    offset = (page - 1) * page_size
    q = (
        select(Document)
        .where(Document.owner_id == current_user.id)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    if status_filter:
        q = q.where(Document.status == status_filter)

    total_q = (
        select(func.count())
        .select_from(Document)
        .where(Document.owner_id == current_user.id)
    )

    docs = (await db.execute(q)).scalars().all()
    total = (await db.execute(total_q)).scalar_one()

    return APIResponse(
        data=PaginatedResponse(
            items=[
                DocumentResponse(
                    id=d.id, filename=d.filename, content_type=d.content_type,
                    size_bytes=d.size_bytes, chunk_count=d.chunk_count,
                    status=d.status, created_at=d.created_at,
                )
                for d in docs
            ],
            total=total, page=page, page_size=page_size,
            pages=(total + page_size - 1) // page_size,
        )
    )


# ---------------------------------------------------------------------------
# GET /documents/{id}
# ---------------------------------------------------------------------------

@router.get(
    "/{document_id}",
    response_model=APIResponse[DocumentResponse],
    summary="Get document status and metadata",
)
async def get_document(
    document_id: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse[DocumentResponse]:
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id)
        .where(Document.owner_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return APIResponse(
        data=DocumentResponse(
            id=doc.id, filename=doc.filename, content_type=doc.content_type,
            size_bytes=doc.size_bytes, chunk_count=doc.chunk_count,
            status=doc.status, created_at=doc.created_at,
        )
    )


# ---------------------------------------------------------------------------
# DELETE /documents/{id}
# ---------------------------------------------------------------------------

@router.delete(
    "/{document_id}",
    response_model=APIResponse,
    summary="Delete a document from DB and Weaviate",
)
async def delete_document(
    document_id: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id)
        .where(Document.owner_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete chunks from Weaviate first
    deleted_chunks = await delete_document_chunks(document_id)

    # Delete DB record (cascades to document_chunks)
    await db.delete(doc)
    await db.commit()

    logger.info(
        "document.deleted",
        doc_id=document_id,
        chunks_deleted=deleted_chunks,
        user_id=current_user.id,
    )
    return APIResponse(
        message=f"Document deleted. {deleted_chunks} vector chunks removed."
    )


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------

@router.post(
    "/search",
    response_model=APIResponse[DocumentSearchResponse],
    summary="Hybrid search across user's documents",
)
async def search_documents(
    body: DocumentSearchRequest,
    current_user: CurrentUser,
) -> APIResponse[DocumentSearchResponse]:
    """
    Perform hybrid (BM25 + vector) search across all documents owned by the current user.

    alpha parameter:
      0.0 = BM25 only (exact keyword matching)
      1.0 = vector only (semantic matching)
      0.5 = balanced blend (recommended default)
    """
    import time
    start = time.perf_counter()

    results = await hybrid_search(
        query=body.query,
        owner_id=current_user.id,
        top_k=body.top_k,
        alpha=body.alpha,
        document_ids=body.document_ids,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return APIResponse(
        data=DocumentSearchResponse(
            results=[
                SourceChunk(
                    document_id=r.document_id,
                    chunk_id=r.weaviate_id,
                    filename=r.filename,
                    content=r.content,
                    score=r.score,
                )
                for r in results
            ],
            total_found=len(results),
            latency_ms=round(latency_ms, 2),
        )
    )