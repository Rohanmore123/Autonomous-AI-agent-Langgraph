"""
app/services/vector_db/weaviate_client.py
=========================================
Weaviate v4 async client for:
  • Schema management (auto-create collections on startup)
  • Document ingestion with chunking + embedding
  • Hybrid search (BM25 keyword + vector similarity, weighted by alpha)
  • Pure vector search (cosine similarity)
  • Pure keyword search (BM25)
  • Object deletion and updates

HYBRID SEARCH EXPLAINED:
  alpha=0.0  → 100% BM25 (keyword exact/fuzzy match)
  alpha=1.0  → 100% vector (semantic similarity)
  alpha=0.5  → equal blend (usually best for general Q&A)

  The two scores are normalised and combined:
    final_score = alpha * vector_score + (1-alpha) * bm25_score

  This outperforms either alone because:
    • BM25 catches exact terminology (product codes, names)
    • Vector catches paraphrases and related concepts

EMBEDDING:
  We use sentence-transformers locally (no API cost, no latency from external call).
  Model: all-MiniLM-L6-v2  (384-dim, 80ms/sentence on CPU, very good quality).
  In high-throughput scenarios, run the embedding model on GPU or replace with
  a hosted embedding API.

CHUNKING STRATEGY:
  Fixed-size chunks of 512 tokens with 64-token overlap.
  Overlap ensures a sentence split at a chunk boundary appears fully in at least one chunk.
"""

from __future__ import annotations

import time
import uuid
import json
from typing import Any
from urllib.parse import urlparse

import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Embedding model (loaded once at module import)
# ---------------------------------------------------------------------------

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("embedding.model.loading", model=settings.llm.embedding_model)
        _embedding_model = SentenceTransformer(settings.llm.embedding_model)
        logger.info("embedding.model.ready")
    return _embedding_model


def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a list of floats."""
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embedding — much faster than calling embed_text in a loop."""
    model = get_embedding_model()
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Weaviate connection
# ---------------------------------------------------------------------------

_client: weaviate.WeaviateClient | None = None

DOCUMENT_COLLECTION = f"{settings.weaviate.class_prefix}DocumentChunk"


def get_weaviate_client() -> weaviate.WeaviateClient:
    global _client
    if _client is None or not _client.is_connected():
        auth = (
            weaviate.auth.AuthApiKey(settings.weaviate.api_key)
            if settings.weaviate.api_key
            else None
        )
        url = settings.weaviate.url.strip()
        if "://" not in url:
            url = f"https://{url}"
        parsed = urlparse(url)

        if parsed.hostname and parsed.hostname.endswith(".weaviate.cloud"):
            _client = weaviate.connect_to_wcs(
                cluster_url=parsed.hostname,
                auth_credentials=auth,
            )
        else:
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 8080)
            _client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=parsed.scheme == "https",
                grpc_host=host,
                grpc_port=443 if parsed.scheme == "https" else 50051,
                grpc_secure=parsed.scheme == "https",
                auth_credentials=auth,
            )
    return _client


async def init_weaviate() -> None:
    """
    Ensure the DocumentChunk collection exists with correct schema.
    Called from app lifespan.
    """
    client = get_weaviate_client()
    collection_name = DOCUMENT_COLLECTION

    if not client.collections.exists(collection_name):
        client.collections.create(
            name=collection_name,
            description="RAG document chunks with dense + sparse vectors",
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # We supply our own vectors
            vector_index_config=wvc.config.Configure.VectorIndex.hfresh(),
            properties=[
                wvc.config.Property(name="document_id",   data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_id",      data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="owner_id",      data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="content",       data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="filename",      data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_index",   data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="token_count",   data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="metadata",      data_type=wvc.config.DataType.TEXT),
            ],
            # BM25 inverted index for keyword half of hybrid search
            inverted_index_config=wvc.config.Configure.inverted_index(
                bm25_b=0.75,
                bm25_k1=1.2,
            ),
        )
        logger.info("weaviate.collection.created", name=collection_name)
    else:
        logger.info("weaviate.collection.exists", name=collection_name)


async def close_weaviate() -> None:
    global _client
    if _client:
        _client.close()
        _client = None
    logger.info("weaviate.closed")


async def check_weaviate_health() -> bool:
    try:
        client = get_weaviate_client()
        return client.is_live()
    except Exception as exc:
        logger.error("weaviate.health.failed", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """
    Naive word-level sliding window chunker.
    In production, prefer token-aware chunking via tiktoken.
    """
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

async def ingest_document(
    document_id: str,
    owner_id: str,
    filename: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """
    Chunk, embed, and store a document in Weaviate.
    Returns list of Weaviate UUIDs (one per chunk).

    Steps:
      1. Split content into chunks
      2. Batch-embed all chunks (single model.encode() call)
      3. Batch-insert into Weaviate
    """
    client = get_weaviate_client()
    collection = client.collections.get(DOCUMENT_COLLECTION)

    chunks = chunk_text(content)
    logger.info("weaviate.ingest.start", doc_id=document_id, chunks=len(chunks))

    # Batch embed (fast)
    vectors = embed_batch(chunks)

    weaviate_ids: list[str] = []
    objects = []

    for idx, (chunk_text_val, vector) in enumerate(zip(chunks, vectors)):
        weaviate_id = str(uuid.uuid4())
        weaviate_ids.append(weaviate_id)
        objects.append(
            wvc.data.DataObject(
                uuid=weaviate_id,
                properties={
                    "document_id": document_id,
                    "chunk_id":    weaviate_id,
                    "owner_id":    owner_id,
                    "content":     chunk_text_val,
                    "filename":    filename,
                    "chunk_index": idx,
                    "token_count": len(chunk_text_val.split()),
                    "metadata":    json.dumps(metadata or {}),
                },
                vector=vector,
            )
        )

    # Batch insert (single network round-trip)
    response = collection.data.insert_many(objects)
    if response.has_errors:
        for err in response.errors.values():
            logger.error("weaviate.ingest.error", error=str(err))

    logger.info("weaviate.ingest.done", doc_id=document_id, stored=len(weaviate_ids))
    return weaviate_ids


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchResult:
    """One search hit."""
    __slots__ = ("weaviate_id", "document_id", "filename", "content", "score", "chunk_index")

    def __init__(
        self,
        weaviate_id: str,
        document_id: str,
        filename: str,
        content: str,
        score: float,
        chunk_index: int,
    ) -> None:
        self.weaviate_id = weaviate_id
        self.document_id = document_id
        self.filename = filename
        self.content = content
        self.score = score
        self.chunk_index = chunk_index


async def hybrid_search(
    query: str,
    owner_id: str,
    top_k: int = 5,
    alpha: float = 0.5,
    document_ids: list[str] | None = None,
) -> list[SearchResult]:
    """
    Weaviate hybrid search:
      • Converts query to vector (dense)
      • Runs BM25 on content field (sparse)
      • Weaviate merges scores with alpha weighting

    owner_id filter ensures strict data isolation — user A never sees user B's docs.
    document_ids allows scoping to a specific subset of documents.
    """
    start = time.perf_counter()
    client = get_weaviate_client()
    collection = client.collections.get(DOCUMENT_COLLECTION)

    query_vector = embed_text(query)

    # Build filter
    filters = wvc.query.Filter.by_property("owner_id").equal(owner_id)
    if document_ids:
        doc_filter = wvc.query.Filter.by_property("document_id").contains_any(document_ids)
        filters = filters & doc_filter

    response = collection.query.hybrid(
        query=query,            # BM25 uses this
        vector=query_vector,    # Dense search uses this
        alpha=alpha,
        limit=top_k,
        filters=filters,
        return_metadata=wvc.query.MetadataQuery(score=True),
    )

    results = [
        SearchResult(
            weaviate_id=str(obj.uuid),
            document_id=obj.properties["document_id"],
            filename=obj.properties["filename"],
            content=obj.properties["content"],
            score=obj.metadata.score or 0.0,
            chunk_index=obj.properties.get("chunk_index", 0),
        )
        for obj in response.objects
    ]

    latency = (time.perf_counter() - start) * 1000
    logger.info(
        "weaviate.hybrid_search",
        query=query[:50],
        top_k=top_k,
        alpha=alpha,
        results=len(results),
        latency_ms=round(latency, 2),
    )
    return results


async def delete_document_chunks(document_id: str) -> int:
    """Remove all chunks belonging to a document. Returns count deleted."""
    client = get_weaviate_client()
    collection = client.collections.get(DOCUMENT_COLLECTION)

    result = collection.data.delete_many(
        where=wvc.query.Filter.by_property("document_id").equal(document_id)
    )
    count = result.successful if hasattr(result, "successful") else 0
    logger.info("weaviate.chunks.deleted", document_id=document_id, count=count)
    return count