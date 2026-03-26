"""Vector search tool - semantic similarity search over the document corpus."""

import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.retrieval.qdrant_client import QdrantManager
from src.schema.models import Chunk

logger = logging.getLogger(__name__)

_encoder: SentenceTransformer | None = None
_qdrant: QdrantManager | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(settings.embedding_model)
    return _encoder


def _get_qdrant() -> QdrantManager:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantManager()
    return _qdrant


def vector_search(
    query: str, filters: dict[str, Any] | None = None
) -> list[Chunk]:
    """Run semantic search against the vector store.

    Args:
        query: Natural-language search query.
        filters: Optional filters for section, date_range, entity_type, etc.

    Returns:
        Scored Chunk objects ordered by relevance.
    """
    encoder = _get_encoder()
    qdrant = _get_qdrant()

    query_vector = encoder.encode(query).tolist()

    # Translate high-level filter keys into payload filters
    payload_filters: dict[str, Any] | None = None
    if filters:
        payload_filters = {}
        if "section" in filters:
            payload_filters["section_ref"] = filters["section"]
        if "date_range" in filters:
            payload_filters["date"] = filters["date_range"]
        if "entity_type" in filters:
            payload_filters["entity_type"] = filters["entity_type"]
        if not payload_filters:
            payload_filters = None

    hits = qdrant.search(
        collection=settings.qdrant_collection_name,
        query_vector=query_vector,
        filters=payload_filters,
        limit=settings.retrieval_top_k,
    )

    chunks = []
    for hit in hits:
        payload = hit.get("payload", {})
        chunks.append(
            Chunk(
                id=str(hit["id"]),
                content=payload.get("content", ""),
                section_ref=payload.get("section_ref", ""),
                metadata={k: str(v) for k, v in payload.items()
                          if k not in ("content", "section_ref")},
                score=hit.get("score", 0.0),
            )
        )
    return chunks
