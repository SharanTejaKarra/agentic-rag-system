"""Vector search tool - semantic similarity search over the document corpus."""

import logging
from typing import Any

from config.settings import settings
from src.schema.models import Chunk

logger = logging.getLogger(__name__)


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
    try:
        from src.retrieval.shared import get_encoder, get_chroma
        encoder = get_encoder()
        chroma = get_chroma()
    except Exception:
        logger.exception("Failed to connect to ChromaDB or load encoder")
        return []

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

    try:
        hits = chroma.search(
            collection=settings.chroma_collection_name,
            query_vector=query_vector,
            filters=payload_filters,
            limit=settings.retrieval_top_k,
        )
    except Exception:
        logger.exception("ChromaDB search failed")
        return []

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
