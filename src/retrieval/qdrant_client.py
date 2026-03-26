"""Qdrant vector database client for semantic search."""

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from config.settings import settings

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manages connections and operations against a Qdrant instance."""

    def __init__(self) -> None:
        try:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
            logger.info("Connected to Qdrant at %s", settings.qdrant_url)
        except Exception:
            logger.exception("Failed to connect to Qdrant")
            raise

    # -- collection management --------------------------------------------------

    def create_collection(self, name: str, vector_size: int) -> None:
        """Create a collection with cosine distance if it doesn't exist."""
        try:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection '%s' (dim=%d)", name, vector_size)
        except Exception:
            logger.exception("Failed to create collection '%s'", name)
            raise

    # -- search -----------------------------------------------------------------

    def search(
        self,
        collection: str,
        query_vector: list[float],
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run similarity search with optional payload filters.

        Returns a list of dicts with keys: id, score, payload.
        """
        limit = limit or settings.retrieval_top_k
        qdrant_filter = self._build_filter(filters) if filters else None

        try:
            hits = self._client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=settings.similarity_threshold,
            )
        except Exception:
            logger.exception("Qdrant search failed on collection '%s'", collection)
            raise

        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in hits
        ]

    # -- upsert -----------------------------------------------------------------

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Batch upsert points. Each dict needs 'id', 'vector', 'payload'."""
        structs = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]
        try:
            self._client.upsert(collection_name=collection, points=structs)
            logger.info("Upserted %d points into '%s'", len(structs), collection)
        except Exception:
            logger.exception("Qdrant upsert failed on collection '%s'", collection)
            raise

    # -- internal helpers -------------------------------------------------------

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """Convert a simple filter dict into a Qdrant Filter object.

        Supported filter shapes:
          {"section": "4.2"}           - exact match
          {"date_range": {"gte": ..., "lte": ...}} - range filter
        """
        conditions: list[FieldCondition] = []
        for key, value in filters.items():
            if isinstance(value, dict) and ("gte" in value or "lte" in value):
                conditions.append(
                    FieldCondition(key=key, range=Range(**value))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=conditions)
