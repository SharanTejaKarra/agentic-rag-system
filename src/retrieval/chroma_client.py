"""ChromaDB vector database client for semantic search."""

import logging
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


class ChromaManager:
    """Manages connections and operations against a ChromaDB instance."""

    def __init__(self) -> None:
        import chromadb

        try:
            self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
            logger.info("Opened ChromaDB at %s", settings.chroma_persist_dir)
        except Exception:
            logger.exception("Failed to open ChromaDB")
            raise

    def get_or_create_collection(self, name: str):
        """Get or create a collection by name."""
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- search -----------------------------------------------------------------

    def search(
        self,
        collection: str,
        query_vector: list[float],
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run similarity search with optional metadata filters.

        Returns a list of dicts with keys: id, score, payload.
        """
        limit = limit or settings.retrieval_top_k
        coll = self.get_or_create_collection(collection)

        where = self._build_where(filters) if filters else None

        try:
            results = coll.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where,
            )
        except Exception:
            logger.exception("ChromaDB search failed on collection '%s'", collection)
            raise

        hits: list[dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [None] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

        for i, doc_id in enumerate(ids):
            # ChromaDB returns distances (lower = more similar for cosine).
            # Convert to a similarity score in [0, 1].
            distance = distances[i]
            score = 1.0 - distance

            if score < settings.similarity_threshold:
                continue

            payload = dict(metadatas[i]) if metadatas[i] else {}
            if documents[i] is not None:
                payload["content"] = documents[i]

            hits.append({"id": doc_id, "score": score, "payload": payload})

        return hits

    # -- upsert -----------------------------------------------------------------

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add or update documents in a collection."""
        coll = self.get_or_create_collection(collection_name)
        try:
            coll.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Upserted %d items into '%s'", len(ids), collection_name)
        except Exception:
            logger.exception("ChromaDB upsert failed on collection '%s'", collection_name)
            raise

    # -- internal helpers -------------------------------------------------------

    @staticmethod
    def _build_where(filters: dict[str, Any]) -> dict | None:
        """Convert a simple filter dict into a ChromaDB where clause.

        Supports exact match filters. Range filters are skipped since
        ChromaDB where clauses work differently from Qdrant range filters.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                # Skip range filters -- ChromaDB does not support them the same way
                continue
            conditions.append({key: {"$eq": value}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
