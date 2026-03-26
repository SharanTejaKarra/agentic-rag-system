"""Load chunk embeddings into Qdrant vector database."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.schema import Chunk
from config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _client


def load_to_qdrant(
    chunk_embeddings: list[tuple[Chunk, list[float]]],
    collection_name: str,
) -> int:
    """Upsert chunks with embeddings into a Qdrant collection.

    Creates the collection if it doesn't exist. Returns count of loaded chunks.
    """
    if not chunk_embeddings:
        return 0

    client = _get_client()
    vector_dim = len(chunk_embeddings[0][1])

    # Create collection if needed
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d)", collection_name, vector_dim)

    # Build points
    points: list[PointStruct] = []
    for chunk, embedding in chunk_embeddings:
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id).hex
        payload = {
            "chunk_id": chunk.id,
            "content": chunk.content,
            "section_ref": chunk.section_ref,
            **chunk.metadata,
        }
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info("Upserted batch %d-%d to '%s'", i, i + len(batch), collection_name)

    logger.info("Loaded %d chunks into Qdrant collection '%s'", len(points), collection_name)
    return len(points)
