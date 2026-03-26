"""Load chunk embeddings into ChromaDB vector database."""

from src.schema.models import Chunk
from config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_manager = None


def _get_manager():
    global _manager
    if _manager is None:
        from src.retrieval.chroma_client import ChromaManager
        _manager = ChromaManager()
    return _manager


def load_to_chroma(
    chunk_embeddings: list[tuple[Chunk, list[float]]],
    collection_name: str,
) -> int:
    """Upsert chunks with embeddings into a ChromaDB collection.

    Returns count of loaded chunks.
    """
    if not chunk_embeddings:
        return 0

    manager = _get_manager()

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for chunk, embedding in chunk_embeddings:
        ids.append(chunk.id)
        embeddings.append(embedding)
        documents.append(chunk.content)
        meta = {
            "chunk_id": chunk.id,
            "section_ref": chunk.section_ref,
        }
        # ChromaDB metadata values must be str, int, float, or bool
        for k, v in chunk.metadata.items():
            meta[k] = str(v)
        metadatas.append(meta)

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        manager.upsert(
            collection_name=collection_name,
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info("Upserted batch %d-%d to '%s'", i, end, collection_name)

    logger.info("Loaded %d chunks into ChromaDB collection '%s'", len(ids), collection_name)
    return len(ids)
