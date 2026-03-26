"""Embedding generation using LlamaIndex HuggingFaceEmbedding.

Processes chunks in small batches to avoid OOM on large documents.
"""

import gc

from src.schema import Chunk
from config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_embed_model: object = None

# Process this many chunks at a time to keep memory bounded
_BATCH_SIZE = 32


def _get_embed_model() -> object:
    """Return the cached HuggingFace embedding model, loading on first call."""
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
        logger.info("Embedding model loaded")
    return _embed_model


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> list[tuple[Chunk, list[float]]]:
    """Generate embedding vectors for a list of chunks.

    Processes in batches of _BATCH_SIZE to keep memory usage bounded
    on large documents. Returns (chunk, embedding_vector) pairs.
    """
    if not chunks:
        return []

    embed_model = _get_embed_model()
    results: list[tuple[Chunk, list[float]]] = []

    total = len(chunks)
    logger.info("Embedding %d chunks in batches of %d", total, _BATCH_SIZE)

    for start in range(0, total, _BATCH_SIZE):
        end = min(start + _BATCH_SIZE, total)
        batch = chunks[start:end]
        texts = [c.content for c in batch]

        embeddings = embed_model.get_text_embedding_batch(texts)
        results.extend(zip(batch, embeddings))

        logger.info("Embedded batch %d-%d / %d", start + 1, end, total)

    # Free any temp memory from batch processing
    gc.collect()

    dim = len(results[0][1]) if results else 0
    logger.info("Generated %d embeddings (dim=%d)", len(results), dim)
    return results
