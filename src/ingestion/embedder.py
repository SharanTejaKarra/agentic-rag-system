"""Embedding generation using LlamaIndex HuggingFaceEmbedding."""

from src.schema import Chunk
from config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_embed_model: object = None


def _get_embed_model() -> object:
    """Return the cached HuggingFace embedding model, loading on first call."""
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
        logger.info("Loaded embedding model: %s", settings.embedding_model)
    return _embed_model


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> list[tuple[Chunk, list[float]]]:
    """Generate embedding vectors for a list of chunks.

    Returns (chunk, embedding_vector) pairs.
    """
    if not chunks:
        return []

    embed_model = _get_embed_model()
    texts = [chunk.content for chunk in chunks]

    logger.info("Embedding %d chunks with %s", len(texts), model_name)

    # Batch embed
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

    results = list(zip(chunks, embeddings))
    logger.info("Generated %d embeddings (dim=%d)", len(results), len(embeddings[0]) if embeddings else 0)
    return results
