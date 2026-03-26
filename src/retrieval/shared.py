"""Thread-safe shared singletons for DB clients and embedding model.

Every tool module should import from here instead of creating its own
connection. This avoids duplicate model loads and connection pools.
"""

import threading

from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()

_encoder: SentenceTransformer | None = None
_qdrant = None  # QdrantManager, lazy to avoid import-time connection
_neo4j = None   # Neo4jManager, lazy to avoid import-time connection


def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                _encoder = SentenceTransformer(settings.embedding_model)
                logger.info("Loaded embedding model: %s", settings.embedding_model)
    return _encoder


def get_qdrant():
    """Return the shared QdrantManager instance."""
    global _qdrant
    if _qdrant is None:
        with _lock:
            if _qdrant is None:
                from src.retrieval.qdrant_client import QdrantManager
                _qdrant = QdrantManager()
    return _qdrant


def get_neo4j():
    """Return the shared Neo4jManager instance."""
    global _neo4j
    if _neo4j is None:
        with _lock:
            if _neo4j is None:
                from src.retrieval.neo4j_client import Neo4jManager
                _neo4j = Neo4jManager()
    return _neo4j
