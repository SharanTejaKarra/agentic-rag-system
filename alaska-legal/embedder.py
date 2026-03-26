"""
embedder.py

Loads the BGE embedding model, generates embeddings for a list of SectionChunk
objects, and stores them in a persistent ChromaDB collection.

No PDF parsing, no SectionChunk construction, no LLM calls.
"""

import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import SectionChunk

logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "aac_sections"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32


def _load_model() -> SentenceTransformer:
    """Load and return the BGE embedding model."""
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


def _get_collection(client: chromadb.PersistentClient, force: bool) -> chromadb.Collection:
    """
    Return the ChromaDB collection, optionally deleting it first.

    Args:
        client: A chromadb.PersistentClient instance.
        force:  If True, delete the existing collection before creating a new one.

    Returns:
        A ChromaDB Collection object.
    """
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'.", COLLECTION_NAME)
        except Exception:
            pass  # Collection didn't exist — that's fine
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_to_metadata(chunk: SectionChunk) -> dict:
    """
    Build the ChromaDB metadata dict from a SectionChunk.
    Excludes text, raw_lines, and subsections.
    Converts has_appendix bool to int (ChromaDB doesn't support booleans).
    """
    return {
        "section_id":     chunk.section_id,
        "chapter":        chunk.chapter,
        "title":          chunk.title,
        "hierarchy_path": chunk.hierarchy_path,
        "status":         chunk.status,
        "has_appendix":   int(chunk.has_appendix),
        "source_pdf":     chunk.source_pdf,
    }


def _doc_id(chunk: SectionChunk) -> str:
    """Return the ChromaDB document ID for a chunk."""
    return f"chapter_{chunk.chapter}_{chunk.section_id}"


def embed_and_store(chunks: list[SectionChunk], force_reingest: bool = False) -> None:
    """
    Embed all chunks and upsert them into ChromaDB.

    BGE note: this model requires a query-time instruction prefix
    ("Represent this sentence for searching relevant passages: ") for
    semantic search, but NOT at indexing time. We embed raw text here.
    Phase 2 must apply the prefix in retriever.py when enabling vector search.

    Args:
        chunks:         List of SectionChunk objects to embed and store.
        force_reingest: If True, wipe the existing collection first.

    Returns:
        None. Side effect: populated ChromaDB collection at CHROMA_PATH.
    """
    model = _load_model()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = _get_collection(client, force_reingest)

    total = len(chunks)
    stored = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]

        ids       = [_doc_id(c) for c in batch]
        documents = [c.text for c in batch]
        metadatas = [_chunk_to_metadata(c) for c in batch]

        # Embed raw text — no prefix at ingest time for BGE models
        embeddings = model.encode(documents, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        stored += len(batch)
        logger.info(
            "Batch %d–%d ingested (%d/%d total).",
            batch_start + 1, batch_start + len(batch), stored, total,
        )

    logger.info("Embedding complete. %d documents stored in '%s'.", stored, COLLECTION_NAME)
