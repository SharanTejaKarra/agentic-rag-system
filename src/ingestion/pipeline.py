"""Ingestion pipeline orchestrator."""

from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.chunker import hierarchical_chunk
from src.ingestion.embedder import embed_chunks
from src.ingestion.qdrant_loader import load_to_qdrant
from src.ingestion.graph_builder import build_knowledge_graph
from src.utils.logging import get_logger, new_correlation_id

logger = get_logger(__name__)


def run_ingestion(
    input_dir: str,
    collection_name: str = "legal_docs",
) -> dict:
    """Orchestrate full ingestion: parse -> chunk -> embed -> load to Qdrant + build graph.

    Returns summary stats dict.
    """
    cid = new_correlation_id()
    logger.info("Starting ingestion pipeline (correlation_id=%s, dir=%s)", cid, input_dir)

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Step 1: Parse all documents
    all_docs: list[dict] = []
    supported = {".pdf", ".html", ".htm", ".txt", ".text", ".md"}
    files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in supported]

    logger.info("Found %d files to ingest", len(files))
    for file_path in files:
        try:
            docs = parse_document(str(file_path))
            all_docs.extend(docs)
        except Exception:
            logger.exception("Failed to parse %s, skipping", file_path.name)

    if not all_docs:
        logger.warning("No documents parsed, aborting pipeline")
        return {"files_found": len(files), "documents_parsed": 0, "chunks": 0, "loaded": 0, "graph": {}}

    logger.info("Parsed %d document segments from %d files", len(all_docs), len(files))

    # Step 2: Chunk documents
    chunks = hierarchical_chunk(all_docs)
    logger.info("Created %d chunks", len(chunks))

    # Step 3: Generate embeddings
    chunk_embeddings = embed_chunks(chunks)
    logger.info("Generated %d embeddings", len(chunk_embeddings))

    # Step 4: Load to Qdrant
    loaded_count = load_to_qdrant(chunk_embeddings, collection_name)
    logger.info("Loaded %d chunks to Qdrant collection '%s'", loaded_count, collection_name)

    # Step 5: Build knowledge graph
    graph_stats = build_knowledge_graph(chunks)
    logger.info("Knowledge graph: %s", graph_stats)

    summary = {
        "files_found": len(files),
        "documents_parsed": len(all_docs),
        "chunks": len(chunks),
        "loaded": loaded_count,
        "graph": graph_stats,
    }
    logger.info("Ingestion pipeline complete: %s", summary)
    return summary
