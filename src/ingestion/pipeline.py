"""Ingestion pipeline orchestrator.

Processes documents in stages with progress logging and garbage
collection between stages to keep memory usage bounded.

Heavy imports are deferred to the functions that need them so the
full dependency tree is not loaded at startup. This keeps peak
memory low enough for 8 GB MacBooks.
"""

import gc
from pathlib import Path

from src.utils.logging import get_logger, new_correlation_id

logger = get_logger(__name__)

# Process graph building in batches to avoid too many LLM calls
_GRAPH_BATCH_SIZE = 50


def run_ingestion(
    input_dir: str,
    collection_name: str = "legal_docs",
    skip_graph: bool = False,
) -> dict:
    """Orchestrate full ingestion: parse -> chunk -> embed -> load to ChromaDB + build graph.

    Args:
        input_dir: Path to directory containing documents.
        collection_name: ChromaDB collection to load into.
        skip_graph: If True, skip the Neo4j graph building step. Useful when
            Neo4j isn't running or you want faster ingestion.

    Returns summary stats dict.
    """
    cid = new_correlation_id()
    logger.info("Starting ingestion pipeline (correlation_id=%s, dir=%s)", cid, input_dir)

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Step 1: Parse all documents
    from src.ingestion.parser import parse_document, ALL_SUPPORTED_EXTENSIONS

    all_docs: list[dict] = []
    supported = ALL_SUPPORTED_EXTENSIONS
    files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported
    )

    logger.info("Found %d files to ingest", len(files))
    for idx, file_path in enumerate(files, 1):
        logger.info("Parsing file %d/%d: %s", idx, len(files), file_path.name)
        try:
            docs = parse_document(str(file_path))
            all_docs.extend(docs)
            logger.info("  -> %d segments extracted", len(docs))
        except Exception:
            logger.exception("Failed to parse %s, skipping", file_path.name)

    if not all_docs:
        logger.warning("No documents parsed, aborting pipeline")
        return {
            "files_found": len(files), "documents_parsed": 0,
            "chunks": 0, "loaded": 0, "graph": {},
        }

    logger.info("Parsed %d document segments from %d files", len(all_docs), len(files))

    # Step 2: Chunk documents
    logger.info("Step 2: Chunking documents...")
    from src.ingestion.chunker import hierarchical_chunk
    chunks = hierarchical_chunk(all_docs)
    logger.info("Created %d chunks", len(chunks))

    # Free parsed docs, we only need chunks from here
    del all_docs
    gc.collect()

    # Step 3: Generate embeddings (batched internally)
    logger.info("Step 3: Generating embeddings...")
    from src.ingestion.embedder import embed_chunks
    chunk_embeddings = embed_chunks(chunks)
    logger.info("Generated %d embeddings", len(chunk_embeddings))

    # Step 4: Load to ChromaDB (batched internally by chroma_loader)
    logger.info("Step 4: Loading to ChromaDB...")
    from src.ingestion.chroma_loader import load_to_chroma
    loaded_count = load_to_chroma(chunk_embeddings, collection_name)
    logger.info("Loaded %d chunks to ChromaDB collection '%s'", loaded_count, collection_name)

    # Free embeddings, we only need chunks for graph building
    del chunk_embeddings
    gc.collect()

    # Step 5: Build knowledge graph (optional)
    graph_stats = {"nodes_created": 0, "edges_created": 0}
    if skip_graph:
        logger.info("Step 5: Skipping graph building (skip_graph=True)")
    else:
        logger.info("Step 5: Building knowledge graph...")
        try:
            from src.ingestion.graph_builder import build_knowledge_graph
            graph_stats = build_knowledge_graph(chunks)
            logger.info("Knowledge graph: %s", graph_stats)
        except Exception:
            logger.exception(
                "Graph building failed (is Neo4j running?). "
                "Vector search will still work. Use --skip-graph to skip this step."
            )

    summary = {
        "files_found": len(files),
        "documents_parsed": len(chunks),
        "chunks": len(chunks),
        "loaded": loaded_count,
        "graph": graph_stats,
    }
    logger.info("Ingestion pipeline complete: %s", summary)
    return summary
