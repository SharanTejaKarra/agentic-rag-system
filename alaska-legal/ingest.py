"""
ingest.py

Orchestrates the full ingest pipeline: parse -> chunk -> embed.
Run once to populate ChromaDB, or re-run with --force to rebuild.

Usage:
    python ingest.py
    python ingest.py --data-dir ./data --force
"""

import argparse
import logging
from pathlib import Path

from parser import parse_pdf
from chunker import build_chunk, SectionChunk
from embedder import embed_and_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest(data_dir: Path, force: bool) -> None:
    """
    Run the full ingest pipeline for all PDFs in data_dir.

    Args:
        data_dir: Directory containing AAC PDF files.
        force:    If True, wipe and rebuild the ChromaDB collection.

    Returns:
        None. Side effect: populated ChromaDB collection.
    """
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", data_dir)
        return

    all_chunks: list[SectionChunk] = []
    total_repealed = 0
    total_appendix = 0

    for pdf_path in pdf_files:
        raw_sections = parse_pdf(pdf_path)
        chunks = [build_chunk(s, pdf_path.name) for s in raw_sections]
        all_chunks.extend(chunks)
        total_repealed += sum(1 for c in chunks if c.status == "repealed")
        total_appendix += sum(1 for c in chunks if c.has_appendix)
        logger.info("'%s': %d sections chunked.", pdf_path.name, len(chunks))

    embed_and_store(all_chunks, force_reingest=force)

    logger.info(
        "Ingest complete — PDFs: %d | Sections: %d | Repealed: %d | With appendix: %d",
        len(pdf_files), len(all_chunks), total_repealed, total_appendix,
    )


def main() -> None:
    """Parse CLI arguments and run ingest."""
    parser = argparse.ArgumentParser(description="Ingest AAC PDFs into ChromaDB.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"),
                        help="Directory containing PDF files (default: ./data)")
    parser.add_argument("--force", action="store_true",
                        help="Delete and rebuild the ChromaDB collection from scratch")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        logger.error("Data directory '%s' does not exist.", args.data_dir)
        raise SystemExit(1)

    ingest(args.data_dir, args.force)


if __name__ == "__main__":
    main()
