"""CLI script to run the document ingestion pipeline."""

import argparse
import sys

from src.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest legal documents into the RAG system."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing documents to ingest (PDF, HTML, TXT, MD).",
    )
    parser.add_argument(
        "--collection",
        default="legal_docs",
        help="ChromaDB collection name (default: legal_docs).",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip Neo4j graph building (faster, vector search still works).",
    )
    args = parser.parse_args()

    print(f"Starting ingestion from: {args.input_dir}")
    print(f"Target collection: {args.collection}")
    if args.skip_graph:
        print("Skipping graph building")

    try:
        stats = run_ingestion(args.input_dir, args.collection, skip_graph=args.skip_graph)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nIngestion complete. Summary:")
    print(f"  Files found:       {stats['files_found']}")
    print(f"  Documents parsed:  {stats['documents_parsed']}")
    print(f"  Chunks created:    {stats['chunks']}")
    print(f"  Loaded to ChromaDB: {stats['loaded']}")
    print(f"  Graph stats:       {stats['graph']}")


if __name__ == "__main__":
    main()
