"""
validate_parser.py

Runs the parser on all PDFs in a directory and prints a human-readable
validation report. Use this before ingest to confirm parsing correctness.

No ChromaDB, no embeddings, no LLM.

Usage:
    python validate_parser.py
    python validate_parser.py --data-dir ./data
"""

import argparse
import re
from pathlib import Path

from parser import parse_pdf
from chunker import build_chunk

# Expected section ID format after normalisation: "XX.XXX"
VALID_SECTION_ID_RE = re.compile(r"^\d+\.\d{3}$")


def _is_malformed(section_id: str) -> bool:
    """Return True if the section ID does not match the expected XX.XXX format."""
    return not VALID_SECTION_ID_RE.match(section_id)


def report_pdf(pdf_path: Path) -> None:
    """
    Parse a single PDF and print its validation report to stdout.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        None. Prints to stdout.
    """
    raw_sections = parse_pdf(pdf_path)
    chunks = [build_chunk(s, pdf_path.name) for s in raw_sections]

    repealed   = [c for c in chunks if c.status == "repealed"]
    appendixed = [c for c in chunks if c.has_appendix]
    malformed  = [c for c in chunks if _is_malformed(c.section_id)]

    print(f"\n{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"{'='*60}")
    print(f"  Total sections : {len(chunks)}")
    print(f"  Repealed       : {len(repealed)}")
    print(f"  With appendix  : {len(appendixed)}")

    print(f"\n  Section IDs ({len(chunks)} found):")
    for chunk in chunks:
        print(f"    {chunk.section_id}  [{chunk.status}]  {chunk.title[:60]}")

    if malformed:
        print(f"\n  *** MALFORMED SECTION IDs ({len(malformed)}) ***")
        for chunk in malformed:
            print(f"    {chunk.section_id!r}  header: {chunk.title[:60]!r}")
    else:
        print("\n  Malformed IDs  : none")

    print("\n  Sanity check — first 80 chars of text for first 3 sections:")
    for chunk in chunks[:3]:
        preview = chunk.text[:80].replace("\n", " ")
        print(f"    [{chunk.section_id}] {preview!r}")


def main() -> None:
    """Parse CLI arguments and run the validation report for all PDFs."""
    parser = argparse.ArgumentParser(description="Validate AAC PDF parser output.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"),
                        help="Directory containing PDF files (default: ./data)")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: Directory '{args.data_dir}' does not exist.")
        raise SystemExit(1)

    pdf_files = sorted(args.data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{args.data_dir}'.")
        raise SystemExit(1)

    print(f"Validating {len(pdf_files)} PDF(s) in '{args.data_dir}'...")
    for pdf_path in pdf_files:
        report_pdf(pdf_path)

    print(f"\n{'='*60}")
    print("Validation complete.")


if __name__ == "__main__":
    main()
