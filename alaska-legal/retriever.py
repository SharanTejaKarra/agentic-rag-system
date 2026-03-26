"""
retriever.py

Accepts a raw user query, extracts and normalises the section identifier,
queries ChromaDB by metadata filter, and returns a SectionChunk or an error dict.

No PDF reading, no embedding model calls, no LLM calls.
"""

import re
import logging

import chromadb

from chunker import SectionChunk

logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "aac_sections"

# Tolerant section ID pattern:
# - optional spaces between "3", "AAC", and the number
# - case-insensitive via re.IGNORECASE on the literal "AAC"
# [Aa]{3} was wrong — C is not in [Aa] so it could never match "AAC"
# Decimal length validation is handled in _extract_and_normalise.
SECTION_ID_RE = re.compile(r"3\s*AAC\s*(\d+)\.(\d+)", re.IGNORECASE)


def _extract_and_normalise(query: str) -> str | None:
    """
    Extract a section identifier from a free-text query and normalise it.

    Only accepts decimals that are exactly 3 digits, or exactly 2 digits
    where right-padding to 3 is unambiguous (e.g. "08" -> "080").
    All sections in the Alaska AAC PDFs in scope use the format XX.0XX or
    XX.XXX — the decimal component is right-padded, never left-padded.
    Decimals of 1 digit or 4+ digits are rejected as ambiguous or invalid.

    Args:
        query: Raw user query string.

    Returns:
        Normalised section ID string (e.g. "26.080"), or None if not found
        or if the decimal length is ambiguous/invalid.
    """
    m = SECTION_ID_RE.search(query)
    if not m:
        return None

    chapter = m.group(1)
    decimal_raw = m.group(2)

    if len(decimal_raw) == 3:
        # Already fully specified — accept as-is.
        decimal = decimal_raw
    elif len(decimal_raw) == 2:
        # Right-pad one zero: "08" -> "080". Safe for this document set.
        decimal = decimal_raw.ljust(3, "0")
    else:
        # 1 digit (e.g. "26.8") or 4+ digits — ambiguous or malformed; reject.
        logger.debug("Rejecting ambiguous decimal '%s' in query: %s", decimal_raw, query)
        return None

    return f"{chapter}.{decimal}"


def _get_collection() -> chromadb.Collection:
    """Open the persistent ChromaDB client and return the AAC collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def _reconstruct_chunk(doc: str, metadata: dict) -> SectionChunk:
    """
    Reconstruct a SectionChunk from a ChromaDB document and its metadata.

    raw_lines and subsections are not stored in ChromaDB — set to empty lists.
    has_appendix is stored as int; convert back to bool.

    Args:
        doc:      The document text string from ChromaDB.
        metadata: The metadata dict from ChromaDB.

    Returns:
        A SectionChunk object.
    """
    return SectionChunk(
        section_id=metadata["section_id"],
        chapter=metadata["chapter"],
        title=metadata["title"],
        hierarchy_path=metadata["hierarchy_path"],
        status=metadata["status"],
        has_appendix=bool(metadata["has_appendix"]),
        source_pdf=metadata["source_pdf"],
        text=doc,
        subsections=[],
        raw_lines=[],
    )


def retrieve(query: str) -> SectionChunk | dict:
    """
    Extract a section ID from the query and return the matching SectionChunk.

    Phase 1 uses metadata filtering only — no vector similarity search.
    Phase 2 note: BGE models require the prefix
    "Represent this sentence for searching relevant passages: "
    prepended to the query when enabling vector search. Do not apply it here.

    Args:
        query: Raw user query string.

    Returns:
        A SectionChunk if the section is found, or a dict with an "error" key.
    """
    section_id = _extract_and_normalise(query)

    if section_id is None:
        return {"error": "Please provide a section number in the format 3 AAC XX.XXX"}

    logger.info("Looking up section_id='%s' from query: %s", section_id, query)

    collection = _get_collection()

    # Phase 1: exact metadata filter — no embedding or similarity search
    results = collection.get(where={"section_id": section_id})

    if not results["documents"]:
        return {
            "error": (
                f"Section 3 AAC {section_id} was not found in the database. "
                "Verify the section number and ensure the relevant chapter has been ingested."
            )
        }

    chunk = _reconstruct_chunk(results["documents"][0], results["metadatas"][0])
    logger.info("Retrieved section '%s' (%s).", chunk.section_id, chunk.status)
    return chunk