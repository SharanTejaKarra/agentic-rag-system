"""Chunking strategies: hierarchical (rule-based) and propositional (LLM-based)."""

import json
import re
import uuid

from src.schema import Chunk, FactType
from src.ingestion.metadata import extract_hierarchy, extract_references
from src.utils.logging import get_logger

logger = get_logger(__name__)


def hierarchical_chunk(
    documents: list[dict],
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk text while preserving section boundaries.

    Each chunk inherits its section's hierarchy metadata.
    """
    chunks: list[Chunk] = []

    for doc in documents:
        text = doc["text"]
        metadata = doc.get("metadata", {})

        # Build hierarchy for the document
        hierarchy = extract_hierarchy(text)
        refs = extract_references(text)

        # Split on section boundaries first, then by size
        sections = _split_by_sections(text)

        for section_text, section_ref in sections:
            section_chunks = _split_by_size(section_text, chunk_size, overlap)
            for chunk_text in section_chunks:
                chunk = Chunk(
                    id=uuid.uuid4().hex[:16],
                    content=chunk_text,
                    section_ref=section_ref or metadata.get("source", "unknown"),
                    metadata={
                        **metadata,
                        "chunk_type": "hierarchical",
                        "references": ",".join(refs[:10]),
                    },
                )
                chunks.append(chunk)

    logger.info("Produced %d hierarchical chunks from %d documents", len(chunks), len(documents))
    return chunks


def propositional_chunk(documents: list[dict]) -> list[Chunk]:
    """Use Claude to extract structured propositions from text.

    Tags each chunk with its FactType (rule, exception, penalty, condition, definition).
    """
    chunks: list[Chunk] = []

    for doc in documents:
        text = doc["text"]
        metadata = doc.get("metadata", {})

        # Process in windows to stay within context limits
        windows = _split_by_size(text, 2000, 100)

        from src.llm.client import get_llm_response

        for window in windows:
            prompt = (
                "Extract structured propositions from the following legal text. "
                "For each proposition, provide:\n"
                "- fact_type: one of [rule, exception, penalty, condition, definition]\n"
                "- content: the proposition text\n"
                "- section_ref: the section reference if identifiable\n\n"
                "Return a JSON list of objects. Text:\n\n" + window
            )

            try:
                raw = get_llm_response(prompt)
                # Try to extract JSON from the response
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                propositions = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse propositions, falling back to raw chunk")
                propositions = [{"fact_type": "rule", "content": window, "section_ref": ""}]

            for prop in propositions:
                fact_type_str = prop.get("fact_type", "rule")
                try:
                    fact_type = FactType(fact_type_str)
                except ValueError:
                    fact_type = FactType.RULE

                chunk = Chunk(
                    id=uuid.uuid4().hex[:16],
                    content=prop.get("content", window),
                    section_ref=prop.get("section_ref", metadata.get("source", "unknown")),
                    metadata={
                        **metadata,
                        "chunk_type": "propositional",
                        "fact_type": fact_type.value,
                    },
                )
                chunks.append(chunk)

    logger.info("Produced %d propositional chunks from %d documents", len(chunks), len(documents))
    return chunks


def _split_by_sections(text: str) -> list[tuple[str, str]]:
    """Split text by section headers. Returns (text, section_ref) pairs."""
    pattern = re.compile(r"(?m)^(?:SECTION|Section|ARTICLE|Article|CHAPTER|Chapter)\s+(\S+)")
    matches = list(pattern.finditer(text))

    if not matches:
        return [(text, "")]

    sections: list[tuple[str, str]] = []

    # Text before first section
    if matches[0].start() > 0:
        sections.append((text[:matches[0].start()].strip(), "preamble"))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        section_ref = match.group(0).strip()
        if section_text:
            sections.append((section_text, section_ref))

    return sections


def _split_by_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of roughly chunk_size chars with overlap."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks
