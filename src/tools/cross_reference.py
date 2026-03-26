"""Cross-reference resolution - finds and resolves section references."""

import logging

from src.schema.models import Chunk
from src.utils.references import extract_section_refs, parse_section_ref
from config.settings import settings

logger = logging.getLogger(__name__)


def resolve_cross_reference(
    reference_string: str, context: str = ""
) -> Chunk | None:
    """Parse a reference string and resolve it to a document chunk.

    Args:
        reference_string: Something like "31.020(a)(1)" or "Section 4.2(b)".
        context: Optional surrounding text to improve matching.

    Returns:
        The resolved Chunk with a confidence score, or None if not found.
    """
    parsed = parse_section_ref(reference_string)
    section = parsed["section"]

    # Try graph DB first - exact match on section_ref
    chunk = _resolve_via_graph(section, parsed)
    if chunk is not None:
        return chunk

    # Fall back to vector search with section filter
    chunk = _resolve_via_vector(reference_string, section, context)
    return chunk


def cross_reference_search(query: str) -> list[Chunk]:
    """Search for cross-references mentioned in the query text.

    Extracts section references from the query, resolves each one, and
    returns any chunks found. Intended for use as a graph-node tool.
    """
    refs = extract_section_refs(query)
    if not refs:
        # No explicit refs found - try treating the whole query as a reference
        result = resolve_cross_reference(query, context="")
        return [result] if result else []

    chunks: list[Chunk] = []
    for ref in refs:
        result = resolve_cross_reference(ref, context=query)
        if result is not None:
            chunks.append(result)
    return chunks


def _resolve_via_graph(section: str, parsed: dict) -> Chunk | None:
    """Look up the referenced section in the knowledge graph."""
    try:
        from src.retrieval.shared import get_neo4j
        neo4j = get_neo4j()
    except Exception:
        logger.exception("Failed to connect to Neo4j for cross-ref resolution")
        return None

    try:
        results = neo4j.find_entity(section, label="Section")
    except Exception:
        logger.exception("Graph lookup failed for section '%s'", section)
        return None

    if not results:
        return None

    node = results[0].get("n", {})
    if isinstance(node, dict):
        return Chunk(
            id=node.get("id", section),
            content=node.get("content", node.get("text", "")),
            section_ref=section,
            metadata={"source": "graph", "subsection": parsed.get("subsection", "")},
            score=1.0,
        )
    return None


def _resolve_via_vector(
    reference_string: str, section: str, context: str
) -> Chunk | None:
    """Fall back to vector search for the reference."""
    try:
        from src.retrieval.shared import get_encoder, get_chroma
        encoder = get_encoder()
        chroma = get_chroma()
    except Exception:
        logger.exception("Failed to connect to ChromaDB for cross-ref resolution")
        return None

    search_text = f"{reference_string} {context}".strip()
    query_vector = encoder.encode(search_text).tolist()

    try:
        hits = chroma.search(
            collection=settings.chroma_collection_name,
            query_vector=query_vector,
            filters={"section_ref": section} if section else None,
            limit=3,
        )
    except Exception:
        logger.exception("ChromaDB search failed for cross-ref '%s'", reference_string)
        return None

    if not hits:
        return None

    best = hits[0]
    payload = best.get("payload", {})
    return Chunk(
        id=str(best["id"]),
        content=payload.get("content", ""),
        section_ref=payload.get("section_ref", section),
        metadata={k: str(v) for k, v in payload.items()
                  if k not in ("content", "section_ref")},
        score=best.get("score", 0.0),
    )
