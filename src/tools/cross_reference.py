"""Cross-reference resolution - finds and resolves section references."""

import logging

from src.retrieval.neo4j_client import Neo4jManager
from src.retrieval.qdrant_client import QdrantManager
from src.schema.models import Chunk
from src.utils.references import extract_section_refs, parse_section_ref
from config.settings import settings

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_encoder: SentenceTransformer | None = None
_qdrant: QdrantManager | None = None
_neo4j: Neo4jManager | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(settings.embedding_model)
    return _encoder


def _get_qdrant() -> QdrantManager:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantManager()
    return _qdrant


def _get_neo4j() -> Neo4jManager:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jManager()
    return _neo4j


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
    neo4j = _get_neo4j()

    results = neo4j.find_entity(section, label="Section")
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
    encoder = _get_encoder()
    qdrant = _get_qdrant()

    search_text = f"{reference_string} {context}".strip()
    query_vector = encoder.encode(search_text).tolist()

    hits = qdrant.search(
        collection=settings.qdrant_collection_name,
        query_vector=query_vector,
        filters={"section_ref": section} if section else None,
        limit=3,
    )

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
