"""Hierarchical lookup - navigate the document structure via the knowledge graph."""

import logging

from src.schema.models import Chunk
from src.utils.references import extract_section_refs

logger = logging.getLogger(__name__)

# Cypher relationship types for each navigation direction.
_DIRECTION_MAP = {
    "children": "HAS_CHILD",
    "parent": "HAS_CHILD",   # reversed in query
    "siblings": "HAS_CHILD", # find parent, then its children
}


def hierarchical_lookup(
    target_section: str, direction: str = "children"
) -> list[Chunk]:
    """Navigate the document hierarchy from a given section.

    When called with a full query string (from the graph node), section
    references are extracted first. Falls back to treating the whole
    string as a section identifier.

    Args:
        target_section: Section identifier or query to extract refs from.
        direction: One of "parent", "children", or "siblings".

    Returns:
        Related Chunk objects from the hierarchy.
    """
    try:
        from src.retrieval.shared import get_neo4j
        neo4j = get_neo4j()
    except Exception:
        logger.exception("Failed to connect to Neo4j")
        return []

    # Extract section refs if this looks like a full sentence
    sections = extract_section_refs(target_section)
    if not sections:
        sections = [target_section]

    all_chunks: list[Chunk] = []
    seen_ids: set[str] = set()

    for section in sections:
        try:
            if direction == "children":
                chunks = _get_children(neo4j, section)
            elif direction == "parent":
                chunks = _get_parent(neo4j, section)
            elif direction == "siblings":
                chunks = _get_siblings(neo4j, section)
            else:
                logger.warning("Unknown direction '%s', defaulting to children", direction)
                chunks = _get_children(neo4j, section)

            for chunk in chunks:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)
        except Exception:
            logger.exception("Hierarchical lookup failed for section '%s'", section)

    return all_chunks


def _get_children(neo4j, section: str) -> list[Chunk]:
    cypher = (
        "MATCH (parent)-[:HAS_CHILD]->(child) "
        "WHERE parent.name = $section "
        "RETURN child ORDER BY child.name"
    )
    return _records_to_chunks(neo4j.query(cypher, {"section": section}), "child")


def _get_parent(neo4j, section: str) -> list[Chunk]:
    cypher = (
        "MATCH (parent)-[:HAS_CHILD]->(child) "
        "WHERE child.name = $section "
        "RETURN parent"
    )
    return _records_to_chunks(neo4j.query(cypher, {"section": section}), "parent")


def _get_siblings(neo4j, section: str) -> list[Chunk]:
    cypher = (
        "MATCH (parent)-[:HAS_CHILD]->(child) "
        "WHERE child.name = $section "
        "WITH parent "
        "MATCH (parent)-[:HAS_CHILD]->(sibling) "
        "WHERE sibling.name <> $section "
        "RETURN sibling ORDER BY sibling.name"
    )
    return _records_to_chunks(neo4j.query(cypher, {"section": section}), "sibling")


def _records_to_chunks(records: list[dict], node_key: str) -> list[Chunk]:
    """Convert Neo4j records into Chunk objects."""
    chunks: list[Chunk] = []
    for record in records:
        node = record.get(node_key, {})
        if not isinstance(node, dict):
            continue
        chunks.append(
            Chunk(
                id=node.get("id", node.get("name", "")),
                content=node.get("content", node.get("text", "")),
                section_ref=node.get("name", ""),
                metadata={"source": "graph", "level": node.get("level", "")},
                score=1.0,
            )
        )
    return chunks
