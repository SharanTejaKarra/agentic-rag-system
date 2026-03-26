"""Hierarchical lookup - navigate the document structure via the knowledge graph."""

import logging

from src.retrieval.neo4j_client import Neo4jManager
from src.schema.models import Chunk

logger = logging.getLogger(__name__)

_neo4j: Neo4jManager | None = None

# The document hierarchy from broadest to narrowest.
_HIERARCHY_LEVELS = ["Article", "Chapter", "Section", "Subsection"]

# Cypher relationship types for each navigation direction.
_DIRECTION_MAP = {
    "children": "HAS_CHILD",
    "parent": "HAS_CHILD",   # reversed in query
    "siblings": "HAS_CHILD", # find parent, then its children
}


def _get_neo4j() -> Neo4jManager:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jManager()
    return _neo4j


def hierarchical_lookup(
    target_section: str, direction: str = "children"
) -> list[Chunk]:
    """Navigate the document hierarchy from a given section.

    Args:
        target_section: Section identifier (e.g. "4.2", "Article 3").
        direction: One of "parent", "children", or "siblings".

    Returns:
        Related Chunk objects from the hierarchy.
    """
    neo4j = _get_neo4j()

    if direction == "children":
        return _get_children(neo4j, target_section)
    elif direction == "parent":
        return _get_parent(neo4j, target_section)
    elif direction == "siblings":
        return _get_siblings(neo4j, target_section)
    else:
        logger.warning("Unknown direction '%s', defaulting to children", direction)
        return _get_children(neo4j, target_section)


def _get_children(neo4j: Neo4jManager, section: str) -> list[Chunk]:
    cypher = (
        "MATCH (parent)-[:HAS_CHILD]->(child) "
        "WHERE parent.name = $section "
        "RETURN child ORDER BY child.name"
    )
    return _records_to_chunks(neo4j.query(cypher, {"section": section}), "child")


def _get_parent(neo4j: Neo4jManager, section: str) -> list[Chunk]:
    cypher = (
        "MATCH (parent)-[:HAS_CHILD]->(child) "
        "WHERE child.name = $section "
        "RETURN parent"
    )
    return _records_to_chunks(neo4j.query(cypher, {"section": section}), "parent")


def _get_siblings(neo4j: Neo4jManager, section: str) -> list[Chunk]:
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
