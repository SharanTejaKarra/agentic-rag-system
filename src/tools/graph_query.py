"""Graph query tool - entity and relationship traversal via Neo4j."""

import logging

from src.retrieval.neo4j_client import Neo4jManager
from src.schema.models import Chunk

logger = logging.getLogger(__name__)

_neo4j: Neo4jManager | None = None


def _get_neo4j() -> Neo4jManager:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jManager()
    return _neo4j


def graph_query(
    entity_name: str,
    relationship_type: str | None = None,
    depth: int = 1,
) -> list[Chunk]:
    """Query the knowledge graph for an entity and its relationships.

    When called with just a string (e.g. from the graph node), treats it as
    an entity name and searches with default depth.

    Args:
        entity_name: Name of the entity to look up.
        relationship_type: Optional relationship type to filter on.
        depth: How many hops to traverse (default 1).

    Returns:
        List of Chunk objects built from matching graph nodes.
    """
    neo4j = _get_neo4j()
    chunks: list[Chunk] = []
    seen_ids: set[str] = set()

    # Find the root entity
    root_results = neo4j.find_entity(entity_name)
    for record in root_results:
        node = record.get("n", {})
        chunk = _node_to_chunk(node)
        if chunk and chunk.id not in seen_ids:
            chunks.append(chunk)
            seen_ids.add(chunk.id)

    # Traverse relationships
    rel_results = neo4j.find_relationships(
        entity=entity_name,
        rel_type=relationship_type,
        depth=depth,
    )
    for record in rel_results:
        for key in ("n", "m"):
            node = record.get(key, {})
            chunk = _node_to_chunk(node)
            if chunk and chunk.id not in seen_ids:
                chunks.append(chunk)
                seen_ids.add(chunk.id)

    return chunks


def _node_to_chunk(node: object) -> Chunk | None:
    """Convert a Neo4j node (dict or Node object) to a Chunk."""
    if not node:
        return None
    if isinstance(node, dict):
        props = node
    elif hasattr(node, "items"):
        props = dict(node)
    else:
        return None

    node_id = str(props.get("id", props.get("name", "")))
    if not node_id:
        return None

    return Chunk(
        id=node_id,
        content=props.get("content", props.get("text", "")),
        section_ref=props.get("name", props.get("section_ref", "")),
        metadata={
            "source": "graph",
            **{k: str(v) for k, v in props.items()
               if k not in ("id", "content", "text", "name", "section_ref")},
        },
        score=1.0,
    )
