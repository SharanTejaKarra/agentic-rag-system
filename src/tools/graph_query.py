"""Graph query tool - entity and relationship traversal via Neo4j."""

import logging

from src.schema.models import Chunk
from src.utils.references import extract_section_refs

logger = logging.getLogger(__name__)


def graph_query(
    entity_name: str,
    relationship_type: str | None = None,
    depth: int = 1,
) -> list[Chunk]:
    """Query the knowledge graph for an entity and its relationships.

    When called with a full natural-language query (from the graph node),
    section references are extracted first. If none are found, the raw
    string is used as the entity name.

    Args:
        entity_name: Entity name or a query string to extract entities from.
        relationship_type: Optional relationship type to filter on.
        depth: How many hops to traverse (default 1).

    Returns:
        List of Chunk objects built from matching graph nodes.
    """
    try:
        from src.retrieval.shared import get_neo4j
        neo4j = get_neo4j()
    except Exception:
        logger.exception("Failed to connect to Neo4j")
        return []

    # Extract section refs if this looks like a full query rather than
    # a clean entity name (e.g. "How do Section 12 and Section 31 relate?")
    search_names = extract_section_refs(entity_name)
    if not search_names:
        search_names = [entity_name]

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()

    for name in search_names:
        try:
            # Find the root entity
            root_results = neo4j.find_entity(name)
            for record in root_results:
                node = record.get("n", {})
                chunk = _node_to_chunk(node)
                if chunk and chunk.id not in seen_ids:
                    chunks.append(chunk)
                    seen_ids.add(chunk.id)

            # Traverse relationships
            rel_results = neo4j.find_relationships(
                entity=name,
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
        except Exception:
            logger.exception("Graph query failed for entity '%s'", name)

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
