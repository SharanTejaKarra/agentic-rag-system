"""Build a knowledge graph in Neo4j from chunked documents."""

import json

from neo4j import Driver, GraphDatabase

from src.schema.models import Chunk
from src.llm.client import get_llm_response
from config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_driver = None


def _get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


def build_knowledge_graph(chunks: list[Chunk]) -> dict:
    """Extract entities and relationships from chunks and store in Neo4j.

    Uses Claude to extract structured entity/relationship data, then creates
    Neo4j nodes and edges. Also creates hierarchy edges between sections.

    Returns stats dict with counts of nodes and edges created.
    """
    if not chunks:
        return {"nodes_created": 0, "edges_created": 0}

    driver = _get_driver()
    total_nodes = 0
    total_edges = 0

    # Process chunks in groups to reduce LLM calls
    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        combined_text = "\n\n---\n\n".join(
            f"[{c.section_ref}]: {c.content}" for c in batch
        )

        prompt = (
            "Extract entities and relationships from the following legal text sections. "
            "Return a JSON object with two keys:\n"
            '- "entities": list of {label, type, properties} where type is one of '
            "[Section, Applicant, Permit, Penalty, Fee, Deadline, Authority, Condition]\n"
            '- "relationships": list of {source, target, type} where type is one of '
            "[REFERENCES, DEFINES, HAS_PENALTY, APPLIES_TO, REQUIRES, AMENDS, SUPERSEDES]\n\n"
            f"Text:\n{combined_text}"
        )

        try:
            raw = get_llm_response(prompt)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse graph extraction for batch %d, skipping", i)
            continue

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        with driver.session(database=settings.neo4j_database) as session:
            # Create entity nodes
            for entity in entities:
                label = entity.get("label", "Unknown")
                node_type = entity.get("type", "Entity")
                props = entity.get("properties", {})
                props["label"] = label

                session.execute_write(
                    _create_node, node_type, props
                )
                total_nodes += 1

            # Create relationship edges
            for rel in relationships:
                session.execute_write(
                    _create_edge,
                    rel.get("source", ""),
                    rel.get("target", ""),
                    rel.get("type", "RELATED_TO"),
                )
                total_edges += 1

    # Build hierarchy edges from section references
    hierarchy_edges = _build_hierarchy_edges(chunks)
    with driver.session(database=settings.neo4j_database) as session:
        for parent, child, rel_type in hierarchy_edges:
            session.execute_write(_create_edge, parent, child, rel_type)
            total_edges += 1

    stats = {"nodes_created": total_nodes, "edges_created": total_edges}
    logger.info("Knowledge graph built: %d nodes, %d edges", total_nodes, total_edges)
    return stats


def _create_node(tx, node_type: str, properties: dict) -> None:
    """Create a node with the given type and properties."""
    props_str = ", ".join(f"{k}: ${k}" for k in properties)
    query = f"MERGE (n:{node_type} {{{props_str}}})"
    tx.run(query, **properties)


def _create_edge(tx, source_label: str, target_label: str, rel_type: str) -> None:
    """Create a relationship between two nodes identified by label."""
    query = (
        "MATCH (a {label: $source}), (b {label: $target}) "
        f"MERGE (a)-[:{rel_type}]->(b)"
    )
    tx.run(query, source=source_label, target=target_label)


def _build_hierarchy_edges(chunks: list[Chunk]) -> list[tuple[str, str, str]]:
    """Derive parent/child/sibling edges from section references."""
    edges: list[tuple[str, str, str]] = []
    section_refs = [c.section_ref for c in chunks if c.section_ref]

    # Sort to get hierarchical order
    sorted_refs = sorted(set(section_refs))

    for i, ref in enumerate(sorted_refs):
        # Find parent (shorter prefix)
        parts = ref.split(".")
        if len(parts) > 1:
            parent_ref = ".".join(parts[:-1])
            if parent_ref in sorted_refs:
                edges.append((parent_ref, ref, "PARENT_OF"))
                edges.append((ref, parent_ref, "CHILD_OF"))

        # Sibling detection
        if i + 1 < len(sorted_refs):
            next_ref = sorted_refs[i + 1]
            next_parts = next_ref.split(".")
            if len(parts) == len(next_parts) and parts[:-1] == next_parts[:-1]:
                edges.append((ref, next_ref, "SIBLING_OF"))

    return edges
