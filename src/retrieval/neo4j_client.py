"""Neo4j graph database client for entity and relationship queries."""

import logging
from typing import Any

from neo4j import GraphDatabase

from config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jManager:
    """Manages connections and operations against a Neo4j instance."""

    def __init__(self) -> None:
        try:
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", settings.neo4j_uri)
        except Exception:
            logger.exception("Failed to connect to Neo4j")
            raise

    # -- read operations --------------------------------------------------------

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a read transaction with parameterized Cypher."""
        params = params or {}
        try:
            with self._driver.session(database=settings.neo4j_database) as session:
                result = session.execute_read(
                    lambda tx: list(tx.run(cypher, **params))
                )
                return [record.data() for record in result]
        except Exception:
            logger.exception("Neo4j read query failed: %s", cypher[:120])
            raise

    # -- write operations -------------------------------------------------------

    def write(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a write transaction with parameterized Cypher."""
        params = params or {}
        try:
            with self._driver.session(database=settings.neo4j_database) as session:
                result = session.execute_write(
                    lambda tx: list(tx.run(cypher, **params))
                )
                return [record.data() for record in result]
        except Exception:
            logger.exception("Neo4j write query failed: %s", cypher[:120])
            raise

    # -- convenience finders ----------------------------------------------------

    def find_entity(
        self, name: str, label: str | None = None
    ) -> list[dict[str, Any]]:
        """Find a node by name and optional label."""
        if label:
            cypher = f"MATCH (n:{label}) WHERE n.name = $name RETURN n"
        else:
            cypher = "MATCH (n) WHERE n.name = $name RETURN n"
        return self.query(cypher, {"name": name})

    def find_relationships(
        self,
        entity: str,
        rel_type: str | None = None,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Traverse relationships from an entity up to N hops."""
        rel_pattern = f"[r:{rel_type}*1..{depth}]" if rel_type else f"[r*1..{depth}]"
        cypher = (
            f"MATCH (n)-{rel_pattern}-(m) "
            "WHERE n.name = $entity "
            "RETURN n, r, m"
        )
        return self.query(cypher, {"entity": entity})

    def find_path(
        self, from_entity: str, to_entity: str
    ) -> list[dict[str, Any]]:
        """Find shortest path between two nodes."""
        cypher = (
            "MATCH p = shortestPath((a)-[*]-(b)) "
            "WHERE a.name = $from_entity AND b.name = $to_entity "
            "RETURN p"
        )
        return self.query(cypher, {"from_entity": from_entity, "to_entity": to_entity})

    # -- lifecycle --------------------------------------------------------------

    def close(self) -> None:
        """Shut down the driver cleanly."""
        self._driver.close()
        logger.info("Neo4j driver closed")
