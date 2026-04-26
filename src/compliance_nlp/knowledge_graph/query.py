"""Graph traversal and KG scoring for retrieval re-ranking."""

from __future__ import annotations

import logging
from typing import Optional

from neo4j import GraphDatabase

log = logging.getLogger(__name__)


class KGQueryEngine:
    """Queries the Regulatory Knowledge Graph for re-ranking and traversal.

    Supports graph distance computation between provisions, multi-hop
    cross-reference traversal, and KG-score computation for passage re-ranking.
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "",
        max_hops: int = 3,
    ):
        self.driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        self.max_hops = max_hops

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def compute_graph_distance(
        self, source_id: str, target_id: str
    ) -> Optional[int]:
        """Compute shortest path distance between two provisions.

        Args:
            source_id: Source provision ID.
            target_id: Target provision ID.

        Returns:
            Shortest path length, or None if no path exists within max_hops.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath(
                    (s:Provision {id: $source_id})-[*..{max_hops}]-
                    (t:Provision {id: $target_id})
                )
                RETURN length(path) AS dist
                """.replace("{max_hops}", str(self.max_hops)),
                source_id=source_id,
                target_id=target_id,
            ).single()

        return result["dist"] if result else None

    def compute_kg_score(
        self, query_provision_id: str, candidate_provision_id: str
    ) -> float:
        """Compute KG-based relevance score for retrieval re-ranking.

        Score decreases with graph distance: score = 1 / (1 + distance).
        Returns 0.0 if no path exists.

        Args:
            query_provision_id: ID of the query's source provision.
            candidate_provision_id: ID of the candidate passage's provision.

        Returns:
            KG score between 0.0 and 1.0.
        """
        distance = self.compute_graph_distance(
            query_provision_id, candidate_provision_id
        )
        if distance is None:
            return 0.0
        return 1.0 / (1.0 + distance)

    def get_cross_references(
        self, provision_id: str, max_depth: int = 2
    ) -> list[dict]:
        """Retrieve all cross-referenced provisions up to max_depth hops.

        Args:
            provision_id: Source provision ID.
            max_depth: Maximum traversal depth.

        Returns:
            List of dicts with 'id', 'text', 'distance', 'edge_type'.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (s:Provision {id: $pid})-[r*1..{depth}]->(t:Provision)
                WITH t, length(path) AS dist,
                     [rel IN relationships(path) | type(rel)] AS edge_types
                RETURN t.id AS id, t.text AS text, dist,
                       edge_types[size(edge_types)-1] AS edge_type
                ORDER BY dist
                """.replace("{depth}", str(max_depth)),
                pid=provision_id,
            )

            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "distance": record["dist"],
                    "edge_type": record["edge_type"],
                }
                for record in result
            ]

    def get_enforcement_history(self, provision_id: str) -> list[dict]:
        """Retrieve enforcement actions related to a provision.

        Args:
            provision_id: Provision to check enforcement history for.

        Returns:
            List of enforcement action dicts.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Provision {id: $pid})<-[:RELATES_TO]-(e:Enforcement)
                RETURN e.entity_name AS entity, e.penalty_amount AS penalty,
                       e.date AS date, e.description AS description
                ORDER BY e.date DESC
                """,
                pid=provision_id,
            )
            return [dict(record) for record in result]
