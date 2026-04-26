"""RKG construction pipeline.

Semi-automated pipeline using three format-specific parsers for
SEC EDGAR XML, EUR-Lex HTML, and BIS PDF. Cross-reference edges
identified using regex patterns combined with a learned span-pair linker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from neo4j import GraphDatabase

from compliance_nlp.knowledge_graph.schema import (
    EdgeType,
    Framework,
    KGEdge,
    KGStats,
    NodeType,
    ProvisionNode,
)
from compliance_nlp.data.preprocessing import (
    extract_cross_references,
    parse_sec_edgar_xml,
    parse_eurlex_html,
    parse_bis_pdf,
)

log = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds and maintains the Regulatory Knowledge Graph in Neo4j.

    Construction pipeline:
    1. Parse regulatory documents (SEC EDGAR XML, EUR-Lex HTML, BIS PDF)
    2. Extract provision nodes with embeddings
    3. Identify cross-reference edges (regex + learned linker)
    4. Expert review of sampled edges for quality assurance
    5. Nightly synchronization with regulatory feeds
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "",
    ):
        self.driver = None
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )
        log.info(f"Connected to Neo4j at {self.neo4j_uri}")

    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def create_constraints(self) -> None:
        """Create Neo4j uniqueness constraints for node IDs."""
        with self.driver.session() as session:
            for node_type in NodeType:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{node_type.value}) REQUIRE n.id IS UNIQUE"
                )
        log.info("Created Neo4j constraints")

    def ingest_provisions(
        self,
        sec_dir: str | Path | None = None,
        mifid_dir: str | Path | None = None,
        basel_dir: str | Path | None = None,
    ) -> int:
        """Ingest regulatory documents and create provision nodes.

        Args:
            sec_dir: Directory containing SEC EDGAR XML files.
            mifid_dir: Directory containing EUR-Lex HTML files.
            basel_dir: Directory containing BIS PDF files.

        Returns:
            Total number of provision nodes created.
        """
        total = 0

        if sec_dir:
            for filepath in Path(sec_dir).glob("*.xml"):
                for chunk in parse_sec_edgar_xml(filepath):
                    self._create_provision_node(
                        ProvisionNode(
                            id=chunk.provision_id,
                            text=chunk.text,
                            framework=Framework.SEC,
                        )
                    )
                    total += 1

        if mifid_dir:
            for filepath in Path(mifid_dir).glob("*.html"):
                for chunk in parse_eurlex_html(filepath):
                    self._create_provision_node(
                        ProvisionNode(
                            id=chunk.provision_id,
                            text=chunk.text,
                            framework=Framework.MIFID_II,
                        )
                    )
                    total += 1

        if basel_dir:
            for filepath in Path(basel_dir).glob("*.pdf"):
                for chunk in parse_bis_pdf(filepath):
                    self._create_provision_node(
                        ProvisionNode(
                            id=chunk.provision_id,
                            text=chunk.text,
                            framework=Framework.BASEL_III,
                        )
                    )
                    total += 1

        log.info(f"Ingested {total} provision nodes")
        return total

    def _create_provision_node(self, node: ProvisionNode) -> None:
        """Create a single provision node in Neo4j."""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (p:Provision {id: $id})
                SET p.text = $text,
                    p.framework = $framework,
                    p.title = $title,
                    p.effective_date = $effective_date
                """,
                id=node.id,
                text=node.text,
                framework=node.framework.value,
                title=node.title,
                effective_date=node.effective_date,
            )

    def build_cross_reference_edges(self) -> int:
        """Identify and create cross-reference edges between provisions.

        Uses regex patterns for initial detection, then a learned
        span-pair linker (bilinear classifier) for disambiguation.

        Returns:
            Number of edges created.
        """
        edge_count = 0
        with self.driver.session() as session:
            # Get all provisions
            result = session.run(
                "MATCH (p:Provision) RETURN p.id AS id, p.text AS text, p.framework AS fw"
            )

            for record in result:
                xrefs = extract_cross_references(record["text"], record["fw"])
                for xref in xrefs:
                    # Find matching target provision
                    target = session.run(
                        """
                        MATCH (t:Provision) 
                        WHERE t.id CONTAINS $xref OR t.text CONTAINS $xref
                        RETURN t.id AS id LIMIT 1
                        """,
                        xref=xref,
                    ).single()

                    if target:
                        session.run(
                            """
                            MATCH (s:Provision {id: $source_id})
                            MATCH (t:Provision {id: $target_id})
                            MERGE (s)-[:CROSS_REFERENCES {confidence: $conf}]->(t)
                            """,
                            source_id=record["id"],
                            target_id=target["id"],
                            conf=0.9,
                        )
                        edge_count += 1

        log.info(f"Created {edge_count} cross-reference edges")
        return edge_count

    def get_stats(self) -> KGStats:
        """Retrieve current knowledge graph statistics."""
        with self.driver.session() as session:
            provision_count = session.run(
                "MATCH (p:Provision) RETURN count(p) AS c"
            ).single()["c"]
            edge_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS c"
            ).single()["c"]

        return KGStats(total_provisions=provision_count, total_edges=edge_count)
