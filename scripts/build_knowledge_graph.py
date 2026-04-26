#!/usr/bin/env python
"""Build Regulatory Knowledge Graph from regulatory document feeds."""

import argparse
import logging

from compliance_nlp.knowledge_graph.builder import KnowledgeGraphBuilder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build Regulatory Knowledge Graph")
    parser.add_argument("--sec-dir", type=str, default=None)
    parser.add_argument("--mifid-dir", type=str, default=None)
    parser.add_argument("--basel-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/knowledge_graph")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="")
    args = parser.parse_args()

    builder = KnowledgeGraphBuilder(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )

    builder.connect()
    builder.create_constraints()

    total = builder.ingest_provisions(
        sec_dir=args.sec_dir,
        mifid_dir=args.mifid_dir,
        basel_dir=args.basel_dir,
    )

    edges = builder.build_cross_reference_edges()
    stats = builder.get_stats()

    log.info(f"KG built: {total} provisions, {edges} edges")
    log.info(f"Stats: {stats}")

    builder.close()


if __name__ == "__main__":
    main()
