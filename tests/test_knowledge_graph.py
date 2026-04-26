"""Tests for Regulatory Knowledge Graph."""

import pytest
from compliance_nlp.knowledge_graph.schema import (
    NodeType, EdgeType, Framework, ProvisionNode, KGStats,
)


class TestSchema:
    def test_node_types(self):
        assert NodeType.PROVISION == "Provision"
        assert NodeType.OBLIGATION == "Obligation"
        assert len(NodeType) == 5

    def test_edge_types(self):
        assert EdgeType.CROSS_REFERENCES == "CrossReferences"
        assert len(EdgeType) == 5

    def test_frameworks(self):
        assert Framework.SEC == "SEC"
        assert Framework.MIFID_II == "MiFID II"
        assert Framework.BASEL_III == "Basel III"

    def test_provision_node(self):
        node = ProvisionNode(
            id="SEC_Rule10b5",
            text="Anti-fraud provisions",
            framework=Framework.SEC,
        )
        assert node.id == "SEC_Rule10b5"
        assert node.framework == Framework.SEC

    def test_kg_stats(self):
        stats = KGStats()
        assert stats.total_provisions == 12_847
        assert stats.total_edges == 34_219
        assert "12,847" in str(stats)
