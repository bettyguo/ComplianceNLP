"""Tests for hybrid retrieval pipeline."""

import pytest
from compliance_nlp.retrieval.hybrid import HybridRetriever, RetrievedPassage


class TestHybridRetriever:
    def test_init(self):
        retriever = HybridRetriever(alpha=0.7, beta=0.3, top_k=5)
        assert retriever.alpha == 0.7
        assert retriever.beta == 0.3
        assert retriever.top_k == 5

    def test_retrieved_passage(self):
        passage = RetrievedPassage(
            text="Sample text",
            provision_id="SEC_123",
            dense_score=0.8,
            sparse_score=0.6,
            hybrid_score=0.74,
        )
        assert passage.hybrid_score == 0.74
        assert passage.provision_id == "SEC_123"
