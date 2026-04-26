"""Hybrid dense+sparse retrieval with KG-based re-ranking.

Retrieval score: s(q,d) = α·sim_dense(q,d) + (1-α)·BM25(q,d)
KG re-ranking:   s_KG(q,d) = β·KGScore(q,d,G) + (1-β)·s(q,d)

Default: α=0.7, β=0.3, top_k=5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    """A retrieved passage with scoring metadata."""

    text: str
    provision_id: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    kg_score: float = 0.0
    final_score: float = 0.0
    graph_distance: Optional[int] = None


class HybridRetriever:
    """Hybrid dense+sparse retrieval pipeline with KG re-ranking.

    Combines a dense bi-encoder (legal-domain fine-tuned from all-MiniLM-L6-v2
    on 50K regulatory passage pairs) with BM25 sparse retrieval, then re-ranks
    using knowledge graph proximity scores.
    """

    def __init__(
        self,
        dense_model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 0.7,
        beta: float = 0.3,
        top_k: int = 5,
        faiss_index_path: Optional[str] = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k
        self.dense_model_name = dense_model_name

        self._dense_model = None
        self._faiss_index = None
        self._bm25 = None
        self._corpus: list[dict] = []
        self._corpus_texts: list[str] = []

    @property
    def dense_model(self):
        """Lazy-load dense encoder."""
        if self._dense_model is None:
            from sentence_transformers import SentenceTransformer

            self._dense_model = SentenceTransformer(self.dense_model_name)
            log.info(f"Loaded dense encoder: {self.dense_model_name}")
        return self._dense_model

    def index_corpus(self, documents: list[dict]) -> None:
        """Build retrieval indices from a corpus of regulatory passages.

        Args:
            documents: List of dicts with 'text', 'provision_id' keys.
        """
        import faiss

        self._corpus = documents
        self._corpus_texts = [doc["text"] for doc in documents]

        # Build BM25 index
        tokenized = [text.lower().split() for text in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)

        # Build FAISS dense index
        embeddings = self.dense_model.encode(
            self._corpus_texts, show_progress_bar=True, batch_size=64
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self._faiss_index.add(embeddings)

        log.info(
            f"Indexed {len(documents)} passages "
            f"(dense dim={dim}, BM25 vocab={len(self._bm25.idf)})"
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        kg_engine=None,
        query_provision_id: Optional[str] = None,
    ) -> list[RetrievedPassage]:
        """Retrieve and optionally KG-rerank passages for a query.

        Args:
            query: Query text (regulatory obligation or question).
            top_k: Number of passages to return (default: self.top_k).
            kg_engine: Optional KGQueryEngine for re-ranking.
            query_provision_id: Source provision ID for KG scoring.

        Returns:
            List of RetrievedPassage objects, sorted by final score.
        """
        top_k = top_k or self.top_k
        candidates = self._hybrid_retrieve(query, top_k=top_k * 2)

        # KG re-ranking
        if kg_engine and query_provision_id:
            for passage in candidates:
                kg_score = kg_engine.compute_kg_score(
                    query_provision_id, passage.provision_id
                )
                passage.kg_score = kg_score
                passage.final_score = (
                    self.beta * kg_score + (1 - self.beta) * passage.hybrid_score
                )
                distance = kg_engine.compute_graph_distance(
                    query_provision_id, passage.provision_id
                )
                passage.graph_distance = distance
        else:
            for passage in candidates:
                passage.final_score = passage.hybrid_score

        # Sort by final score and return top_k
        candidates.sort(key=lambda p: p.final_score, reverse=True)
        return candidates[:top_k]

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[RetrievedPassage]:
        """Perform hybrid dense+sparse retrieval.

        Args:
            query: Query text.
            top_k: Number of candidates to retrieve.

        Returns:
            List of RetrievedPassage with hybrid scores.
        """
        import faiss

        # Dense retrieval
        query_embedding = self.dense_model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        dense_scores, dense_indices = self._faiss_index.search(query_embedding, top_k)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Combine scores for candidates from dense retrieval
        candidate_set = set(dense_indices.tolist())

        # Also add top BM25 candidates
        bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        candidate_set.update(bm25_top_indices.tolist())

        # Normalize BM25 scores to [0, 1]
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0

        passages = []
        for idx in candidate_set:
            if idx < 0 or idx >= len(self._corpus):
                continue
            doc = self._corpus[idx]

            d_score = 0.0
            if idx in dense_indices:
                pos = list(dense_indices).index(idx)
                d_score = float(dense_scores[pos])

            s_score = float(bm25_scores[idx]) / bm25_max

            hybrid = self.alpha * d_score + (1 - self.alpha) * s_score

            passages.append(
                RetrievedPassage(
                    text=doc["text"],
                    provision_id=doc["provision_id"],
                    dense_score=d_score,
                    sparse_score=s_score,
                    hybrid_score=hybrid,
                )
            )

        passages.sort(key=lambda p: p.hybrid_score, reverse=True)
        return passages[:top_k]
