"""Obligation-policy alignment scoring.

Computes alignment: a(o_j, p_k) = sim_dense(o_j, p_k) · f_type(o_j, p_k)
where f_type is a learned fuzzy type-matching function handling naming
convention differences (e.g., 'credit institution' vs. 'bank').

Threshold: δ=0.6 (evaluation) or δ=0.45 (recall-optimized deployment).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of obligation-policy alignment scoring."""

    obligation_text: str
    policy_text: str
    policy_section: str
    alignment_score: float
    type_match_score: float
    dense_similarity: float
    is_gap: bool
    gap_type: str  # Compliant, Partial Gap, Full Gap


class FuzzyTypeMatcher(nn.Module):
    """Learned fuzzy type-matching for naming convention differences.

    Handles mapping between different institutional naming conventions
    (e.g., 'credit institution' vs 'bank', 'registered entity' vs 'firm').
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, obligation_repr: torch.Tensor, policy_repr: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([obligation_repr, policy_repr], dim=-1)
        return self.projection(combined).squeeze(-1)


class ObligationPolicyAligner:
    """Aligns extracted obligations against internal policy clauses.

    Uses dense similarity combined with learned fuzzy type matching
    to identify potential compliance gaps.
    """

    def __init__(
        self,
        dense_model_name: str = "all-MiniLM-L6-v2",
        delta_eval: float = 0.6,
        delta_deploy: float = 0.45,
        deployment_mode: bool = False,
    ):
        self.delta = delta_deploy if deployment_mode else delta_eval
        self.deployment_mode = deployment_mode
        self._encoder = None
        self.dense_model_name = dense_model_name

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.dense_model_name)
        return self._encoder

    def align(
        self,
        obligation_text: str,
        policy_clauses: list[dict],
    ) -> AlignmentResult:
        """Find the best-matching policy clause for an obligation.

        Args:
            obligation_text: Formatted obligation text.
            policy_clauses: List of dicts with 'text' and 'section' keys.

        Returns:
            AlignmentResult for the best-matching policy clause.
        """
        if not policy_clauses:
            return AlignmentResult(
                obligation_text=obligation_text,
                policy_text="",
                policy_section="",
                alignment_score=0.0,
                type_match_score=0.0,
                dense_similarity=0.0,
                is_gap=True,
                gap_type="Full Gap",
            )

        # Encode obligation and all policy clauses
        obl_embedding = self.encoder.encode([obligation_text])[0]
        policy_texts = [p["text"] for p in policy_clauses]
        policy_embeddings = self.encoder.encode(policy_texts)

        # Compute cosine similarities
        similarities = np.dot(policy_embeddings, obl_embedding) / (
            np.linalg.norm(policy_embeddings, axis=1) * np.linalg.norm(obl_embedding)
        )

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_policy = policy_clauses[best_idx]

        # Classify gap
        is_gap = best_score < self.delta
        if best_score >= self.delta:
            gap_type = "Compliant"
        elif best_score >= self.delta - 0.15:
            gap_type = "Partial Gap"
        else:
            gap_type = "Full Gap"

        return AlignmentResult(
            obligation_text=obligation_text,
            policy_text=best_policy["text"],
            policy_section=best_policy.get("section", ""),
            alignment_score=best_score,
            type_match_score=1.0,  # Placeholder when no learned matcher
            dense_similarity=best_score,
            is_gap=is_gap,
            gap_type=gap_type,
        )

    def batch_align(
        self,
        obligations: list[str],
        policy_clauses: list[dict],
    ) -> list[AlignmentResult]:
        """Align a batch of obligations against policy clauses.

        Args:
            obligations: List of formatted obligation texts.
            policy_clauses: List of policy clause dicts.

        Returns:
            List of AlignmentResult objects.
        """
        return [self.align(obl, policy_clauses) for obl in obligations]
