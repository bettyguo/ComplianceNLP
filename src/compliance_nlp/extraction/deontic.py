"""Deontic modality classification.

Classifies regulatory sentences into four deontic categories:
  Obligation  – "shall", "must"
  Permission  – "may", "is permitted to"
  Prohibition – "shall not", "must not", "may not"
  Recommendation – "should", "is encouraged to"
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from compliance_nlp.data.datasets import DEONTIC_LABELS

log = logging.getLogger(__name__)

# Keyword heuristics (fallback when no model is available)
_KEYWORDS: dict[str, list[str]] = {
    "Obligation": ["shall", "must", "is required to", "are required to"],
    "Permission": ["may", "is permitted to", "is allowed to", "can"],
    "Prohibition": ["shall not", "must not", "may not", "is prohibited"],
    "Recommendation": ["should", "is encouraged", "is recommended"],
}


def keyword_classify(text: str) -> str:
    """Rule-based deontic classification using keyword matching.

    This serves as a lightweight fallback for quick labelling or as a
    feature for the neural classifier.

    Args:
        text: Regulatory sentence.

    Returns:
        Predicted deontic label string.
    """
    lower = text.lower()
    # Check prohibitions first (they contain negated obligation keywords)
    for label in ["Prohibition", "Obligation", "Permission", "Recommendation"]:
        for kw in _KEYWORDS[label]:
            if kw in lower:
                return label
    return "Obligation"  # conservative default


def softmax_to_label(logits: torch.Tensor) -> tuple[str, float]:
    """Convert deontic head logits to a label and confidence.

    Args:
        logits: (num_classes,) raw model logits.

    Returns:
        Tuple of (label_string, confidence).
    """
    probs = F.softmax(logits, dim=-1)
    idx = int(probs.argmax())
    return DEONTIC_LABELS[idx], float(probs[idx])
