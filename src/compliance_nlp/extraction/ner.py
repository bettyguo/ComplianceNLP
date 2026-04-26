"""CRF-based regulatory Named Entity Recognition.

Recognises 23 entity types spanning financial regulation across SEC,
MiFID II, and Basel III.  Uses a CRF layer on top of contextual
representations from the shared LEGAL-BERT encoder.

Entity types range from REGULATORY_BODY and REPORTING_ENTITY to
domain-specific types like CAPITAL_REQUIREMENT and EXEMPTION_CLAUSE.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from compliance_nlp.data.datasets import ENTITY_TYPES, ID2LABEL, LABEL2ID

log = logging.getLogger(__name__)


def decode_bio_tags(token_ids: list[int], tokens: list[str]) -> list[dict]:
    """Convert BIO tag-id sequences into entity spans.

    Args:
        token_ids: Predicted tag indices per token.
        tokens: Original tokens (or sub-word pieces).

    Returns:
        List of entity dicts with *text*, *label*, *start*, *end*.
    """
    entities: list[dict] = []
    current: dict | None = None

    for idx, (tag_id, token) in enumerate(zip(token_ids, tokens)):
        label = ID2LABEL.get(tag_id, "O")

        if label.startswith("B-"):
            if current is not None:
                entities.append(current)
            current = {
                "label": label[2:],
                "tokens": [token],
                "start": idx,
                "end": idx + 1,
            }
        elif label.startswith("I-") and current is not None:
            if label[2:] == current["label"]:
                current["tokens"].append(token)
                current["end"] = idx + 1
            else:
                entities.append(current)
                current = None
        else:
            if current is not None:
                entities.append(current)
                current = None

    if current is not None:
        entities.append(current)

    for ent in entities:
        ent["text"] = " ".join(ent.pop("tokens"))

    return entities


def entity_level_f1(
    pred_entities: list[list[dict]],
    gold_entities: list[list[dict]],
) -> dict[str, float]:
    """Compute entity-level precision / recall / F1.

    Matching criterion: exact span boundaries **and** label match.

    Args:
        pred_entities: Per-sample list of predicted entities.
        gold_entities: Per-sample list of gold entities.

    Returns:
        Dict with *precision*, *recall*, *f1*.
    """
    tp = fp = fn = 0

    for preds, golds in zip(pred_entities, gold_entities):
        pred_set = {(e["label"], e["start"], e["end"]) for e in preds}
        gold_set = {(e["label"], e["start"], e["end"]) for e in golds}

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}
