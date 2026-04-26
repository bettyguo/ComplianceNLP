"""Quality metrics for compliance NLP evaluation.

Supports NER F1, deontic accuracy, gap detection F1 (macro/per-class),
grounding accuracy, exact match, and paired bootstrap significance testing.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

log = logging.getLogger(__name__)


def compute_ner_f1(
    predictions: list[list[int]],
    references: list[list[int]],
    ignore_index: int = 0,
) -> float:
    """Compute micro-averaged NER F1 score (excluding O tags).

    Args:
        predictions: Nested list of predicted tag indices.
        references: Nested list of reference tag indices.
        ignore_index: Tag index to exclude (typically 'O' tag).

    Returns:
        Micro-averaged F1 score.
    """
    tp, fp, fn = 0, 0, 0

    for pred_seq, ref_seq in zip(predictions, references):
        for p, r in zip(pred_seq, ref_seq):
            if r == -100:
                continue
            if p != ignore_index and r != ignore_index:
                if p == r:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif p != ignore_index:
                fp += 1
            elif r != ignore_index:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def compute_gap_detection_f1(
    predictions: list[str],
    references: list[str],
    labels: list[str] = None,
) -> dict:
    """Compute per-class and macro-averaged gap detection F1.

    Args:
        predictions: List of predicted gap labels.
        references: List of reference gap labels.
        labels: Optional label list. Defaults to standard gap labels.

    Returns:
        Dict with per-class P/R/F1 and macro averages.
    """
    if labels is None:
        labels = ["Compliant", "Partial Gap", "Full Gap"]

    metrics = {}
    f1_scores = []

    for label in labels:
        tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
        fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
        fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"{label}_precision"] = precision
        metrics[f"{label}_recall"] = recall
        metrics[f"{label}_f1"] = f1
        f1_scores.append(f1)

    metrics["macro_precision"] = np.mean(
        [metrics[f"{l}_precision"] for l in labels]
    )
    metrics["macro_recall"] = np.mean(
        [metrics[f"{l}_recall"] for l in labels]
    )
    metrics["macro_f1"] = np.mean(f1_scores)

    return metrics


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Compute exact match accuracy."""
    if not predictions:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)


def paired_bootstrap_test(
    system_a_scores: list[float],
    system_b_scores: list[float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap significance test.

    Args:
        system_a_scores: Per-example scores for system A.
        system_b_scores: Per-example scores for system B.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Dict with 'p_value', 'mean_diff', 'ci_lower', 'ci_upper'.
    """
    rng = np.random.RandomState(seed)
    n = len(system_a_scores)
    assert n == len(system_b_scores), "Score lists must have same length"

    diffs = np.array(system_a_scores) - np.array(system_b_scores)
    observed_diff = np.mean(diffs)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_diff = np.mean(diffs[indices])
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    p_value = np.mean(bootstrap_diffs <= 0) if observed_diff > 0 else np.mean(bootstrap_diffs >= 0)

    return {
        "p_value": float(p_value),
        "mean_diff": float(observed_diff),
        "ci_lower": float(np.percentile(bootstrap_diffs, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_diffs, 97.5)),
        "significant": p_value < 0.05,
    }
