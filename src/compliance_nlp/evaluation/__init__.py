"""Evaluation metrics and latency benchmarking."""

from compliance_nlp.evaluation.metrics import (
    compute_ner_f1,
    compute_gap_detection_f1,
    paired_bootstrap_test,
)
from compliance_nlp.evaluation.latency_benchmark import (
    LatencyBenchmarkResult,
    benchmark_latency,
    benchmark_scaling,
)

__all__ = [
    "compute_ner_f1",
    "compute_gap_detection_f1",
    "paired_bootstrap_test",
    "LatencyBenchmarkResult",
    "benchmark_latency",
    "benchmark_scaling",
]
