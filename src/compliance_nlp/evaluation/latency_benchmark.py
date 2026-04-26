"""Production latency benchmarking utilities.

Measures throughput (QPS), latency percentiles (P50, P90, P99),
memory usage, and GPU utilization for production deployment validation.
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class LatencyBenchmarkResult:
    """Results from latency benchmarking."""

    latency_p50: float  # ms
    latency_p90: float
    latency_p99: float
    latency_mean: float
    latency_std: float
    throughput_qps: float
    peak_memory_mb: float
    batch_size: int
    num_samples: int
    warmup_samples: int

    def __str__(self) -> str:
        return (
            f"\nLatency Benchmark Results\n"
            f"{'=' * 55}\n"
            f"  Latency (ms):\n"
            f"    P50:  {self.latency_p50:.2f}\n"
            f"    P90:  {self.latency_p90:.2f}\n"
            f"    P99:  {self.latency_p99:.2f}\n"
            f"    Mean: {self.latency_mean:.2f} +/- {self.latency_std:.2f}\n"
            f"\n"
            f"  Throughput:    {self.throughput_qps:.1f} QPS\n"
            f"  Peak Memory:  {self.peak_memory_mb:.1f} MB\n"
            f"\n"
            f"  Config:\n"
            f"    Batch Size:  {self.batch_size}\n"
            f"    Samples:     {self.num_samples}\n"
            f"    Warmup:      {self.warmup_samples}\n"
            f"{'=' * 55}\n"
        )


def benchmark_latency(
    predict_fn: Callable,
    inputs: list,
    batch_size: int = 1,
    num_samples: int = 100,
    warmup_samples: int = 10,
) -> LatencyBenchmarkResult:
    """Benchmark inference latency.

    Args:
        predict_fn: Function to benchmark (takes list of inputs).
        inputs: Sample inputs (will be cycled).
        batch_size: Batch size for inference.
        num_samples: Number of measurement samples.
        warmup_samples: Number of warmup iterations.

    Returns:
        LatencyBenchmarkResult with all metrics.
    """
    # Prepare batches
    batches = []
    for i in range(0, max(len(inputs), batch_size), batch_size):
        batch = inputs[i : i + batch_size]
        if len(batch) < batch_size:
            batch = batch + inputs[: batch_size - len(batch)]
        batches.append(batch)
    if not batches:
        batches = [inputs[:batch_size]]

    # Warmup
    log.info(f"Warming up with {warmup_samples} samples...")
    for i in range(warmup_samples):
        batch = batches[i % len(batches)]
        _ = predict_fn(batch)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Benchmark
    log.info(f"Benchmarking {num_samples} samples (batch_size={batch_size})...")
    latencies = []

    for i in tqdm(range(num_samples), desc="Benchmarking"):
        batch = batches[i % len(batches)]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = predict_fn(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    # Calculate statistics
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    # Memory
    peak_memory = 0.0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Throughput
    total_time_sec = sum(latencies) / 1000
    throughput = (num_samples * batch_size) / total_time_sec

    return LatencyBenchmarkResult(
        latency_p50=latencies_sorted[int(0.50 * n)],
        latency_p90=latencies_sorted[int(0.90 * n)],
        latency_p99=latencies_sorted[min(int(0.99 * n), n - 1)],
        latency_mean=statistics.mean(latencies),
        latency_std=statistics.stdev(latencies) if n > 1 else 0,
        throughput_qps=throughput,
        peak_memory_mb=peak_memory,
        batch_size=batch_size,
        num_samples=num_samples,
        warmup_samples=warmup_samples,
    )


def benchmark_scaling(
    predict_fn: Callable,
    inputs: list,
    batch_sizes: list[int] = [1, 2, 4, 8, 16, 32],
    num_samples: int = 50,
) -> dict[int, LatencyBenchmarkResult]:
    """Benchmark latency across different batch sizes.

    Returns dict mapping batch_size -> results.
    """
    results = {}
    for bs in batch_sizes:
        log.info(f"\nBenchmarking batch_size={bs}")
        results[bs] = benchmark_latency(
            predict_fn=predict_fn,
            inputs=inputs,
            batch_size=bs,
            num_samples=num_samples,
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run latency benchmark")
    parser.add_argument("--model", type=str, default="", help="Path to model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    # Example: benchmark with dummy function
    def dummy_predict(batch):
        time.sleep(0.001)
        return batch

    sample_inputs = [f"Sample regulatory text {i}" for i in range(100)]
    result = benchmark_latency(
        dummy_predict, sample_inputs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        warmup_samples=args.warmup,
    )
    print(result)
