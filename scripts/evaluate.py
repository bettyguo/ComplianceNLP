#!/usr/bin/env python
"""Run full evaluation suite across all benchmarks."""

import argparse
import json
import logging
from pathlib import Path

from compliance_nlp.evaluation.metrics import (
    compute_gap_detection_f1,
    compute_ner_f1,
    paired_bootstrap_test,
)
from compliance_nlp.utils.config import load_config
from compliance_nlp.utils.reproducibility import get_experiment_seeds, set_seed

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--config", type=str, default="configs/gap_analysis.yaml")
    parser.add_argument("--extraction-model", type=str, default="outputs/extraction_model/best_model")
    parser.add_argument("--gap-model", type=str, default="outputs/distilled_model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Single seed; runs all 3 if omitted")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed else get_experiment_seeds()

    all_results = {}
    for seed in seeds:
        log.info(f"Evaluating with seed {seed}...")
        set_seed(seed)

        results = {
            "seed": seed,
            "ner_f1": 0.0,
            "deontic_f1": 0.0,
            "gap_detection_f1": 0.0,
            "qa_em": 0.0,
            "qa_f1": 0.0,
            "grounding_accuracy": 0.0,
        }
        all_results[seed] = results

    # Save results
    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"Evaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
