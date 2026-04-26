#!/usr/bin/env python
"""Train/distill gap analysis model (70B → 8B)."""

import argparse
import logging

from compliance_nlp.optimization.distillation import ComplianceDistiller
from compliance_nlp.utils.config import load_config
from compliance_nlp.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Distill gap analysis model")
    parser.add_argument("--config", type=str, default="configs/distillation.yaml")
    parser.add_argument("--teacher", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--student", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data-dir", type=str, default="data/compliance_instructions")
    parser.add_argument("--output-dir", type=str, default="outputs/distilled_model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    distiller = ComplianceDistiller(
        teacher_path=args.teacher,
        student_path=args.student,
        gamma=config.get("distillation", {}).get("gamma", 0.5),
    )

    results = distiller.distill(
        train_data_path=args.data_dir,
        output_dir=args.output_dir,
        epochs=config.get("training", {}).get("epochs", 3),
    )

    log.info(f"Distillation complete: {results}")


if __name__ == "__main__":
    main()
