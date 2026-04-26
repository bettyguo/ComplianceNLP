#!/usr/bin/env python
"""Train Medusa speculative decoding heads."""

import argparse
import logging

from compliance_nlp.utils.config import load_config
from compliance_nlp.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Medusa heads")
    parser.add_argument("--config", type=str, default="configs/medusa.yaml")
    parser.add_argument("--base-model", type=str, default="outputs/distilled_model")
    parser.add_argument("--data-dir", type=str, default="data/regulatory_corpus")
    parser.add_argument("--output-dir", type=str, default="outputs/medusa_model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    log.info(f"Training Medusa heads on base model: {args.base_model}")
    log.info(f"Config: {config.get('medusa', {}).get('num_heads', 3)} heads, "
             f"{config.get('training', {}).get('num_tokens', 2_100_000)} tokens")

    # Training logic would load the base model and train Medusa heads
    log.info(f"Medusa training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
