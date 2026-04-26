#!/usr/bin/env python
"""Train multi-task obligation extraction model."""

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer

from compliance_nlp.data.datasets import RegObligationDataset
from compliance_nlp.extraction.multitask import MultiTaskTrainer
from compliance_nlp.models.legal_bert import MultiTaskLegalBERT
from compliance_nlp.utils.config import load_config
from compliance_nlp.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train multi-task extraction model")
    parser.add_argument("--config", type=str, default="configs/extraction.yaml")
    parser.add_argument("--data-dir", type=str, default="data/regobligation")
    parser.add_argument("--output-dir", type=str, default="outputs/extraction_model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    backbone = config.get("model", {}).get("backbone", "nlpaueb/legal-bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = MultiTaskLegalBERT(backbone=backbone)

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = RegObligationDataset(args.data_dir, split="train", tokenizer=tokenizer)
    val_dataset = RegObligationDataset(args.data_dir, split="val", tokenizer=tokenizer)

    trainer = MultiTaskTrainer(model=model, tokenizer=tokenizer)
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=config.get("training", {}).get("epochs", 10),
        batch_size=config.get("training", {}).get("batch_size", 32),
        output_dir=args.output_dir,
        seed=args.seed,
    )

    log.info(f"Training complete. Best val F1: {results['best_val_f1']:.4f}")


if __name__ == "__main__":
    main()
