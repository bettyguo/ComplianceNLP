"""Multi-task training orchestrator for obligation extraction.

Coordinates joint training of NER, deontic classification, and
cross-reference resolution over a shared LEGAL-BERT encoder.

Combined loss: L = λ₁·L_NER + λ₂·L_deontic + λ₃·L_xref
with λ₁=0.4, λ₂=0.3, λ₃=0.3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from compliance_nlp.data.datasets import RegObligationDataset
from compliance_nlp.models.legal_bert import MultiTaskLegalBERT
from compliance_nlp.utils.reproducibility import set_seed

log = logging.getLogger(__name__)


class MultiTaskTrainer:
    """Trainer for multi-task LEGAL-BERT obligation extraction.

    Training configuration:
    - 8,742 annotated sentences (SEC: 3,211; MiFID II: 2,987; Basel III: 2,544)
    - 3 compliance expert annotators (κ=0.84)
    - Augmented with ObliQA and COLING 2025 shared task data
    """

    def __init__(
        self,
        model: MultiTaskLegalBERT,
        tokenizer: AutoTokenizer,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda") if fp16 and self.device.type == "cuda" else None

    def train(
        self,
        train_dataset: RegObligationDataset,
        val_dataset: Optional[RegObligationDataset] = None,
        epochs: int = 10,
        batch_size: int = 32,
        output_dir: str = "outputs/extraction_model",
        seed: int = 42,
    ) -> dict:
        """Train the multi-task extraction model.

        Args:
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            output_dir: Directory to save checkpoints.
            seed: Random seed.

        Returns:
            Dict with training history and best metrics.
        """
        set_seed(seed)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.warmup_steps, total_steps
        )

        best_val_f1 = 0.0
        history = {"train_loss": [], "val_f1": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                if self.fp16 and self.scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(**batch)
                        loss = outputs["loss"]
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

                if (step + 1) % 100 == 0:
                    log.info(
                        f"Epoch {epoch+1}/{epochs}, Step {step+1}, "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = epoch_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            log.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

            # Validation
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset, batch_size)
                val_f1 = val_metrics.get("ner_f1", 0.0)
                history["val_f1"].append(val_f1)
                log.info(f"Validation NER F1: {val_f1:.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self._save_checkpoint(output_path / "best_model")
                    log.info(f"New best model saved (F1: {val_f1:.4f})")

        # Save final model
        self._save_checkpoint(output_path / "final_model")

        return {
            "history": history,
            "best_val_f1": best_val_f1,
        }

    @torch.no_grad()
    def evaluate(
        self, dataset: RegObligationDataset, batch_size: int = 32
    ) -> dict:
        """Evaluate the model on a dataset.

        Args:
            dataset: Evaluation dataset.
            batch_size: Evaluation batch size.

        Returns:
            Dict with NER F1, Deontic accuracy, and XRef accuracy.
        """
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_ner_preds, all_ner_labels = [], []
        all_deontic_preds, all_deontic_labels = [], []

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = self.model(**batch)

            # NER predictions
            if "ner_logits" in outputs:
                ner_preds = outputs["ner_logits"].argmax(dim=-1)
                all_ner_preds.extend(ner_preds.cpu().tolist())
                if "ner_labels" in batch:
                    all_ner_labels.extend(batch["ner_labels"].cpu().tolist())

            # Deontic predictions
            if "deontic_logits" in outputs:
                deontic_preds = outputs["deontic_logits"].argmax(dim=-1)
                all_deontic_preds.extend(deontic_preds.cpu().tolist())
                if "deontic_label" in batch:
                    all_deontic_labels.extend(batch["deontic_label"].cpu().tolist())

        metrics = {}

        # Compute NER F1 (micro, excluding O tags)
        if all_ner_preds and all_ner_labels:
            from compliance_nlp.evaluation.metrics import compute_ner_f1

            metrics["ner_f1"] = compute_ner_f1(all_ner_preds, all_ner_labels)

        # Compute deontic accuracy
        if all_deontic_preds and all_deontic_labels:
            correct = sum(
                p == l for p, l in zip(all_deontic_preds, all_deontic_labels)
            )
            metrics["deontic_accuracy"] = correct / len(all_deontic_preds)

        return metrics

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "pytorch_model.bin")
        self.tokenizer.save_pretrained(path)
        log.info(f"Checkpoint saved to {path}")
