"""Multi-task LEGAL-BERT encoder for regulatory obligation extraction.

Jointly trains three task heads over a shared LEGAL-BERT encoder:
  (a) CRF-based NER for 23 regulatory entity types
  (b) Sentence-level deontic modality classification
  (c) Span-pair cross-reference resolution

Combined loss: L = λ₁·L_NER + λ₂·L_deontic + λ₃·L_xref
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from compliance_nlp.data.datasets import ENTITY_TYPES, DEONTIC_LABELS

log = logging.getLogger(__name__)


class CRFLayer(nn.Module):
    """Conditional Random Field layer for sequence labeling."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            emissions: (batch, seq_len, num_tags) emission scores.
            tags: (batch, seq_len) gold tag indices.
            mask: (batch, seq_len) boolean attention mask.

        Returns:
            Scalar negative log-likelihood loss.
        """
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        return (forward_score - gold_score).mean()

    def _forward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            score += self.transitions[tags[:, i - 1], tags[:, i]] * mask[:, i]
            score += (
                emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
                * mask[:, i]
            )

        last_tag_indices = mask.sum(dim=1).long() - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding to find best tag sequence.

        Args:
            emissions: (batch, seq_len, num_tags) emission scores.
            mask: (batch, seq_len) boolean mask.

        Returns:
            List of best tag sequences for each sample in batch.
        """
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        score += self.end_transitions
        _, best_last_tags = score.max(dim=1)

        best_paths = [best_last_tags.tolist()]
        for hist in reversed(history):
            best_last_tags = hist.gather(1, best_last_tags.unsqueeze(1)).squeeze(1)
            best_paths.insert(0, best_last_tags.tolist())

        # Transpose to get per-sample paths
        return [
            [best_paths[t][b] for t in range(seq_len)]
            for b in range(batch_size)
        ]


class MultiTaskLegalBERT(nn.Module):
    """Multi-task LEGAL-BERT for regulatory obligation extraction.

    Architecture:
        Shared LEGAL-BERT encoder
        ├── NER Head (CRF) → 23 entity types
        ├── Deontic Head (Linear) → 4 modality classes
        └── Cross-Ref Head (Bilinear) → span-pair linking
    """

    def __init__(
        self,
        backbone: str = "nlpaueb/legal-bert-base-uncased",
        num_entity_types: int = len(ENTITY_TYPES),
        num_deontic_classes: int = len(DEONTIC_LABELS),
        dropout: float = 0.1,
        lambda_ner: float = 0.4,
        lambda_deontic: float = 0.3,
        lambda_xref: float = 0.3,
        use_crf: bool = True,
    ):
        super().__init__()
        self.lambda_ner = lambda_ner
        self.lambda_deontic = lambda_deontic
        self.lambda_xref = lambda_xref
        self.use_crf = use_crf

        # Shared encoder
        config = AutoConfig.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone, config=config)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # Task head: NER (CRF-based)
        self.ner_classifier = nn.Linear(hidden_size, num_entity_types)
        if use_crf:
            self.crf = CRFLayer(num_entity_types)

        # Task head: Deontic modality classification
        self.deontic_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_deontic_classes),
        )

        # Task head: Cross-reference resolution (bilinear)
        self.xref_bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
        self.xref_loss_fn = nn.BCEWithLogitsLoss()

        # Standard NER loss (when not using CRF)
        self.ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.deontic_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ner_labels: Optional[torch.Tensor] = None,
        deontic_label: Optional[torch.Tensor] = None,
        xref_source_spans: Optional[torch.Tensor] = None,
        xref_target_spans: Optional[torch.Tensor] = None,
        xref_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through shared encoder and all task heads.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) attention mask.
            ner_labels: (batch, seq_len) NER tag indices.
            deontic_label: (batch,) deontic class indices.
            xref_source_spans: (batch, num_xrefs, 2) source span start/end.
            xref_target_spans: (batch, num_xrefs, 2) target span start/end.
            xref_labels: (batch, num_xrefs) binary cross-ref labels.

        Returns:
            Dict with 'loss', 'ner_logits', 'deontic_logits', 'xref_logits'.
        """
        # Shared encoding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # (B, L, H)
        pooled_output = sequence_output[:, 0, :]  # [CLS] token

        result = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)

        # ── NER Head ──
        ner_emissions = self.ner_classifier(sequence_output)  # (B, L, num_tags)
        result["ner_logits"] = ner_emissions

        if ner_labels is not None:
            if self.use_crf:
                ner_loss = self.crf(ner_emissions, ner_labels, attention_mask.float())
            else:
                ner_loss = self.ner_loss_fn(
                    ner_emissions.view(-1, ner_emissions.size(-1)),
                    ner_labels.view(-1),
                )
            total_loss = total_loss + self.lambda_ner * ner_loss
            result["ner_loss"] = ner_loss

        # ── Deontic Head ──
        deontic_logits = self.deontic_classifier(pooled_output)  # (B, num_deontic)
        result["deontic_logits"] = deontic_logits

        if deontic_label is not None:
            deontic_loss = self.deontic_loss_fn(deontic_logits, deontic_label)
            total_loss = total_loss + self.lambda_deontic * deontic_loss
            result["deontic_loss"] = deontic_loss

        # ── Cross-Reference Head ──
        if xref_source_spans is not None and xref_target_spans is not None:
            source_repr = self._get_span_repr(sequence_output, xref_source_spans)
            target_repr = self._get_span_repr(sequence_output, xref_target_spans)
            xref_logits = self.xref_bilinear(source_repr, target_repr).squeeze(-1)
            result["xref_logits"] = xref_logits

            if xref_labels is not None:
                xref_loss = self.xref_loss_fn(xref_logits, xref_labels.float())
                total_loss = total_loss + self.lambda_xref * xref_loss
                result["xref_loss"] = xref_loss

        result["loss"] = total_loss
        return result

    def _get_span_repr(
        self, sequence_output: torch.Tensor, spans: torch.Tensor
    ) -> torch.Tensor:
        """Extract span representations by mean-pooling.

        Args:
            sequence_output: (batch, seq_len, hidden) encoder outputs.
            spans: (batch, num_spans, 2) start/end indices.

        Returns:
            (batch, num_spans, hidden) span representations.
        """
        batch_size, num_spans, _ = spans.shape
        hidden_size = sequence_output.size(-1)
        span_reprs = torch.zeros(batch_size, num_spans, hidden_size, device=spans.device)

        for b in range(batch_size):
            for s in range(num_spans):
                start, end = spans[b, s, 0].item(), spans[b, s, 1].item()
                if start < end and end <= sequence_output.size(1):
                    span_reprs[b, s] = sequence_output[b, start:end].mean(dim=0)

        return span_reprs

    def predict_ner(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[list[int]]:
        """Predict NER tags using Viterbi decoding (CRF) or argmax."""
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            emissions = self.ner_classifier(outputs.last_hidden_state)

            if self.use_crf:
                return self.crf.decode(emissions, attention_mask)
            else:
                return emissions.argmax(dim=-1).tolist()

    def predict(self, texts: list[str]) -> list[dict]:
        """High-level prediction interface for serving.

        Args:
            texts: List of regulatory text strings.

        Returns:
            List of prediction dicts with NER tags and deontic labels.
        """
        raise NotImplementedError(
            "Use predict_ner/predict_deontic with a tokenizer for full predictions."
        )

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> MultiTaskLegalBERT:
        """Load a pretrained multi-task model.

        Args:
            path: Path to saved model directory or HuggingFace model ID.

        Returns:
            Loaded MultiTaskLegalBERT model.
        """
        import json
        from pathlib import Path as P

        model_dir = P(path)
        if model_dir.exists() and (model_dir / "config.json").exists():
            with open(model_dir / "config.json") as f:
                config = json.load(f)
            model = cls(**config)
            state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            model = cls(backbone=path, **kwargs)

        return model
