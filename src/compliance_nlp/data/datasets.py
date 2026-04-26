"""Dataset loaders for RegObligation and GapBench benchmarks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RegulatoryEntity:
    """A named entity in regulatory text."""

    text: str
    label: str  # One of 23 entity types
    start: int
    end: int
    confidence: float = 0.0


@dataclass
class Obligation:
    """A structured regulatory obligation extracted from text."""

    entity: str
    action: str
    modality: str  # Obligation, Permission, Prohibition, Recommendation
    condition: str
    source_provision: str
    cross_references: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class GapAnnotation:
    """A compliance gap annotation (obligation-policy pair)."""

    obligation: Obligation
    policy_text: str
    policy_section: str
    label: str  # Compliant, Partial Gap, Full Gap
    severity: str = ""  # Minor, Moderate, Major, Critical
    gap_description: str = ""


@dataclass
class ExtractionSample:
    """A single sample for multi-task extraction training/evaluation."""

    text: str
    entities: list[RegulatoryEntity]
    deontic_label: str
    cross_references: list[str]
    framework: str  # SEC, MiFID II, Basel III


# ─────────────────────────────────────────────────────────────────────────────
# Entity Types
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "O",  # Outside
    "B-REGULATORY_BODY", "I-REGULATORY_BODY",
    "B-REPORTING_ENTITY", "I-REPORTING_ENTITY",
    "B-EFFECTIVE_DATE", "I-EFFECTIVE_DATE",
    "B-THRESHOLD_VALUE", "I-THRESHOLD_VALUE",
    "B-FINANCIAL_INSTRUMENT", "I-FINANCIAL_INSTRUMENT",
    "B-OBLIGATION_ACTION", "I-OBLIGATION_ACTION",
    "B-COMPLIANCE_PERIOD", "I-COMPLIANCE_PERIOD",
    "B-JURISDICTION", "I-JURISDICTION",
    "B-PENALTY_AMOUNT", "I-PENALTY_AMOUNT",
    "B-RISK_CATEGORY", "I-RISK_CATEGORY",
    "B-CAPITAL_REQUIREMENT", "I-CAPITAL_REQUIREMENT",
    "B-DISCLOSURE_ITEM", "I-DISCLOSURE_ITEM",
    "B-FILING_TYPE", "I-FILING_TYPE",
    "B-COUNTERPARTY", "I-COUNTERPARTY",
    "B-SUPERVISORY_AUTHORITY", "I-SUPERVISORY_AUTHORITY",
    "B-MARKET_TYPE", "I-MARKET_TYPE",
    "B-TRANSACTION_TYPE", "I-TRANSACTION_TYPE",
    "B-GOVERNANCE_ROLE", "I-GOVERNANCE_ROLE",
    "B-AUDIT_REQUIREMENT", "I-AUDIT_REQUIREMENT",
    "B-REPORTING_FREQUENCY", "I-REPORTING_FREQUENCY",
    "B-LEGAL_REFERENCE", "I-LEGAL_REFERENCE",
    "B-CROSS_BORDER_PROVISION", "I-CROSS_BORDER_PROVISION",
    "B-EXEMPTION_CLAUSE", "I-EXEMPTION_CLAUSE",
]

DEONTIC_LABELS = ["Obligation", "Permission", "Prohibition", "Recommendation"]

LABEL2ID = {label: idx for idx, label in enumerate(ENTITY_TYPES)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Classes
# ─────────────────────────────────────────────────────────────────────────────


class RegObligationDataset(Dataset):
    """Dataset for RegObligation benchmark (1,847 regulatory sentences).

    Supports multi-task training with NER, deontic classification,
    and cross-reference resolution annotations.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        tokenizer=None,
        max_length: int = 512,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[ExtractionSample] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load and parse RegObligation data from JSON."""
        filepath = self.data_dir / f"{self.split}.json"
        if not filepath.exists():
            log.warning(f"Data file not found: {filepath}")
            return

        with open(filepath) as f:
            data = json.load(f)

        for item in data:
            entities = [
                RegulatoryEntity(
                    text=e["text"],
                    label=e["label"],
                    start=e["start"],
                    end=e["end"],
                )
                for e in item.get("entities", [])
            ]
            sample = ExtractionSample(
                text=item["text"],
                entities=entities,
                deontic_label=item.get("deontic_label", "Obligation"),
                cross_references=item.get("cross_references", []),
                framework=item.get("framework", "SEC"),
            )
            self.samples.append(sample)

        log.info(f"Loaded {len(self.samples)} samples from {filepath}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        if self.tokenizer is None:
            return {
                "text": sample.text,
                "entities": [(e.start, e.end, e.label) for e in sample.entities],
                "deontic_label": DEONTIC_LABELS.index(sample.deontic_label),
                "cross_references": sample.cross_references,
                "framework": sample.framework,
            }

        encoding = self.tokenizer(
            sample.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Align NER labels to tokenized input
        ner_labels = self._align_labels(
            encoding["offset_mapping"][0], sample.entities
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ner_labels": torch.tensor(ner_labels, dtype=torch.long),
            "deontic_label": torch.tensor(
                DEONTIC_LABELS.index(sample.deontic_label), dtype=torch.long
            ),
            "framework": sample.framework,
        }

    def _align_labels(
        self, offsets: torch.Tensor, entities: list[RegulatoryEntity]
    ) -> list[int]:
        """Align entity labels to subword token positions."""
        labels = [LABEL2ID["O"]] * len(offsets)

        for entity in entities:
            for idx, (start, end) in enumerate(offsets.tolist()):
                if start == end:
                    continue
                if start >= entity.start and end <= entity.end:
                    if start == entity.start:
                        labels[idx] = LABEL2ID.get(f"B-{entity.label}", 0)
                    else:
                        labels[idx] = LABEL2ID.get(f"I-{entity.label}", 0)

        return labels


class GapBenchDataset(Dataset):
    """Dataset for GapBench benchmark (423 obligation-policy pairs).

    Supports compliance gap detection evaluation with three-class
    classification: Compliant, Partial Gap, Full Gap.
    """

    GAP_LABELS = ["Compliant", "Partial Gap", "Full Gap"]

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "test",
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.annotations: list[GapAnnotation] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load GapBench annotations from JSON."""
        filepath = self.data_dir / f"gapbench_{self.split}.json"
        if not filepath.exists():
            log.warning(f"Data file not found: {filepath}")
            return

        with open(filepath) as f:
            data = json.load(f)

        for item in data:
            obligation = Obligation(
                entity=item["obligation"]["entity"],
                action=item["obligation"]["action"],
                modality=item["obligation"]["modality"],
                condition=item["obligation"].get("condition", ""),
                source_provision=item["obligation"]["source_provision"],
                cross_references=item["obligation"].get("cross_references", []),
            )
            annotation = GapAnnotation(
                obligation=obligation,
                policy_text=item["policy_text"],
                policy_section=item.get("policy_section", ""),
                label=item["label"],
                severity=item.get("severity", ""),
                gap_description=item.get("gap_description", ""),
            )
            self.annotations.append(annotation)

        log.info(f"Loaded {len(self.annotations)} GapBench annotations")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        return {
            "obligation_text": (
                f"Entity: {ann.obligation.entity}\n"
                f"Action: {ann.obligation.action}\n"
                f"Modality: {ann.obligation.modality}\n"
                f"Condition: {ann.obligation.condition}\n"
                f"Source: {ann.obligation.source_provision}"
            ),
            "policy_text": ann.policy_text,
            "policy_section": ann.policy_section,
            "label": self.GAP_LABELS.index(ann.label),
            "label_str": ann.label,
            "severity": ann.severity,
        }

    def iter_by_framework(self, framework: str) -> Iterator[dict]:
        """Iterate over samples filtered by regulatory framework."""
        for idx in range(len(self)):
            item = self[idx]
            if framework.lower() in self.annotations[idx].obligation.source_provision.lower():
                yield item
