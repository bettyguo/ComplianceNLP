"""Configuration management using OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval pipeline."""

    alpha: float = 0.7  # Dense/sparse weight
    beta: float = 0.3  # KG re-ranking weight
    top_k: int = 5
    dense_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""


@dataclass
class ExtractionConfig:
    """Configuration for multi-task obligation extraction."""

    backbone: str = "nlpaueb/legal-bert-base-uncased"
    num_entity_types: int = 23
    deontic_classes: int = 4  # Obligation, Permission, Prohibition, Recommendation
    max_seq_length: int = 512
    lambda_ner: float = 0.4
    lambda_deontic: float = 0.3
    lambda_xref: float = 0.3
    dropout: float = 0.1
    crf_enabled: bool = True


@dataclass
class GapAnalysisConfig:
    """Configuration for compliance gap analysis."""

    generator_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    delta_eval: float = 0.6  # Evaluation threshold
    delta_deploy: float = 0.45  # Recall-optimized deployment threshold
    grounding_threshold: float = 0.85  # MiniCheck confidence threshold
    minicheck_model: str = "lytang/MiniCheck-DeBERTa-v3-large"
    max_generation_length: int = 512
    num_beams: int = 1  # Greedy for latency


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation (70B → 8B)."""

    teacher_model: str = "meta-llama/Meta-Llama-3-70B-Instruct"
    student_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    gamma: float = 0.5  # KD/SFT balance weight
    num_training_samples: int = 15000
    learning_rate: float = 1e-5
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8


@dataclass
class MedusaConfig:
    """Configuration for Medusa speculative decoding heads."""

    num_heads: int = 3
    num_training_tokens: int = 2_100_000
    learning_rate: float = 3e-4
    epochs: int = 2
    hidden_dim: int = 4096


@dataclass
class ServingConfig:
    """Configuration for production serving."""

    host: str = "0.0.0.0"
    port: int = 8080
    max_batch_size: int = 32
    max_sequence_length: int = 512
    model_warmup: bool = True
    prometheus_enabled: bool = True
    cors_enabled: bool = True


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    scheduler: str = "cosine_with_warmup"
    epochs: int = 10
    batch_size: int = 32
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42


@dataclass
class ComplianceNLPConfig:
    """Top-level configuration for the ComplianceNLP system."""

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    gap_analysis: GapAnalysisConfig = field(default_factory=GapAnalysisConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    medusa: MedusaConfig = field(default_factory=MedusaConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Loading
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: str | Path) -> DictConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        OmegaConf DictConfig object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def load_config_with_defaults(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ComplianceNLPConfig:
    """Load configuration with defaults and optional overrides.

    Args:
        config_path: Optional path to YAML config file.
        overrides: Optional dict of overrides to apply.

    Returns:
        Structured ComplianceNLPConfig object.
    """
    base = OmegaConf.structured(ComplianceNLPConfig)

    if config_path is not None:
        file_config = load_config(config_path)
        base = OmegaConf.merge(base, file_config)

    if overrides:
        override_config = OmegaConf.create(overrides)
        base = OmegaConf.merge(base, override_config)

    return OmegaConf.to_object(base)
