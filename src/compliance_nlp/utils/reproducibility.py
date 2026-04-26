"""Reproducibility utilities for deterministic training and evaluation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Args:
        seed: Random seed value.
        deterministic: If True, enable CUDA deterministic algorithms.
            Set to False in production for better performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def get_experiment_seeds() -> list[int]:
    """Return the standard set of experiment seeds for multi-seed evaluation."""
    return [42, 123, 456]
