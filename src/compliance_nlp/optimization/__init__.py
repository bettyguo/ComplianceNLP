"""Production optimizations: knowledge distillation and Medusa speculative decoding."""

from compliance_nlp.optimization.distillation import ComplianceDistiller
from compliance_nlp.optimization.medusa_heads import MedusaHead, MedusaDecoder

__all__ = [
    "ComplianceDistiller",
    "MedusaHead",
    "MedusaDecoder",
]
