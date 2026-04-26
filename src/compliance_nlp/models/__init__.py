"""Model architectures for ComplianceNLP."""

from compliance_nlp.models.legal_bert import CRFLayer, MultiTaskLegalBERT
from compliance_nlp.models.gap_generator import GapAnalysisGenerator, GapReport

__all__ = [
    "CRFLayer",
    "MultiTaskLegalBERT",
    "GapAnalysisGenerator",
    "GapReport",
]
