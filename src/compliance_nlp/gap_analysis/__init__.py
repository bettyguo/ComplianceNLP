"""Gap analysis pipeline: alignment, grounding, severity, and reporting."""

from compliance_nlp.gap_analysis.alignment import ObligationPolicyAligner, AlignmentResult
from compliance_nlp.gap_analysis.grounding import GroundingVerifier, GroundingResult
from compliance_nlp.gap_analysis.severity import Severity, compute_severity
from compliance_nlp.gap_analysis.report import GapFinding, ComplianceGapReport, compile_report

__all__ = [
    "ObligationPolicyAligner",
    "AlignmentResult",
    "GroundingVerifier",
    "GroundingResult",
    "Severity",
    "compute_severity",
    "GapFinding",
    "ComplianceGapReport",
    "compile_report",
]
