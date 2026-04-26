"""Compliance gap report generation and formatting."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from compliance_nlp.gap_analysis.severity import Severity

log = logging.getLogger(__name__)


@dataclass
class GapFinding:
    """A single compliance gap finding in a report."""

    obligation_entity: str
    obligation_action: str
    obligation_modality: str
    source_provision: str
    cross_references: list[str]
    matched_policy_section: str
    alignment_score: float
    classification: str  # Compliant, Partial Gap, Full Gap
    severity: str
    gap_description: str
    recommended_action: str
    grounding_confidence: float
    requires_human_review: bool


@dataclass
class ComplianceGapReport:
    """Full compliance gap analysis report."""

    report_id: str
    generated_at: str
    regulatory_document: str
    framework: str
    total_obligations: int
    findings: list[GapFinding] = field(default_factory=list)
    compliant_count: int = 0
    partial_gap_count: int = 0
    full_gap_count: int = 0
    critical_count: int = 0

    def to_json(self, output_path: str | Path) -> None:
        """Save report as JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        log.info(f"Report saved to {output_path}")

    def summary(self) -> str:
        """Generate a human-readable summary."""
        return (
            f"Compliance Gap Report ({self.report_id})\n"
            f"{'=' * 50}\n"
            f"Document: {self.regulatory_document}\n"
            f"Framework: {self.framework}\n"
            f"Generated: {self.generated_at}\n"
            f"\nFindings Summary:\n"
            f"  Total Obligations: {self.total_obligations}\n"
            f"  Compliant: {self.compliant_count}\n"
            f"  Partial Gaps: {self.partial_gap_count}\n"
            f"  Full Gaps: {self.full_gap_count}\n"
            f"  Critical Severity: {self.critical_count}\n"
        )


def compile_report(
    findings: list[GapFinding],
    document_name: str,
    framework: str,
) -> ComplianceGapReport:
    """Compile individual findings into a structured report.

    Args:
        findings: List of GapFinding objects.
        document_name: Name of the analyzed regulatory document.
        framework: Regulatory framework (SEC, MiFID II, Basel III).

    Returns:
        ComplianceGapReport.
    """
    report = ComplianceGapReport(
        report_id=f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        generated_at=datetime.now().isoformat(),
        regulatory_document=document_name,
        framework=framework,
        total_obligations=len(findings),
        findings=findings,
        compliant_count=sum(1 for f in findings if f.classification == "Compliant"),
        partial_gap_count=sum(1 for f in findings if f.classification == "Partial Gap"),
        full_gap_count=sum(1 for f in findings if f.classification == "Full Gap"),
        critical_count=sum(1 for f in findings if f.severity == "Critical"),
    )
    return report
