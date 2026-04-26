"""Tests for compliance gap analysis."""

import pytest
from compliance_nlp.gap_analysis.severity import Severity, compute_severity
from compliance_nlp.gap_analysis.alignment import AlignmentResult
from compliance_nlp.gap_analysis.report import GapFinding, compile_report


class TestSeverity:
    def test_critical_obligation_full_gap(self):
        severity = compute_severity("Obligation", "Full Gap")
        assert severity == Severity.CRITICAL

    def test_minor_recommendation_partial(self):
        severity = compute_severity("Recommendation", "Partial Gap")
        assert severity == Severity.MINOR

    def test_escalation_with_enforcement(self):
        severity = compute_severity("Permission", "Full Gap", has_enforcement_history=True)
        assert severity == Severity.MAJOR  # Escalated from MODERATE

    def test_compliant_returns_minor(self):
        severity = compute_severity("Obligation", "Compliant")
        assert severity == Severity.MINOR


class TestReport:
    def test_compile_empty_report(self):
        report = compile_report([], "test_doc.pdf", "SEC")
        assert report.total_obligations == 0
        assert report.compliant_count == 0

    def test_compile_with_findings(self):
        findings = [
            GapFinding(
                obligation_entity="Bank",
                obligation_action="Report quarterly",
                obligation_modality="Obligation",
                source_provision="SEC Rule 10b-5",
                cross_references=[],
                matched_policy_section="§4.1",
                alignment_score=0.75,
                classification="Compliant",
                severity="N/A",
                gap_description="",
                recommended_action="",
                grounding_confidence=0.95,
                requires_human_review=False,
            ),
            GapFinding(
                obligation_entity="Bank",
                obligation_action="Disclose risks",
                obligation_modality="Obligation",
                source_provision="Basel III Para 50",
                cross_references=["BCBS d295"],
                matched_policy_section="§12.2",
                alignment_score=0.31,
                classification="Full Gap",
                severity="Critical",
                gap_description="Policy missing HQLA criteria",
                recommended_action="Revise §12.2",
                grounding_confidence=0.87,
                requires_human_review=True,
            ),
        ]
        report = compile_report(findings, "test_doc.pdf", "Basel III")
        assert report.total_obligations == 2
        assert report.compliant_count == 1
        assert report.full_gap_count == 1
        assert report.critical_count == 1
