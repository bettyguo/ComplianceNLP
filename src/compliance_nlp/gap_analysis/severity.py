"""Gap severity scoring based on obligation modality, entity type, and enforcement history."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Severity(IntEnum):
    """Gap severity levels, ordered by escalation urgency."""
    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    CRITICAL = 4


# Severity rules based on obligation modality and gap type
SEVERITY_RULES = {
    ("Obligation", "Full Gap"): Severity.CRITICAL,
    ("Obligation", "Partial Gap"): Severity.MAJOR,
    ("Prohibition", "Full Gap"): Severity.CRITICAL,
    ("Prohibition", "Partial Gap"): Severity.CRITICAL,
    ("Permission", "Full Gap"): Severity.MODERATE,
    ("Permission", "Partial Gap"): Severity.MINOR,
    ("Recommendation", "Full Gap"): Severity.MODERATE,
    ("Recommendation", "Partial Gap"): Severity.MINOR,
}


def compute_severity(
    modality: str,
    gap_type: str,
    has_enforcement_history: bool = False,
    cross_reference_count: int = 0,
) -> Severity:
    """Compute gap severity based on obligation characteristics.

    Args:
        modality: Deontic modality (Obligation, Permission, etc.).
        gap_type: Gap classification (Compliant, Partial Gap, Full Gap).
        has_enforcement_history: Whether related enforcement actions exist.
        cross_reference_count: Number of cross-references involved.

    Returns:
        Severity level.
    """
    if gap_type == "Compliant":
        return Severity.MINOR  # No gap

    base_severity = SEVERITY_RULES.get(
        (modality, gap_type), Severity.MAJOR
    )

    # Escalate if enforcement history exists
    if has_enforcement_history and base_severity < Severity.CRITICAL:
        base_severity = Severity(base_severity + 1)

    # Escalate for complex multi-reference obligations
    if cross_reference_count >= 3 and base_severity < Severity.CRITICAL:
        base_severity = Severity(base_severity + 1)

    return base_severity
