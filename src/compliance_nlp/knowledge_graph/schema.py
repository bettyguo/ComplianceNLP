"""Regulatory Knowledge Graph (RKG) schema definitions.

Five node types: Provision, Entity, Obligation, Threshold, Enforcement.
Five edge types: Amends, Supersedes, CrossReferences, Implements, AppliesTo.

Statistics: 12,847 provision nodes (SEC: 4,932; MiFID II: 4,218; Basel III: 3,697),
1,247 entity nodes, 8,431 obligation nodes, 612 threshold nodes, 847 enforcement nodes,
34,219 edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    """Node types in the Regulatory Knowledge Graph."""

    PROVISION = "Provision"
    ENTITY = "Entity"
    OBLIGATION = "Obligation"
    THRESHOLD = "Threshold"
    ENFORCEMENT = "Enforcement"


class EdgeType(str, Enum):
    """Edge types in the Regulatory Knowledge Graph."""

    AMENDS = "Amends"
    SUPERSEDES = "Supersedes"
    CROSS_REFERENCES = "CrossReferences"
    IMPLEMENTS = "Implements"
    APPLIES_TO = "AppliesTo"


class Framework(str, Enum):
    """Supported regulatory frameworks."""

    SEC = "SEC"
    MIFID_II = "MiFID II"
    BASEL_III = "Basel III"


@dataclass
class ProvisionNode:
    """A regulatory article, section, or clause."""

    id: str
    text: str
    framework: Framework
    title: str = ""
    effective_date: str = ""
    source_url: str = ""
    embedding: Optional[list[float]] = None


@dataclass
class EntityNode:
    """A regulated entity type (e.g., 'credit institution', 'investment firm')."""

    id: str
    name: str
    entity_type: str
    framework: Framework
    description: str = ""


@dataclass
class ObligationNode:
    """A structured obligation with modality, action, and conditions."""

    id: str
    entity: str
    action: str
    modality: str  # Obligation, Permission, Prohibition, Recommendation
    condition: str = ""
    source_provision_id: str = ""
    confidence: float = 0.0


@dataclass
class ThresholdNode:
    """A quantitative regulatory threshold (e.g., Basel III CET1 >= 4.5%)."""

    id: str
    name: str
    value: str
    metric: str
    framework: Framework
    source_provision_id: str = ""


@dataclass
class EnforcementNode:
    """A documented enforcement action."""

    id: str
    entity_name: str
    penalty_amount: str
    date: str
    description: str = ""
    framework: Framework = Framework.SEC


@dataclass
class KGEdge:
    """An edge connecting two nodes in the RKG."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class KGStats:
    """Statistics about the Regulatory Knowledge Graph."""

    total_provisions: int = 12_847
    sec_provisions: int = 4_932
    mifid_provisions: int = 4_218
    basel_provisions: int = 3_697
    entity_nodes: int = 1_247
    obligation_nodes: int = 8_431
    threshold_nodes: int = 612
    enforcement_nodes: int = 847
    total_edges: int = 34_219
    edge_precision: float = 0.947
    edge_recall: float = 0.873

    def __str__(self) -> str:
        return (
            f"RKG Stats: {self.total_provisions} provisions "
            f"(SEC: {self.sec_provisions}, MiFID II: {self.mifid_provisions}, "
            f"Basel III: {self.basel_provisions}), "
            f"{self.total_edges} edges "
            f"(precision: {self.edge_precision:.1%}, recall: {self.edge_recall:.1%})"
        )
