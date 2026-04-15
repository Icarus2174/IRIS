from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


WorkloadType = Literal["training", "inference", "archive"]
LatencySensitivity = Literal["strict", "high", "medium", "low", "none"]
EnergyPreference = Literal["lowest_energy", "low_cost", "balanced", "performance"]
Reliability = Literal["high", "medium", "low"]
PlacementPreference = Literal["earth", "space", "hybrid", "any"]
NetworkPreference = Literal["packet", "optical", "hybrid", "any"]
AcceleratorType = Literal["gpu", "tpu", "none"]


@dataclass(frozen=True)
class ParsedIntent:
    case_name: str
    job_type: WorkloadType
    accelerator_count: int
    accelerator_type: AcceleratorType
    min_memory_gb: int
    objectives: List[str]
    latency_sensitivity: LatencySensitivity
    energy_preference: EnergyPreference
    reliability: Reliability
    placement_preference: PlacementPreference
    network_preference: NetworkPreference
    runtime_policy: Dict[str, Any]
    raw_source: Dict[str, Any] = field(default_factory=dict)
    explanations: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IR:
    case_name: str
    workload_class: WorkloadType
    communication_intensity: Literal["low", "medium", "high"]
    latency_class: Literal["strict", "normal", "relaxed"]
    delay_tolerant: bool
    hardware_requirement: Dict[str, Any]
    placement_candidates: List[str]
    topology_requirement: Dict[str, Any]
    scheduling_urgency: Literal["immediate", "soon", "whenever"]
    space_constraints: Dict[str, Any]
    fallback_sensitivity: Literal["low", "medium", "high"]
    explanation: Dict[str, str]


@dataclass(frozen=True)
class Cluster:
    id: str
    domain: Literal["earth", "orbit"]
    accelerators_total: int
    accelerators_type: Literal["gpu", "tpu"]
    memory_gb_per_accelerator: int
    energy_profile: Literal["grid", "solar", "mixed"]
    link_confidence: float  # 0..1
    notes: str = ""


@dataclass(frozen=True)
class Allocation:
    case_name: str
    selected_clusters: List[Dict[str, Any]]
    placement: Literal["earth", "orbit", "hybrid"]
    packed: bool
    auxiliary_roles: Dict[str, Any]
    rationale: Dict[str, Any]


@dataclass(frozen=True)
class Schedule:
    case_name: str
    decision: Literal[
        "launch_immediately",
        "wait_for_topology",
        "delay_due_to_runtime_risk",
        "safe_mode_schedule",
    ]
    actions: List[Dict[str, Any]]
    topology_required_before_launch: bool
    rationale: Dict[str, Any]


@dataclass(frozen=True)
class Topology:
    case_name: str
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    groups: List[Dict[str, Any]]
    rationale: Dict[str, Any]


@dataclass(frozen=True)
class RuntimeEvent:
    case_name: str
    type: str
    severity: Literal["low", "medium", "high"]
    details: Dict[str, Any]


@dataclass(frozen=True)
class FeedbackResponse:
    case_name: str
    applied: bool
    effects: Dict[str, Any]
    rationale: Dict[str, Any]


@dataclass(frozen=True)
class Evaluation:
    case_name: str
    checks: List[Dict[str, Any]]
    summary: Dict[str, Any]
    baseline_comparison: Optional[Dict[str, Any]] = None

