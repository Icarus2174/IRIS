from __future__ import annotations

from typing import Any, Dict, List

from .models import IR, ParsedIntent


def lower_to_ir(parsed: ParsedIntent) -> IR:
    explanation: Dict[str, str] = {}

    # Workload-derived properties
    if parsed.job_type == "training":
        communication_intensity = "high"
        latency_class = "normal"
        delay_tolerant = False
        topology_style = "bandwidth_heavy"
        urgency = "soon"
        explanation["workload"] = "distributed training modeled as high communication intensity"
    elif parsed.job_type == "inference":
        communication_intensity = "low"
        latency_class = "strict" if parsed.latency_sensitivity == "strict" else "normal"
        delay_tolerant = False
        topology_style = "local_simple"
        urgency = "immediate"
        explanation["workload"] = "inference modeled as latency-sensitive and low communication intensity"
    else:  # archive
        communication_intensity = "low"
        latency_class = "relaxed"
        delay_tolerant = True
        topology_style = "storage_transfer"
        urgency = "whenever"
        explanation["workload"] = "archive modeled as delay-tolerant and storage-transfer oriented"

    # Placement candidates
    candidates: List[str] = []
    if parsed.placement_preference == "earth":
        candidates = ["earth"]
    elif parsed.placement_preference == "space":
        candidates = ["orbit", "hybrid"]
    elif parsed.placement_preference == "hybrid":
        candidates = ["hybrid", "earth", "orbit"]
    else:
        candidates = ["earth", "hybrid", "orbit"]

    allow_hybrid = bool(parsed.runtime_policy.get("allow_hybrid", True))
    if not allow_hybrid:
        candidates = [c for c in candidates if c != "hybrid"]
        explanation["runtime_policy.allow_hybrid"] = "disabled hybrid; removed 'hybrid' from placement candidates"

    # Space-specific constraints (explicit)
    orbit_candidate = "orbit" in candidates or "hybrid" in candidates
    space_constraints: Dict[str, Any] = {
        "orbit_candidate": orbit_candidate,
        "min_link_confidence_for_orbit": 0.6 if parsed.runtime_policy.get("conservative") else 0.45,
        "prefer_orbit_for_energy": parsed.energy_preference in ("lowest_energy",) and orbit_candidate,
    }

    # Fallback sensitivity: strict latency and high reliability demand higher caution
    if parsed.latency_sensitivity == "strict" or parsed.reliability == "high":
        fallback = "high"
        explanation["fallback"] = "strict latency and/or high reliability increases fallback sensitivity"
    elif parsed.latency_sensitivity in ("high", "medium"):
        fallback = "medium"
        explanation["fallback"] = "moderate latency sensitivity yields moderate fallback sensitivity"
    else:
        fallback = "low"
        explanation["fallback"] = "low/none latency sensitivity yields low fallback sensitivity"

    # Topology requirement (normalized)
    preferred_link_mode = parsed.network_preference
    if preferred_link_mode == "hybrid" and not allow_hybrid:
        preferred_link_mode = "packet"
        explanation["network.preference"] = "hybrid requested but disallowed by runtime policy; coerced to packet"

    topology_requirement: Dict[str, Any] = {
        "style": topology_style,
        "preferred_link_mode": preferred_link_mode,
        "need_dense_interconnect": communication_intensity == "high",
        "stability_requirement": "high" if fallback == "high" else "medium",
    }

    hardware_req: Dict[str, Any] = {
        "accelerators": {"count": parsed.accelerator_count, "type": parsed.accelerator_type},
        "min_memory_gb": parsed.min_memory_gb,
    }

    return IR(
        case_name=parsed.case_name,
        workload_class=parsed.job_type,
        communication_intensity=communication_intensity,  # type: ignore[arg-type]
        latency_class=latency_class,  # type: ignore[arg-type]
        delay_tolerant=delay_tolerant,
        hardware_requirement=hardware_req,
        placement_candidates=candidates,
        topology_requirement=topology_requirement,
        scheduling_urgency=urgency,  # type: ignore[arg-type]
        space_constraints=space_constraints,
        fallback_sensitivity=fallback,  # type: ignore[arg-type]
        explanation=explanation,
    )

