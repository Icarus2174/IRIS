from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import Allocation, IR, RuntimeEvent, Schedule


def schedule(ir: IR, allocation: Allocation, runtime_event: Optional[RuntimeEvent] = None) -> Schedule:
    rationale: Dict[str, Any] = {"inputs": {}, "reasons": []}
    actions: List[Dict[str, Any]] = []

    rationale["inputs"] = {
        "workload_class": ir.workload_class,
        "communication_intensity": ir.communication_intensity,
        "latency_class": ir.latency_class,
        "placement": allocation.placement,
        "runtime_event": runtime_event.type if runtime_event else None,
    }

    # Default: topology compiled before launch for distributed/hybrid workloads.
    topology_required = allocation.placement in ("hybrid", "orbit") or ir.communication_intensity == "high"

    # Runtime caution: certain events force delay or safe mode.
    if runtime_event and runtime_event.type in ("orbital_link_confidence_low", "optical_path_unavailable", "topology_instability_detected"):
        if ir.latency_class == "strict" and allocation.placement != "earth":
            rationale["reasons"].append("strict latency + degraded orbit signals => force earth-safe scheduling")
            decision = "safe_mode_schedule"
            topology_required = False
            actions = [
                {"action": "switch_to_earth_only", "reason": "strict latency cannot tolerate degraded orbital confidence"},
                {"action": "launch_job", "reason": "earth resources preferred for strict latency"},
            ]
            return Schedule(
                case_name=ir.case_name,
                decision=decision,  # type: ignore[arg-type]
                actions=actions,
                topology_required_before_launch=topology_required,
                rationale=rationale,
            )

        if ir.communication_intensity == "high":
            decision = "delay_due_to_runtime_risk"
            rationale["reasons"].append("high-comm job + degraded link confidence => delay launch conservatively")
            actions.append({"action": "compile_topology", "reason": "ensure conservative topology before launch"})
            actions.append({"action": "recheck_runtime_confidence", "reason": "wait for improved confidence or fallback trigger"})
            actions.append({"action": "launch_job_when_safe", "reason": "avoid launching communication-heavy job on unstable links"})
            return Schedule(
                case_name=ir.case_name,
                decision=decision,  # type: ignore[arg-type]
                actions=actions,
                topology_required_before_launch=True,
                rationale=rationale,
            )

        # Delay-tolerant work can wait; strict inference should run if earth-ready
        if ir.delay_tolerant:
            decision = "delay_due_to_runtime_risk"
            rationale["reasons"].append("delay-tolerant workload + runtime alert => wait window")
            actions = [
                {"action": "wait", "reason": "runtime risk present; job is delay-tolerant"},
                {"action": "compile_topology", "reason": "topology can be prepared later"},
            ]
            return Schedule(
                case_name=ir.case_name,
                decision=decision,  # type: ignore[arg-type]
                actions=actions,
                topology_required_before_launch=False,
                rationale=rationale,
            )

    # Non-alert scheduling heuristics
    if ir.latency_class == "strict":
        rationale["reasons"].append("strict latency => launch immediately if earth resources allocated")
        decision = "launch_immediately"
        topology_required = False
        actions = [{"action": "launch_job", "reason": "minimize latency; avoid waiting on topology reconfiguration"}]
    elif topology_required:
        rationale["reasons"].append("hybrid/orbit/high-comm => wait for topology compilation")
        decision = "wait_for_topology"
        actions = [
            {"action": "compile_topology", "reason": "placement/topology dependencies must be prepared"},
            {"action": "launch_job", "reason": "launch after topology is ready"},
        ]
    else:
        rationale["reasons"].append("no special constraints => launch")
        decision = "launch_immediately"
        actions = [{"action": "launch_job", "reason": "resources ready and topology not required"}]

    return Schedule(
        case_name=ir.case_name,
        decision=decision,  # type: ignore[arg-type]
        actions=actions,
        topology_required_before_launch=topology_required,
        rationale=rationale,
    )

