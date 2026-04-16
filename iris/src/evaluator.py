from __future__ import annotations

from typing import Any, Dict, Optional

from .models import Allocation, FeedbackResponse, IR, Schedule, Topology


def evaluate_pipeline(
    ir: IR,
    allocation: Allocation,
    schedule: Schedule,
    topology: Topology,
    feedback: Optional[FeedbackResponse],
    baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checks = []

    def add_check(name: str, passed: bool, details: Dict[str, Any]) -> None:
        checks.append(
            {
                "name": name,
                "passed": passed,
                "details": details,
            }
        )

    add_check(
        "normalized_ir_used",
        passed=bool(
            ir.workload_class
            and ir.placement_candidates
            and ir.topology_requirement
        ),
        details={
            "workload_class": ir.workload_class,
            "placement_candidates": ir.placement_candidates,
            "topology_requirement_keys": sorted(list((ir.topology_requirement or {}).keys())),
            "reason": "downstream stages consumed populated IR fields",
        },
    )

    add_check(
        "allocation_present",
        passed=allocation is not None and bool(getattr(allocation, "placement", None)),
        details={
            "placement": getattr(allocation, "placement", None),
        },
    )

    add_check(
        "schedule_present",
        passed=schedule is not None and bool(getattr(schedule, "steps", None)),
        details={
            "num_steps": len(getattr(schedule, "steps", []) or []),
        },
    )

    add_check(
        "topology_present",
        passed=topology is not None and bool(getattr(topology, "links", None)),
        details={
            "num_links": len(getattr(topology, "links", []) or []),
        },
    )

    if feedback is None:
        runtime_visible = True
        runtime_details = {"reason": "no runtime event / no feedback stage output"}
    else:
        effects = feedback.effects or {}
        runtime_visible = bool(
            effects.get("rescheduled") or effects.get("recompiled_topology")
        )
        runtime_details = {
            "applied": feedback.applied,
            "effects": effects,
            "rationale": feedback.rationale,
        }

    add_check(
        "runtime_adaptation_visible",
        passed=runtime_visible,
        details=runtime_details,
    )

    passed_all = all(c["passed"] for c in checks)

    return {
        "passed": passed_all,
        "checks": checks,
        "baseline": baseline or {},
    }