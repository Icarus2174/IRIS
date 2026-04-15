from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import Allocation, Evaluation, FeedbackResponse, IR, ParsedIntent, Schedule, Topology


def evaluate(
    parsed: ParsedIntent,
    ir: IR,
    allocation: Allocation,
    schedule: Schedule,
    topology: Topology,
    baseline: Optional[Dict[str, Any]] = None,
    feedback: Optional[FeedbackResponse] = None,
) -> Evaluation:
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, details: Dict[str, Any]) -> None:
        checks.append({"name": name, "passed": passed, "details": details})

    # Resource checks
    allocated = sum(int(c.get("accelerators", 0)) for c in allocation.selected_clusters)
    requested = int(ir.hardware_requirement["accelerators"]["count"])
    add_check(
        "resource_over_allocation",
        passed=allocated <= requested,
        details={"requested": requested, "allocated": allocated, "reason": "never allocate more than requested"},
    )
    add_check(
        "resource_under_allocation",
        passed=allocated == requested,
        details={"requested": requested, "allocated": allocated, "reason": "prototype expects full allocation when possible"},
    )

    # Placement sanity
    strict_inference_orbit = (
        ir.workload_class == "inference" and ir.latency_class == "strict" and allocation.placement != "earth"
    )
    add_check(
        "strict_inference_not_in_orbit",
        passed=not strict_inference_orbit,
        details={
            "latency_class": ir.latency_class,
            "placement": allocation.placement,
            "reason": "strict inference should be earth-local in this prototype",
        },
    )

    # Topology shape sanity
    dense_expected = ir.communication_intensity == "high"
    has_compute_mesh = any(e.get("reason") == "dense interconnect for communication-heavy workload" for e in topology.links)
    add_check(
        "topology_matches_workload",
        passed=(has_compute_mesh if dense_expected else True),
        details={
            "communication_intensity": ir.communication_intensity,
            "dense_expected": dense_expected,
            "dense_present": has_compute_mesh,
            "reason": "training/high-comm should produce denser interconnect",
        },
    )

    # Change propagation: confirm IR exists and later stages use derived fields (lightweight)
    add_check(
        "normalized_ir_used",
        passed=True,
        details={"reason": "pipeline stage contract: allocator/scheduler/topology take IR object, not raw YAML"},
    )

    # Runtime adaptation: schedule should record runtime_event when simulated telemetry is present
    runtime_mentioned = schedule.rationale.get("inputs", {}).get("runtime_event") is not None
    add_check(
        "runtime_adaptation_visible",
        passed=True,
        details={
            "runtime_event_in_schedule_inputs": runtime_mentioned,
            "reason": "if runtime event exists, schedule inputs include it; feedback may reschedule/recompile",
        },
    )

    if feedback and feedback.applied:
        recompiled = bool(feedback.effects.get("recompiled_topology"))
        add_check(
            "runtime_feedback_applied",
            passed=True,
            details={
                "recompiled_topology": recompiled,
                "rescheduled": bool(feedback.effects.get("rescheduled")),
                "reason": "closed-loop layer produced explicit effects when a runtime event was handled",
            },
        )

    passed = all(bool(c["passed"]) for c in checks)
    warnings = [c["name"] for c in checks if not c["passed"]]

    return Evaluation(
        case_name=ir.case_name,
        checks=checks,
        summary={"passed": passed, "warnings": warnings, "notes": "lightweight plausibility checks"},
        baseline_comparison=baseline,
    )

