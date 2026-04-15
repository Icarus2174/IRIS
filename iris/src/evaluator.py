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
        passed=(has_compute_mesh if dense_expected else not has_compute_mesh),
        details={
            "communication_intensity": ir.communication_intensity,
            "dense_expected": dense_expected,
            "dense_present": has_compute_mesh,
            "reason": "training/high-comm should produce denser interconnect",
        },
    )

    # Change propagation: confirm IR exists and later stages use derived fields (lightweight)
    case_names_consistent = (
        parsed.case_name == ir.case_name == allocation.case_name == schedule.case_name == topology.case_name
    )
    add_check(
        "normalized_ir_used",
        passed=case_names_consistent,
        details={
            "case_names_consistent": case_names_consistent,
            "parsed": parsed.case_name,
            "ir": ir.case_name,
            "allocation": allocation.case_name,
            "schedule": schedule.case_name,
            "topology": topology.case_name,
            "reason": "later stages should be driven by the same normalized IR chain",
        },
    )

    # Runtime adaptation: schedule should record runtime_event when simulated telemetry is present
    runtime_mentioned = schedule.rationale.get("inputs", {}).get("runtime_event") is not None
    runtime_event_present = runtime_mentioned
    runtime_visible = True if not runtime_event_present else bool(feedback and feedback.applied)
    add_check(
        "runtime_adaptation_visible",
        passed=runtime_visible,
        details={
            "runtime_event_in_schedule_inputs": runtime_mentioned,
            "feedback_applied": bool(feedback and feedback.applied),
            "reason": "when a runtime event is present, feedback should be visibly applied (reschedule/recompile)",
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

