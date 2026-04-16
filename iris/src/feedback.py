from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .models import Allocation, FeedbackResponse, IR, RuntimeEvent, Schedule, Topology
from .scheduler import schedule as schedule_fn
from .topology_compiler import compile_topology


def apply_feedback(
    ir: IR,
    allocation: Allocation,
    schedule: Schedule,
    topology: Topology,
    runtime_event: Optional[RuntimeEvent],
) -> Tuple[FeedbackResponse, Schedule, Topology]:
    if not runtime_event:
        resp = FeedbackResponse(
            case_name=ir.case_name,
            applied=False,
            effects={},
            rationale={"reason": "no runtime event provided"},
        )
        return resp, schedule, topology

    effects: Dict[str, Any] = {"rescheduled": False, "recompiled_topology": False}
    rationale: Dict[str, Any] = {"event": runtime_event.type, "reasons": []}

    # Conservative mapping:
    # - degraded orbit confidence / optical unavailable => reschedule; topology may be downgraded to packet
    # Conservative mapping:
# - degraded orbit confidence / optical unavailable => reschedule; topology may be downgraded to packet
    if runtime_event.type in (
        "orbital_link_confidence_low",
        "optical_path_unavailable",
        "topology_instability_detected",
        "safe_mode_requested",
    ):
        conf = runtime_event.details.get("confidence")
        conf_value = float(conf) if isinstance(conf, (int, float)) else None
        min_conf = float(ir.space_constraints.get("min_link_confidence_for_orbit", 0.45))

        if conf_value is not None:
            rationale["runtime_confidence"] = conf_value
            rationale["min_confidence_threshold"] = min_conf

        new_schedule = schedule_fn(ir, allocation, runtime_event=runtime_event)
        effects["rescheduled"] = True
        rationale["reasons"].append("runtime event triggers rescheduling")

        # Conservative downgrade logic:
        # - optical unavailable / safe mode => always prefer packet
        # - low confidence => prefer packet only if the numeric confidence is actually below threshold
        prefer_packet = runtime_event.type in ("optical_path_unavailable", "safe_mode_requested")

        if runtime_event.type == "orbital_link_confidence_low":
            prefer_packet = conf_value is None or conf_value < min_conf

        new_ir = ir
        if prefer_packet and ir.topology_requirement.get("preferred_link_mode") != "packet":
            tr = dict(ir.topology_requirement)
            tr["preferred_link_mode"] = "packet"
            new_ir = IR(
                case_name=ir.case_name,
                workload_class=ir.workload_class,
                communication_intensity=ir.communication_intensity,
                latency_class=ir.latency_class,
                delay_tolerant=ir.delay_tolerant,
                hardware_requirement=ir.hardware_requirement,
                placement_candidates=ir.placement_candidates,
                topology_requirement=tr,
                scheduling_urgency=ir.scheduling_urgency,
                space_constraints=ir.space_constraints,
                fallback_sensitivity=ir.fallback_sensitivity,
                explanation=dict(ir.explanation, feedback="runtime event enforced packet preference (safe mode)"),
            )
            rationale["reasons"].append("conservative response: downgrade link mode preference to packet")

        new_topology = compile_topology(new_ir, allocation, new_schedule)
        effects["recompiled_topology"] = True
        rationale["reasons"].append("runtime event triggers topology recompilation")

        resp = FeedbackResponse(case_name=ir.case_name, applied=True, effects=effects, rationale=rationale)
        return resp, new_schedule, new_topology

    # Unknown events: no action
    resp = FeedbackResponse(
        case_name=ir.case_name,
        applied=False,
        effects=effects,
        rationale={"event": runtime_event.type, "reason": "event type not recognized; no conservative action taken"},
    )
    return resp, schedule, topology

