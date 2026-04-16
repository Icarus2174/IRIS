from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from .allocator import allocate
from .evaluator import evaluate
from .feedback import apply_feedback
from .ir import lower_to_ir
from .models import RuntimeEvent
from .parser import parse_intent
from .scheduler import schedule as schedule_fn
from .topology_compiler import compile_topology, topology_to_mermaid
from .utils import dump_json, dump_yaml, ensure_dir, write_text


def _runtime_event_from_dict(case_name: str, d: Dict[str, Any]) -> Optional[RuntimeEvent]:
    if not d:
        return None
    return RuntimeEvent(
        case_name=case_name,
        type=str(d.get("type", "unknown")),
        severity=str(d.get("severity", "medium")),  # type: ignore[arg-type]
        details=dict(d.get("details") or {}),
    )


def run_case(case_path: str | Path, outputs_root: str | Path = "outputs") -> Path:
    parsed, runtime_event_dict = parse_intent(case_path)
    ir = lower_to_ir(parsed)
    allocation = allocate(ir)
    runtime_event = _runtime_event_from_dict(parsed.case_name, runtime_event_dict)

    schedule = schedule_fn(ir, allocation, runtime_event=runtime_event)
    topology = compile_topology(ir, allocation, schedule)

    feedback_resp, schedule2, topology2 = apply_feedback(
        ir, allocation, schedule, topology, runtime_event
    )

    final_link_modes = sorted(
        {
            link.get("type")
            for link in topology2.links
            if isinstance(link, dict) and "type" in link
        }
    )

    baseline = {
        "baseline": "earth_only_packet_only_no_feedback",
        "notes": "naive baseline for evaluation narrative",
        "diff_hints": {
            "placement": {
                "baseline": "earth",
                "chosen": allocation.placement,
            },
            "preferred_link_mode": {
                "baseline": "packet",
                "chosen": (
                    final_link_modes[0]
                    if len(final_link_modes) == 1
                    else ",".join(final_link_modes)
                    if final_link_modes
                    else "none"
                ),
            },
            "runtime_feedback": {
                "baseline": "disabled",
                "chosen": "enabled",
            },
        },
    }

    evaluation = evaluate(
        parsed,
        ir,
        allocation,
        schedule2,
        topology2,
        baseline=baseline,
        feedback=feedback_resp,
    )

    out_dir = ensure_dir(Path(outputs_root) / parsed.case_name)

    dump_json(out_dir / "parsed_intent.json", parsed)
    dump_json(out_dir / "ir.json", ir)
    dump_json(out_dir / "allocation.json", allocation)
    dump_json(out_dir / "schedule.json", schedule2)
    dump_json(out_dir / "runtime_event.json", runtime_event_dict or {})
    dump_json(out_dir / "runtime_response.json", feedback_resp)
    dump_yaml(out_dir / "topology_spec.yaml", topology2)
    write_text(out_dir / "topology.mmd", topology_to_mermaid(topology2))
    dump_json(out_dir / "evaluation.json", evaluation)

    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="IRIS (Intent-Resolved Infrastructure Synthesis) prototype compiler pipeline"
    )
    ap.add_argument("--case", required=True, help="Path to YAML case file (inputs/*.yaml)")
    ap.add_argument("--outputs", default="outputs", help="Outputs root directory")
    args = ap.parse_args()

    out_dir = run_case(args.case, outputs_root=args.outputs)
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()