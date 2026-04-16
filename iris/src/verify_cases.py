from pathlib import Path

from main import run_case

CASES = [
    "case_training_space.yaml",
    "case_inference_earth.yaml",
    "case_archive_orbit.yaml",
    "case_runtime_alert.yaml",
]

REQUIRED = [
    "parsed_intent.json",
    "ir.json",
    "allocation.json",
    "schedule.json",
    "runtime_response.json",
    "topology_spec.yaml",
    "topology.mmd",
    "evaluation.json",
]

for case in CASES:
    out = Path(run_case(case))
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        raise SystemExit(f"{case} failed: missing {missing}")

print("All cases generated expected artifacts.")