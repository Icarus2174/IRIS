# IRIS — Intent-Resolved Infrastructure Synthesis (prototype intent-to-topology compiler)

IRIS is a **course-project prototype** that turns a **structured workload intent** (YAML) into:

- **Inspectable intermediate artifacts** (parsed intent, IR, allocation, schedule, evaluation)
- A generated topology in **two forms**:
  - **Human-readable Diagram-as-Code** (`topology.mmd`, Mermaid)
  - **Machine-readable topology spec** (`topology_spec.yaml`)

It is **not** a real deployment/orchestration system:

- Allocation/scheduling are **rule-based heuristics**
- Runtime feedback is **simulated**
- No real networking, router configs, optical control, or cluster APIs

## Quickstart

```bash
cd iris
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --case inputs/case_training_space.yaml
python -m src.main --case inputs/case_inference_earth.yaml
python -m src.main --case inputs/case_archive_orbit.yaml
python -m src.main --case inputs/case_runtime_alert.yaml
```

Outputs are written to `outputs/<case_name>/`.

## Pipeline stages

1. Parse + validate intent (`parsed_intent.json`)
2. Lower to normalized IR (`ir.json`)
3. Allocate resources (`allocation.json`)
4. Schedule actions (`schedule.json`)
5. Compile topology:
   - `topology_spec.yaml`
   - `topology.mmd`
6. Apply (optional) runtime feedback (may re-run schedule/topology)
7. Evaluate sanity checks (`evaluation.json`)

## Inputs

See `inputs/` for the included cases. Intent is declarative (YAML) and is the **only** user-controlled input to the pipeline.

