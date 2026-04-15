# Architecture notes

IRIS (Intent-Resolved Infrastructure Synthesis) is structured like a compiler pipeline:

- **Intent** (YAML) is parsed, validated, defaulted, and normalized.
- Intent is lowered into a **normalized IR** used by all later stages.
- The IR drives **resource allocation**.
- Allocation drives **scheduling**.
- Allocation + schedule drive **topology compilation** into:
  - Mermaid diagram-as-code (`topology.mmd`)
  - Machine-readable topology spec (`topology_spec.yaml`)
- A **simulated runtime event** can conservatively trigger re-evaluation.
- An **evaluator** runs sanity checks and summarizes pass/fail + warnings.

Each stage emits an artifact to `outputs/<case_name>/` so decisions are auditable and screenshot-able.

