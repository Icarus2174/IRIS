# Compiling Topology from Intent for Earth-Based and Space-Borne AI Data Centers

**First draft (conceptual) — revised with implementation evidence**

This document merges the architectural narrative of the first-draft report with **empirical support from the IRIS (Intent-Resolved Infrastructure Synthesis) prototype** in `/iris`. Use it in three ways:

1. **In the written report:** Copy the “Implementation evidence” paragraphs and abbreviated artifact listings into your PDF; cite repository paths and figure numbers you create from `outputs/<case>/`.
2. **In slides:** Show the pipeline diagram (six stages), one `inputs/*.yaml` snippet, then side-by-side `ir.json` and `topology.mmd` for a case that changed intent.
3. **Reproducibility:** Anyone can re-run the pipeline and regenerate the same JSON/YAML/Mermaid artifacts (see §Reproducibility).

---

## Abstract

Future AI infrastructure will likely span heterogeneous compute environments rather than a single, static terrestrial datacenter. In such settings, operators must reason simultaneously about workload intent, hardware placement, scheduling, and network topology. Existing systems address important slices of this problem: Helios shows that datacenter topology can be reconfigured using hybrid electrical and optical switching; Merlin, Propane, and Aura show that high-level network intent can be compiled into enforceable lower-level behavior; DCM and COPE show that cluster orchestration can be driven by declarative policy and constraint solving; and Gavel, Tiresias, Pollux, and CASSINI show that accelerator heterogeneity, queueing uncertainty, adaptation, and network-aware scheduling materially affect AI job performance. However, these systems generally still treat topology as a design assumption, a control substrate, or a manually specified artifact rather than as a first-class compilation target. This report proposes an intent-driven architecture in which topology itself becomes a programmable entity: a machine-generated, machine-readable artifact derived from structured computational intent, space-specific constraints, and runtime network state. The proposed system uses a six-layer pipeline: an intent front end, an intermediate representation, a resource allocation backend, a scheduling backend, a topology compiler backend, and a runtime feedback layer. The resulting output is both a logical Diagram-as-Code artifact and a machine-readable deployment specification. The report argues that this architecture is plausible, practical, and innovative, and it outlines a prototype path that implements the core compilation pipeline while simulating runtime adaptation.

**Implementation evidence (new).** The accompanying **IRIS (Intent-Resolved Infrastructure Synthesis)** prototype (`iris/`) implements that pipeline in Python: structured YAML intent files are parsed and canonicalized; lowered to a normalized intermediate representation; passed through a rule-based allocator and scheduler; compiled into a single internal topology object that is rendered identically into **Mermaid** (`topology.mmd`) and **YAML** (`topology_spec.yaml`); optional simulated runtime events can trigger conservative rescheduling and topology recompilation; and an `evaluation.json` file records explicit plausibility checks. The sections below reference concrete code paths and saved artifacts so the design claim is supported by reproducible outputs, not only by narrative.

---

## 1. Introduction

Large AI workloads increasingly stress not only compute capacity but also interconnect design, job placement, and scheduling discipline. Training workloads generate sustained, communication-heavy traffic patterns; inference workloads impose tighter latency demands; checkpointing and data movement introduce large but delay-tolerant transfers. In terrestrial datacenters, these issues already strain traditional assumptions about fixed topology and human-managed orchestration. In hybrid earth-based and space-borne infrastructure, the problem becomes harder: placement must consider long and variable paths, intermittent or confidence-sensitive optical links, heterogeneous accelerator pools, and topology changes induced by orbital motion rather than only terrestrial congestion.

This report takes the position that topology should not be treated merely as a background assumption, a one-off architectural sketch, or a substrate beneath schedulers. Instead, topology should be treated as a **compilation target**. Under this view, an operator declares computational intent in structured form—for example, a large distributed training job that prefers space-borne compute, or an inference service that must remain Earth-local and latency conservative—and the system compiles that intent into a placement plan, a schedule, and a machine-readable logical topology.

A second guiding decision is to treat **space-specific constraints** as explicit fields in the intent model, constraints in allocation and scheduling, and triggers in runtime feedback, so that space is a design driver rather than window dressing.

**Implementation evidence.** The introduction’s claims are reflected in inspectable artifacts. Editing `iris/inputs/case_training_space.yaml` (placement `space`, hybrid network preference) and running the compiler yields `outputs/case_training_space/`, where `ir.json` encodes explicit space-related fields (for example `placement_candidates`, `space_constraints`) and `topology.mmd` separates Earth, orbit, coordination, and storage subgraphs. That supports the argument that the workflow starts from a **formal request**, not from a hand-drawn diagram.

---

## 2–3. Background, problem statement, and design goals

*(Your original Sections 2–3 develop Helios, intent compilation, scheduling precedents, ATAR-style runtime caution, and the five design goals. Those sections stand unchanged in your course submission; they provide the literature and motivation for the architecture below.)*

**Cross-reference to code.** The prototype’s `README.md` states scope honestly: heuristics rather than full solvers, simulated runtime rather than live telemetry—consistent with Section 9 limitations. Mapping from **papers → prototype modules** is also documented in `iris/docs/research_mapping.md`.

---

## 4. Proposed architecture: IRIS (with prototype realization)

The six layers below follow the first-draft structure. After each layer, **Implementation** describes what the repository actually does and points to code and outputs.

### 4.1 Layer 1: Intent front end

**Design.** The front end accepts structured intent (workload type, scale, hardware class, latency sensitivity, placement preference, energy preference, topology-related preferences) and validates and canonicalizes it rather than optimizing.

**Implementation.** `parse_intent()` in `src/parser.py` loads YAML, applies defaults, validates required keys, and maps synonymous strings (for example placement and energy synonyms) through `canonicalize()` in `src/utils.py`. The parser emits a `ParsedIntent` dataclass (`src/models.py`). Rejection of inconsistent inputs (for example inference with zero accelerators) is implemented explicitly.

**Evidence artifact:** `outputs/<case_name>/parsed_intent.json` — demonstrates normalization and traceability from the source YAML.

### 4.2 Layer 2: Intermediate representation

**Design.** The IR is a shared, typed object so allocation, scheduling, and topology logic do not re-parse user language; space-specific constraints are explicit.

**Implementation.** `lower_to_ir()` in `src/ir.py` constructs an `IR` with `communication_intensity`, `placement_candidates`, `topology_requirement`, `space_constraints`, and explanations. For example, training workloads are modeled as high communication intensity; placement preference `space` constrains candidate sites to orbit/hybrid paths in this prototype.

**Evidence artifact:** `outputs/<case_name>/ir.json`. Excerpt for the space-preferred training case:

```json
{
  "workload_class": "training",
  "communication_intensity": "high",
  "placement_candidates": ["orbit", "hybrid"],
  "space_constraints": {
    "min_link_confidence_for_orbit": 0.45,
    "orbit_candidate": true,
    "prefer_orbit_for_energy": false
  },
  "topology_requirement": {
    "need_dense_interconnect": true,
    "preferred_link_mode": "hybrid",
    "style": "bandwidth_heavy"
  }
}
```

### 4.3 Layer 3: Resource allocation backend

**Design.** The allocator chooses concrete placement and accelerator counts subject to inventory and sanity rules, and records rationale.

**Implementation.** `allocate()` in `src/allocator.py` scores a static inventory of Earth and orbit clusters. Strict latency workloads are forced to Earth-only selection; orbit clusters below a confidence threshold can be rejected when fallback sensitivity is high; allocation never exceeds requested accelerator count.

**Evidence artifact:** `outputs/<case_name>/allocation.json` — includes per-cluster scoring and rejection reasons in `rationale`.

### 4.4 Layer 4: Scheduling backend

**Design.** The scheduler chooses among immediate launch, waiting for topology readiness, delay under runtime risk, or safe-mode behavior, with structured explanations.

**Implementation.** `schedule()` in `src/scheduler.py` takes `IR`, `Allocation`, and optional `RuntimeEvent`. Communication-heavy or hybrid placements can require topology before launch; simulated orbital degradation can force delay or safe-mode paths.

**Evidence artifact:** `outputs/<case_name>/schedule.json`.

### 4.5 Layer 5: Topology compiler backend

**Design.** The topology layer produces **two** representations from one decision: Diagram-as-Code (Mermaid) for humans and a machine-readable spec for downstream tooling—driven by a **single** topology structure so the two cannot silently diverge.

**Implementation.** `compile_topology()` builds a `Topology` (nodes, links, groups, rationale). `run_case()` then writes **both** `topology_spec.yaml` and `topology_to_mermaid(topology2)` — after runtime feedback, always from the same final `topology2` object.

```29:64:iris/src/main.py
def run_case(case_path: str | Path, outputs_root: str | Path = "outputs") -> Path:
    parsed, runtime_event_dict = parse_intent(case_path)
    ir = lower_to_ir(parsed)
    allocation = allocate(ir)
    runtime_event = _runtime_event_from_dict(parsed.case_name, runtime_event_dict)
    schedule = schedule_fn(ir, allocation, runtime_event=runtime_event)
    topology = compile_topology(ir, allocation, schedule)

    feedback_resp, schedule2, topology2 = apply_feedback(ir, allocation, schedule, topology, runtime_event)

    # Optional naive baseline comparison (very lightweight): earth-only + packet-only
    baseline = {
        "baseline": "earth_only_packet_only_no_feedback",
        "notes": "naive baseline for evaluation narrative",
        "diff_hints": {
            "placement": {"baseline": "earth", "chosen": allocation.placement},
            "preferred_link_mode": {"baseline": "packet", "chosen": ir.topology_requirement.get("preferred_link_mode")},
            "runtime_feedback": {"baseline": "disabled", "chosen": "enabled"},
        },
    }

    evaluation = evaluate(
        parsed, ir, allocation, schedule2, topology2, baseline=baseline, feedback=feedback_resp
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
```

**Analysis.** This directly supports the report’s claim that Mermaid is the **human-facing** logical topology artifact while YAML carries **machine-facing** structure. Link `type` values (packet vs optical) implement the Helios-inspired split between stable, high-volume paths and lighter or bursty connectivity, at logical granularity.

**Evidence artifacts:** `topology_spec.yaml`, `topology.mmd`. Excerpt (training, two orbit clusters):

```text
subgraph orbit[Orbit]
  orbit-leo-solar-1["orbit-leo-solar-1"]
  orbit-leo-solar-2["orbit-leo-solar-2"]
end
  orbit-leo-solar-1 -- "optical:high" --> orbit-leo-solar-2
```

### 4.6 Layer 6: Runtime feedback layer

**Design.** After compilation, simulated orbital or optical conditions may require conservative behavior: reschedule, downgrade link preference, recompile topology—without pretending to be full network control.

**Implementation.** `apply_feedback()` in `src/feedback.py` handles event types such as `orbital_link_confidence_low`. It may rewrite the IR’s `topology_requirement["preferred_link_mode"]` to `packet`, re-invoke the scheduler, and **recompile** topology.

```31:60:iris/src/feedback.py
    if runtime_event.type in ("orbital_link_confidence_low", "optical_path_unavailable", "topology_instability_detected", "safe_mode_requested"):
        new_schedule = schedule_fn(ir, allocation, runtime_event=runtime_event)
        effects["rescheduled"] = True
        rationale["reasons"].append("runtime event triggers rescheduling")

        # If safe-mode or confidence-low and we had optical, recompile with packet preference (conservative).
        prefer_packet = runtime_event.type in ("orbital_link_confidence_low", "optical_path_unavailable", "safe_mode_requested")
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
```

**Evidence artifacts:** `runtime_event.json`, `runtime_response.json`; case `inputs/case_runtime_alert.yaml` inherits the training intent and adds a `runtime_event` block.

---

## 5. Diagram-as-Code and deployment spec

The first draft argues Mermaid is appropriate for versionable visualization while JSON/YAML carries deployable semantics. The prototype implements this split **from one `Topology` object**, avoiding duplicate competing generators.

---

## 6. Prototype scope and case studies

The repository ships four YAML cases under `iris/inputs/`:

| Case file | Role in the report |
|-----------|---------------------|
| `case_training_space.yaml` | Space-preferred distributed training; stresses allocation and dense optical-style links. |
| `case_inference_earth.yaml` | Low-latency, Earth-local inference. |
| `case_archive_orbit.yaml` | Delay-tolerant archive-style workload. |
| `case_runtime_alert.yaml` | Inherits training intent; injects simulated `orbital_link_confidence_low` for closed-loop demonstration. |

Each successful run writes a full **compilation chain** under `iris/outputs/<case_name>/`, supporting the report’s requirement that scenarios be diffable and reviewable.

---

## 7. Evaluation: plausibility, practicality, innovation grounded in output

The first draft asks for constraint checks and comparison to naive baselines. The evaluator (`src/evaluator.py`) records checks such as allocation not exceeding request, strict inference not placed off-Earth, topology density consistent with high communication intensity, and—when runtime feedback runs—explicit `runtime_feedback_applied` with `recompiled_topology` / `rescheduled` flags.

**Evidence artifact:** `outputs/case_runtime_alert/evaluation.json` includes checks that fail or pass with structured `details`, plus `baseline_comparison` hints (Earth-only, packet-only) generated in `run_case()` for narrative contrast.

**Analysis for the report text.** You can truthfully state: *Plausibility is not only narrative: the prototype emits machine-readable checks in `evaluation.json`. Practicality is supported by a linear artifact chain per case. Innovation is framed as end-to-end intent-to-topology compilation with explicit space constraints and simulated runtime recompilation—demonstrated by differing `topology.mmd` before and after runtime events when `case_runtime_alert` is used.*

---

## 8. Live demo and presentation strategy

Suggested flow aligned with the code:

1. Show `inputs/case_training_space.yaml` (intent).
2. Show `outputs/case_training_space/ir.json` (normalized intent).
3. Show `allocation.json` and `schedule.json` (placement and timing story).
4. Render `topology.mmd` in any Mermaid-capable viewer; show `topology_spec.yaml` beside it.
5. Run `case_inference_earth` or change one YAML field and re-run to show **coherent downstream change**.
6. Run `case_runtime_alert` and contrast `runtime_response.json` and updated topology with the base training case.

**Command (reproducibility):**

```bash
cd iris
python -m src.main --case inputs/case_training_space.yaml
python -m src.main --case inputs/case_runtime_alert.yaml
```

---

## 9. Limitations and future work

The prototype remains a **logical** topology compiler: it does not emit switch configurations, optical control commands, or live Kubernetes manifests. Runtime feedback is **simulated** via YAML. These limits match the honest scope statement in `iris/README.md` and strengthen the report if quoted explicitly.

Future work: richer solvers, formal verification of objectives, richer orbital models, lowering the deployment spec toward real orchestration APIs.

---

## 10. Conclusion

The architectural claim—that topology can be treated as a **compilation target** emerging from structured intent, allocation, scheduling, and cautious runtime adaptation—is implemented in IRIS as a **demonstrable, file-based pipeline**. The code does not replace the theoretical contribution of the full draft (literature integration, full six-layer generality), but it provides **concrete evidence**: versionable inputs, a normalized IR, inspectable rationale, dual topology outputs from a single internal representation, and evaluation metadata suitable for citation in the written report and presentation.

---

## Appendix A: Core pipeline (reference listing)

For verbatim inclusion in an appendix, the end-to-end orchestration is centralized in `run_case()` — see §4.5 for the excerpt that writes IR, allocation, schedule, runtime feedback, topology YAML, Mermaid, and evaluation.

---

## Appendix B: How to cite files in your PDF

Use a consistent naming convention, for example: “Artifact bundle `outputs/case_training_space/` contains …” and reference figures as “Figure 3: excerpt from `topology.mmd` for the space-preferred training case.” This ties every claim about behavior to a path reviewers or instructors can open.
