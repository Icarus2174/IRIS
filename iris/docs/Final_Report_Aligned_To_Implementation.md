# IRIS: Intent-Resolved Infrastructure Synthesis (Implementation-Aligned Report)

## Abstract

Future AI infrastructure is unlikely to remain confined to a single, static terrestrial datacenter. As training, inference, and data-management workloads become larger and more heterogeneous, operators must reason jointly about workload intent, hardware placement, scheduling, and network topology. Prior systems address parts of this problem in isolation: Helios showed that datacenter connectivity can respond to demand through hybrid switching; Merlin, Propane, and Aura showed that high-level network intent can be compiled into enforceable behavior; DCM and COPE showed that cluster management can be driven by declarative policy and constraint solving; Gavel, Tiresias, Pollux, and CASSINI showed that accelerator heterogeneity, job uncertainty, and network sensitivity materially affect AI workload performance. None of these systems treats topology as a machine-generated artifact produced jointly with allocation and scheduling from structured workload intent.

IRIS addresses that gap with a deterministic intent-to-topology pipeline: parse YAML intent → lower to a shared intermediate representation (IR) → allocate from a fixed inventory → schedule launch behavior → compile topology into Mermaid + YAML → optionally apply conservative runtime feedback (reschedule + packet downgrade + topology recompilation) → emit lightweight evaluation checks. The prototype is implemented in Python and produces reproducible artifacts for four included cases (training, inference, archive, runtime alert). In the current prototype, “space-specific” behavior is represented via static orbit-cluster link-confidence values in inventory and simulated runtime degradation events, rather than a physically computed orbital geometry or weather model.

---

## 1. Introduction

Large AI workloads stress interconnect design and placement strategy as much as raw compute capacity. Training workloads generate sustained, communication-heavy traffic. Inference workloads impose tight latency bounds. Checkpointing and data movement introduce large but delay-tolerant transfers. In terrestrial datacenters, these pressures already expose the limits of static topology assumptions and manually coordinated orchestration. In hybrid Earth-and-orbit infrastructure, the problem becomes harder still.

This report takes the position that topology should be treated as a compilation target. The operator specifies structured computational intent—e.g., a distributed training job that prefers orbit-capable resources, or an Earth-local inference service with strict latency sensitivity. IRIS compiles that intent into a placement plan, a schedule decision, and a logical topology artifact that is both human-readable (Mermaid) and machine-readable (YAML).

**Prototype scope note (implementation-aligned).** The prototype does not compute orbital passes, elevation angles, weather degradation, or link budgets. Instead, orbit clusters in a fixed inventory carry static `link_confidence` values (e.g., `0.72`, `0.62`) that influence allocation scoring and conservative gating. Runtime feedback is simulated by injecting a runtime event via YAML.

---

## 2. Background and Related Work

This report builds on four relevant lines of prior work.

**Reconfigurable datacenter topologies.** Helios proposed a hybrid electrical and optical switch architecture in which packet switching handles bursty traffic while optical circuits serve heavier demand [CITATION]. Its core contribution is establishing that connectivity can respond to workload demand rather than remain fixed. IRIS inherits that motivation but focuses on generating a logical topology artifact from intent, rather than designing a switching fabric.

**Intent compilation and network synthesis.** Merlin showed that operators can express desired network behavior declaratively and compile it into enforceable policies [CITATION]. Propane and Aura similarly compile high-level routing intent into distributed configurations [CITATION]. IRIS adopts the separation of intent from realization, but compiles into a topology artifact produced jointly with allocation and scheduling.

**Declarative resource orchestration.** DCM and COPE show that orchestration can be driven by declarative specifications and constraint optimization [CITATION]. IRIS borrows this “structured intent drives resource decisions” framing, but uses heuristic allocation and scheduling appropriate for a course prototype.

**AI-cluster scheduling under heterogeneity and uncertainty.** Gavel, Tiresias, Pollux, and CASSINI demonstrate that heterogeneity, uncertainty, and network effects materially influence scheduling outcomes [CITATION]. These systems motivate IRIS’s explicit IR fields (latency class, communication intensity, fallback sensitivity) and its conservative runtime feedback path.

**Gap.** These systems do not produce a standalone topology artifact compiled jointly with placement and scheduling decisions from structured AI workload intent.

---

## 3. Problem Statement and Design Goals

Given a structured description of an AI workload and system constraints, produce a placement plan, a schedule decision, and a logical topology that are mutually consistent, machine-readable, and revisable under runtime events. In terrestrial settings, pieces of this problem are often solved separately: schedulers assume a topology; topologies are pre-designed; runtime changes are handled ad hoc. IRIS treats topology as an output of the same decision chain that yields placement and schedule.

Design goals:

- **Structured input**: accept declarative YAML intent, not free-form text.
- **Shared IR**: preserve intent through a normalized IR to prevent inter-stage drift.
- **Topology as output**: compile topology from IR + allocation + schedule.
- **Conservative runtime adaptation**: reschedule and downgrade topology preferences under runtime instability via explicit recompilation.
- **Inspectable artifacts**: emit intermediate and final artifacts per case for reproducibility and review.

---

## 4. Implementation Model for “Space-Specific” Constraints (Prototype)

IRIS’s “space” dimension is represented in the current prototype by:

- **Static inventory fields**: orbit clusters have fixed `link_confidence` values (e.g., `0.72`, `0.62`) and domain labels (`earth`, `orbit`).
- **A conservative confidence threshold**: the IR contains `space_constraints.min_link_confidence_for_orbit`, set to `0.45` by default, or `0.60` if `runtime_policy.conservative` is enabled.
- **Simulated runtime events**: a runtime YAML can inject `runtime_event.type = orbital_link_confidence_low`, which triggers conservative adaptation (reschedule + packet downgrade + topology recompilation).

**Non-goals (not implemented).** The prototype does not compute:

- satellite ephemeris, pass windows, or elevation angle,
- weather degradation factors,
- a continuous orbital confidence function \(C(\theta, w)\),
- confidence sweeps that induce phase transitions in placement or link mode.

Those are valid extensions for future work, but they should be treated as conceptual/proposed rather than implemented behavior.

---

## 5. Architecture

IRIS is a deterministic pipeline with an optional feedback loop:

parse YAML intent → normalize to IR → allocate → schedule → compile topology → apply feedback (optional) → evaluate invariants.

### 5.1 Layer 1: Intent Front End

The front end loads YAML, applies defaults, validates required fields, and canonicalizes synonyms (e.g., `space_preferred` → `space`). Output: `parsed_intent.json`.

### 5.2 Layer 2: Intermediate Representation (IR)

The IR captures workload class, communication intensity, latency class, placement candidates, topology requirement (including `preferred_link_mode`), scheduling urgency, fallback sensitivity, and `space_constraints`. Output: `ir.json`.

### 5.3 Layer 3: Resource Allocation Backend

Allocation scores clusters from a static inventory. Strict-latency workloads are penalized heavily for orbit placement. Orbit candidates below the confidence threshold may be rejected for medium/high fallback sensitivity. Output: `allocation.json`.

### 5.4 Layer 4: Scheduling Backend

Scheduling chooses a launch decision such as `launch_immediately`, `wait_for_topology`, or (under runtime risk) `delay_due_to_runtime_risk`. Output: `schedule.json`.

### 5.5 Layer 5: Topology Compiler Backend

Topology compilation constructs a single internal topology object (nodes, links, groups, rationale), then emits:

- `topology.mmd` (Mermaid, Diagram-as-Code)
- `topology_spec.yaml` (machine-readable spec)

Both outputs are generated from the same topology object to prevent drift.

### 5.6 Layer 6: Runtime Feedback (Conservative)

When runtime events occur (simulated via YAML), the feedback layer may:

- reschedule the workload, and
- downgrade the IR’s `preferred_link_mode` to `packet`, and
- recompile topology from the revised state.

**Implementation-aligned behavior (important).** The current prototype does **not** re-run allocation and therefore does **not** change placement under runtime events. It changes schedule and topology mode while keeping the already-chosen placement.

---

## 6. Evaluation (Artifact-Based)

The prototype ships four cases under `iris/inputs/` and emits artifacts under `iris/outputs/<case_name>/`.

### 6.1 Case Studies

- **Training (space-preferred)**: `inputs/case_training_space.yaml`
- **Inference (earth-local)**: `inputs/case_inference_earth.yaml`
- **Archive (delay-tolerant)**: `inputs/case_archive_orbit.yaml`
- **Runtime alert**: `inputs/case_runtime_alert.yaml` (inherits training intent, injects a runtime event)

### 6.2 Allocation Scoring (Implementation Evidence)

Allocation output reflects inventory names and scoring. Notably, the inventory contains `earth-a100-west` and `earth-h100-east` (not `earth-cpu-east`).

**Table 1. Clusters present in the prototype inventory (from `allocation.json`).**

| Domain | Cluster ID |
| ------ | ---------- |
| earth  | `earth-a100-west` |
| earth  | `earth-h100-east` |
| orbit  | `orbit-leo-solar-1` |
| orbit  | `orbit-leo-solar-2` |

**Qualitative alignment.** Inference is forced toward Earth by strict latency penalties on orbit. Training is scored to prefer orbit (when orbit is a candidate) and to favor higher `link_confidence` under high communication intensity.

### 6.3 Topology Structures and Metrics (From Emitted Specs)

Rather than presenting “designed” topology sizes, we report topology metrics directly from the generated `topology_spec.yaml` artifacts.

**Table 2. Topology metrics from `outputs/*/topology_spec.yaml`.**

| Case | Nodes | Links | Groups | Dominant backbone mode |
| ---- | ----- | ----- | ------ | ---------------------- |
| Training (`case_training_space`) | 4 | 3 | 4 | optical (compute backbone) |
| Inference (`case_inference_earth`) | 3 | 2 | 4 | packet |
| Archive (`case_archive_orbit`) | 2 | 0 | 4 | — |
| Runtime alert (post-feedback, `case_runtime_alert`) | 4 | 3 | 4 | packet (compute backbone downgraded) |

**Interpretation.**

- Training produces a dense-enough *logical* backbone between compute sites (one backbone link) plus storage-transfer links.
- Inference produces a simple coordinator-centered topology (compute → coordinator and compute → object store).
- Archive produces no explicit links in the emitted spec (coordinator and object store nodes only), reflecting the prototype’s “storage-transfer oriented” simplification.

### 6.4 Runtime Feedback: Before vs After (Implementation-Aligned)

The runtime-alert case injects a runtime event (`orbital_link_confidence_low`). The feedback layer reschedules conservatively and downgrades preferred link mode to packet, then recompiles topology.

**Table 3. Runtime alert behavior (what the code actually does).**

| Property | Pre-feedback (training) | Post-feedback (`case_runtime_alert`) |
| -------- | ------------------------ | ------------------------------------ |
| Placement | `orbit` | `orbit` (allocation not rerun) |
| Schedule decision | `wait_for_topology` | `delay_due_to_runtime_risk` |
| Compute backbone link type | `optical` | `packet` |
| Topology recompilation | — | `true` (via `runtime_response.json`) |
| Rescheduled | — | `true` (via `runtime_response.json`) |

**Correct summary sentence.** In the runtime-alert case, IRIS does not reallocate the workload to Earth; instead, it conservatively delays launch, downgrades the preferred interconnect to packet mode, and recompiles an orbit-based topology from the revised state.

### 6.5 Evaluation Checks

Each case emits `evaluation.json` containing lightweight internal consistency checks (e.g., strict-latency inference not placed in orbit, allocation does not exceed request, topology style matches low vs high communication intensity, runtime feedback visibility flags). These checks are internal to the system and serve as artifact-level sanity checks rather than independent validation.

---

## 7. Discussion

The prototype supports three defensible claims:

- **End-to-end coherence**: one deterministic pipeline connects intent → IR → allocation → schedule → topology → feedback → evaluation.
- **Explainable allocation**: `allocation.json` records per-cluster scores and reasons, making decisions inspectable.
- **Conservative runtime adaptation**: simulated runtime events trigger rescheduling and topology recompilation with a packet downgrade, and the resulting topology artifacts reflect that change.

**What the prototype does not support (and should not claim).**

- A physically computed orbital geometry or weather-based confidence function.
- A confidence sweep / phase transition analysis that changes placement or backbone mode based on continuous confidence values.
- Runtime-triggered reallocation from orbit to Earth (unless implemented as future work).

---

## 8. Limitations and Future Work

Limitations:

- Heuristic allocation and scheduling (no solver-backed optimization).
- Static inventory with fixed link-confidence values (no geometric orbital model).
- Runtime feedback does not re-run allocation, so placement does not change under degradation.
- No automated quantitative table generation beyond emitted artifacts (node/link counts are available from generated specs, but the system does not currently benchmark itself or run parameter sweeps).

Future work:

- Implement an explicit orbital confidence estimator (ephemeris + elevation + weather) and feed it into allocation/scheduling/feedback.
- Add optional feedback-time reallocation under severe degradation (re-run allocation when confidence drops below threshold and fallback sensitivity is high).
- Add a small regression test suite and a metrics script to compute table values directly from `outputs/` to prevent report drift.

---

## 9. Conclusion

IRIS demonstrates a coherent “topology as compilation target” architecture: structured intent is normalized into a shared IR, used to drive allocation and scheduling, and compiled into a topology emitted in both Mermaid and YAML from a single internal representation. Runtime feedback, while simulated, is real in the sense that it causes explicit rescheduling and topology recompilation with conservative packet downgrade. The result is a small, readable prototype whose claims can be grounded directly in reproducible artifacts—provided the report stays strictly aligned with what the code actually implements.

