# Research influence mapping (prototype rationale)

This prototype **does not implement** the cited systems. It uses their ideas as *inspiration* to structure a believable intent-to-topology compiler pipeline.

## Intent front end + lowering pipeline

- **Merlin / Propane / Aura / DCM**: motivate the idea that users express **policy/intent**, and the system **compiles** it into concrete decisions and a normalized internal form.

## Allocation backend

- **DCM / COPE**: motivate translating goals + constraints into **resource placement/allocation decisions** with explicit rationale.

## Scheduling backend

- **CASSINI / Tiresias**: motivate that “when to run” decisions can be **communication-aware** (network/topology readiness) and incorporate **queueing/risk/fairness-like** heuristics (kept simple here).

## Topology compiler backend

- **Helios**: motivates hybrid thinking: **packet vs optical** (in this prototype: typed links and conservative selection rules).

## Runtime feedback + conservatism

- **ATAR-like ideas**: motivate **runtime caution** (confidence-aware conservative re-evaluation) that can trigger re-scheduling and/or topology downgrades.

## Cross-cutting influences

- **Pollux / Gavel**: motivate adaptivity over time and awareness of **hardware heterogeneity** (accelerator types and fit matter; allocation may be revisited after runtime events).

