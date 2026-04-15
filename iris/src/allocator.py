from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .models import Allocation, Cluster, IR


def _inventory() -> List[Cluster]:
    # Deterministic, static "prototype" inventory.
    return [
        Cluster(
            id="earth-a100-west",
            domain="earth",
            accelerators_total=128,
            accelerators_type="gpu",
            memory_gb_per_accelerator=80,
            energy_profile="grid",
            link_confidence=0.98,
            notes="Earth DC with strong packet fabric",
        ),
        Cluster(
            id="earth-h100-east",
            domain="earth",
            accelerators_total=64,
            accelerators_type="gpu",
            memory_gb_per_accelerator=80,
            energy_profile="grid",
            link_confidence=0.97,
            notes="Earth DC with low latency to users",
        ),
        Cluster(
            id="orbit-leo-solar-1",
            domain="orbit",
            accelerators_total=96,
            accelerators_type="gpu",
            memory_gb_per_accelerator=40,
            energy_profile="solar",
            link_confidence=0.72,
            notes="LEO compute pod with intermittent backhaul",
        ),
        Cluster(
            id="orbit-leo-solar-2",
            domain="orbit",
            accelerators_total=64,
            accelerators_type="gpu",
            memory_gb_per_accelerator=40,
            energy_profile="solar",
            link_confidence=0.62,
            notes="LEO compute pod; lower confidence links",
        ),
    ]


def allocate(ir: IR) -> Allocation:
    inv = _inventory()
    req_count = int(ir.hardware_requirement["accelerators"]["count"])
    req_type = ir.hardware_requirement["accelerators"]["type"]
    min_mem = int(ir.hardware_requirement["min_memory_gb"])

    rationale: Dict[str, Any] = {"scoring": [], "rejected": []}

    def feasible(c: Cluster) -> Tuple[bool, str]:
        if req_type == "none":
            return True, "no accelerators required"
        if c.accelerators_type != req_type:
            return False, "accelerator type mismatch"
        if c.memory_gb_per_accelerator < min_mem:
            return False, "insufficient memory per accelerator"
        return True, "ok"

    # Score clusters for selection
    scored: List[Tuple[float, Cluster, str]] = []
    for c in inv:
        ok, why = feasible(c)
        if not ok:
            rationale["rejected"].append({"cluster": c.id, "reason": why})
            continue

        score = 0.0
        reasons = []

        # Domain preference
        if c.domain == "orbit" and "orbit" in ir.placement_candidates:
            score += 2.0
            reasons.append("orbit is a placement candidate")
        if c.domain == "earth" and "earth" in ir.placement_candidates:
            score += 1.5
            reasons.append("earth is a placement candidate")

        # Latency: strict workloads prefer earth
        if ir.latency_class == "strict":
            if c.domain == "earth":
                score += 4.0
                reasons.append("strict latency prefers earth")
            else:
                score -= 5.0
                reasons.append("strict latency penalizes orbit")

        # Energy: orbit solar favored for lowest_energy when orbit candidate
        if ir.space_constraints.get("prefer_orbit_for_energy") and c.domain == "orbit":
            score += 2.5
            reasons.append("energy preference favors orbit solar")

        # Communication-heavy training prefers high-confidence links, and fewer sites
        if ir.communication_intensity == "high":
            score += 2.0 * c.link_confidence
            reasons.append("high-comm workload favors higher link confidence")

        # Conservative orbit admission gate (soft scored here; hard-checked later)
        min_conf = float(ir.space_constraints.get("min_link_confidence_for_orbit", 0.6))
        if c.domain == "orbit" and c.link_confidence < min_conf:
            score -= 3.0
            reasons.append(f"orbit link confidence below threshold {min_conf:.2f}")

        scored.append((score, c, "; ".join(reasons)))
        rationale["scoring"].append({"cluster": c.id, "score": round(score, 3), "reasons": reasons})

    scored.sort(key=lambda t: (-t[0], t[1].id))

    # Decide placement mode
    # Note: hybrid admissibility is enforced upstream (intent -> IR) via placement candidates.

    selected: List[Dict[str, Any]] = []
    remaining = req_count

    def take_from(cluster: Cluster, count: int, reason: str) -> None:
        nonlocal remaining
        if count <= 0:
            return
        selected.append(
            {
                "cluster": cluster.id,
                "domain": cluster.domain,
                "accelerators": count,
                "reason": reason,
                "cluster_link_confidence": cluster.link_confidence,
            }
        )
        remaining -= count

    # Allocation heuristic:
    # - strict latency -> earth-only (single best earth cluster)
    # - high-comm training -> prefer packing into as few clusters as possible
    # - archive/no-accel -> choose orbit if candidates allow and confidence ok, else earth
    if req_type == "none" or req_count == 0:
        # pick a "compute+storage coordinator" site
        preferred_domain = "orbit" if "orbit" in ir.placement_candidates else "earth"
        best = next((c for _, c, _ in scored if c.domain == preferred_domain), None) or scored[0][1]
        placement = "orbit" if best.domain == "orbit" else "earth"
        packed = True
        auxiliary = {"coordinator": best.id, "storage": "earth-object-store"}
        rationale["decision"] = {
            "reason": "no accelerators requested; selected coordinator site deterministically",
            "chosen": best.id,
        }
        return Allocation(
            case_name=ir.case_name,
            selected_clusters=selected,
            placement=placement,  # type: ignore[arg-type]
            packed=packed,
            auxiliary_roles=auxiliary,
            rationale=rationale,
        )

    if ir.latency_class == "strict":
        earths = [t for t in scored if t[1].domain == "earth"]
        if not earths:
            raise ValueError("No feasible earth clusters for strict latency workload")
        best = earths[0][1]
        take_from(best, min(best.accelerators_total, remaining), "strict latency: allocate earth-local")
        placement = "earth"
        packed = True
        auxiliary = {"coordinator": best.id, "storage": "earth-object-store"}
    else:
        # prefer orbit first if candidate and confidence ok
        min_conf = float(ir.space_constraints.get("min_link_confidence_for_orbit", 0.6))
        orbit_first = "orbit" in ir.placement_candidates and ir.latency_class != "strict"
        ordered = []
        if orbit_first:
            ordered.extend([t for t in scored if t[1].domain == "orbit"])
            ordered.extend([t for t in scored if t[1].domain == "earth"])
        else:
            ordered = scored[:]

        for _, c, why in ordered:
            if remaining <= 0:
                break
            if c.domain == "orbit" and c.link_confidence < min_conf and ir.fallback_sensitivity in ("high", "medium"):
                rationale["rejected"].append(
                    {"cluster": c.id, "reason": f"orbit confidence {c.link_confidence:.2f} below threshold {min_conf:.2f}"}
                )
                continue
            # Pack if high comm intensity: fill one cluster before splitting
            alloc = min(c.accelerators_total, remaining)
            take_from(c, alloc, f"selected by score: {why}")

        domains = {s["domain"] for s in selected}
        if domains == {"orbit"}:
            placement = "orbit"
        elif domains == {"earth"}:
            placement = "earth"
        else:
            placement = "hybrid"
        packed = ir.communication_intensity == "high"
        auxiliary = {"coordinator": selected[0]["cluster"] if selected else "earth-a100-west", "storage": "earth-object-store"}

    if remaining > 0:
        raise ValueError(f"Insufficient accelerators available. Need {req_count}, allocated {req_count-remaining}.")

    rationale["summary"] = {
        "requested_accelerators": req_count,
        "allocated_accelerators": req_count - remaining,
        "placement": placement,
        "packed_preference": packed,
    }

    return Allocation(
        case_name=ir.case_name,
        selected_clusters=selected,
        placement=placement,  # type: ignore[arg-type]
        packed=packed,
        auxiliary_roles=auxiliary,
        rationale=rationale,
    )

