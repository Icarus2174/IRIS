from __future__ import annotations

from typing import Any, Dict, List

from .models import Allocation, IR, Schedule, Topology


def compile_topology(ir: IR, allocation: Allocation, schedule: Schedule) -> Topology:
    rationale: Dict[str, Any] = {"reasons": []}

    # Nodes derived purely from allocation + auxiliary roles
    nodes: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = [
        {"id": "earth", "label": "Earth"},
        {"id": "orbit", "label": "Orbit"},
        {"id": "storage", "label": "Storage"},
        {"id": "coordination", "label": "Coordination"},
    ]

    # Deterministic ordering
    alloc_clusters = sorted(allocation.selected_clusters, key=lambda x: (x.get("domain", ""), x.get("cluster", "")))
    for a in alloc_clusters:
        nodes.append(
            {
                "id": a["cluster"],
                "label": a["cluster"],
                "group": a["domain"],
                "roles": ["compute"],
                "accelerators": a.get("accelerators", 0),
            }
        )

    coord = allocation.auxiliary_roles.get("coordinator", "coordinator")
    nodes.append({"id": "coordinator", "label": f"coordinator@{coord}", "group": "coordination", "roles": ["coordinator"]})
    nodes.append({"id": "object_store", "label": "earth-object-store", "group": "storage", "roles": ["storage"]})

    # Link selection: packet vs optical vs hybrid, using IR-derived requirements.
    preferred = ir.topology_requirement.get("preferred_link_mode", "any")
    need_dense = bool(ir.topology_requirement.get("need_dense_interconnect"))

    links: List[Dict[str, Any]] = []

    def choose_backbone_link_type() -> str:
        # Conservative + explainable: optical only if high-comm and not strict latency inference.
        if preferred == "packet":
            rationale["reasons"].append("network.preference=packet => packet links")
            return "packet"
        if preferred == "optical":
            rationale["reasons"].append("network.preference=optical => optical links where needed")
            return "optical"
        if preferred == "hybrid":
            if ir.communication_intensity == "high":
                rationale["reasons"].append("hybrid preference + high comm => optical backbone between compute sites")
                return "optical"
            rationale["reasons"].append("hybrid preference + low comm => packet sufficient")
            return "packet"
        # any
        if ir.communication_intensity == "high":
            rationale["reasons"].append("no link preference + high comm => optical backbone")
            return "optical"
        rationale["reasons"].append("no link preference + low comm => packet")
        return "packet"

    backbone_type = choose_backbone_link_type()
    bandwidth_class = "high" if ir.communication_intensity == "high" else "medium"

    compute_nodes = [n for n in nodes if "compute" in n.get("roles", [])]
    compute_nodes = sorted(compute_nodes, key=lambda n: n["id"])

    # Dense interconnect if training/high comm across multiple compute sites
    if need_dense and len(compute_nodes) >= 2:
        rationale["reasons"].append("need_dense_interconnect => meshed compute links")
        for i in range(len(compute_nodes)):
            for j in range(i + 1, len(compute_nodes)):
                links.append(
                    {
                        "src": compute_nodes[i]["id"],
                        "dst": compute_nodes[j]["id"],
                        "type": backbone_type,
                        "bandwidth_class": bandwidth_class,
                        "stability": ir.topology_requirement.get("stability_requirement", "medium"),
                        "reason": "dense interconnect for communication-heavy workload",
                    }
                )
    else:
        # Otherwise connect compute nodes to coordinator (star)
        rationale["reasons"].append("no dense interconnect => coordinator-centered links")
        for n in compute_nodes:
            links.append(
                {
                    "src": n["id"],
                    "dst": "coordinator",
                    "type": "packet",
                    "bandwidth_class": "medium",
                    "stability": ir.topology_requirement.get("stability_requirement", "medium"),
                    "reason": "simple topology for low-comm / latency-focused workload",
                }
            )

    # Storage edges: always packet in this prototype, emphasize for archive.
    storage_bw = "high" if ir.workload_class == "archive" else "medium"
    for n in compute_nodes[: max(1, min(2, len(compute_nodes)))]:
        links.append(
            {
                "src": n["id"],
                "dst": "object_store",
                "type": "packet",
                "bandwidth_class": storage_bw,
                "stability": "high",
                "reason": "storage transfer path",
            }
        )

    # Scheduling dependency info
    rationale["schedule_dependency"] = {
        "topology_required_before_launch": schedule.topology_required_before_launch,
        "schedule_decision": schedule.decision,
    }

    return Topology(case_name=ir.case_name, nodes=nodes, links=links, groups=groups, rationale=rationale)


def topology_to_mermaid(top: Topology) -> str:
    # Deterministic ordering of nodes/links for stable diffs.
    nodes = sorted(top.nodes, key=lambda n: (n.get("group", ""), n.get("id", "")))
    links = sorted(top.links, key=lambda e: (e["src"], e["dst"], e.get("type", "")))

    # Group nodes by group id
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        by_group.setdefault(n.get("group", "ungrouped"), []).append(n)

    def node_line(n: Dict[str, Any]) -> str:
        label = n.get("label", n["id"])
        return f'  {n["id"]}["{label}"]'

    lines: List[str] = []
    lines.append("flowchart LR")
    lines.append("")

    group_order = ["earth", "orbit", "coordination", "storage", "ungrouped"]
    for g in group_order:
        if g not in by_group:
            continue
        label = next((gg["label"] for gg in top.groups if gg["id"] == g), g.capitalize())
        lines.append(f"subgraph {g}[{label}]")
        for n in by_group[g]:
            lines.append(node_line(n))
        lines.append("end")
        lines.append("")

    # Link styling: label includes type + bandwidth class
    for e in links:
        edge_label = f'{e.get("type","packet")}:{e.get("bandwidth_class","med")}'
        lines.append(f'  {e["src"]} -- "{edge_label}" --> {e["dst"]}')

    return "\n".join(lines).strip() + "\n"

