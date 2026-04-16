from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .models import ParsedIntent
from .utils import canonicalize, load_yaml


_JOB_TYPE_MAP = {
    "train": "training",
    "training": "training",
    "finetune": "training",
    "infer": "inference",
    "inference": "inference",
    "serve": "inference",
    "archive": "archive",
    "checkpoint": "archive",
    "backup": "archive",
}

_LATENCY_MAP = {
    "strict": "strict",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "none": "none",
    "insensitive": "none",
}

_ENERGY_MAP = {
    "lowest_energy": "lowest_energy",
    "min_energy": "lowest_energy",
    "solar_first": "lowest_energy",
    "solar": "lowest_energy",
    "low_cost": "low_cost",
    "cheap": "low_cost",
    "balanced": "balanced",
    "performance": "performance",
}

_RELIABILITY_MAP = {"high": "high", "medium": "medium", "low": "low"}
_PLACEMENT_MAP = {
    "earth": "earth",
    "space": "space",
    "space_preferred": "space",
    "prefer_space": "space",
    "orbit": "space",
    "prefer_orbit": "space",
    "hybrid": "hybrid",
    "any": "any",
}
_NETWORK_MAP = {"packet": "packet", "optical": "optical", "hybrid": "hybrid", "any": "any"}
_ACCEL_MAP = {"gpu": "gpu", "tpu": "tpu", "none": "none", "null": "none"}

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def _validate_required(d: Dict[str, Any], paths: List[str]) -> List[str]:
    missing = []
    for p in paths:
        cur: Any = d
        ok = True
        for part in p.split("."):
            if not isinstance(cur, dict) or part not in cur:
                ok = False
                break
            cur = cur[part]
        if not ok:
            missing.append(p)
    return missing


def _apply_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(raw)
    d.setdefault("objectives", [])
    d.setdefault("constraints", {})
    d.setdefault("placement", {})
    d.setdefault("network", {})
    d.setdefault("runtime_policy", {})
    d.setdefault("job", {})
    d["job"].setdefault("accelerators", {"count": 0, "type": "none"})
    d["job"].setdefault("hardware", {})
    d["job"]["hardware"].setdefault("min_memory_gb", 8)
    d["constraints"].setdefault("latency_sensitivity", "medium")
    d["constraints"].setdefault("energy_preference", "balanced")
    d["constraints"].setdefault("reliability", "medium")
    d["placement"].setdefault("preference", "any")
    d["network"].setdefault("preference", "any")
    d["runtime_policy"].setdefault("conservative", False)
    d["runtime_policy"].setdefault("allow_hybrid", True)
    return d


def load_case_yaml(case_path: str | Path) -> Dict[str, Any]:
    raw = load_yaml(case_path)
    if "inherits" in raw:
        base = raw["inherits"].get("from")
        if not base:
            raise ValueError("inherits.from must be provided when using inherits")
        base_path = Path(case_path).parent / base
        base_raw = load_yaml(base_path)
        overlay = {k: v for k, v in raw.items() if k != "inherits"}
        return _deep_merge(base_raw, overlay)
    return raw


def parse_intent(case_path: str | Path) -> Tuple[ParsedIntent, Dict[str, Any]]:
    raw_case = load_case_yaml(case_path)
    raw_case = _apply_defaults(raw_case)

    missing = _validate_required(raw_case, ["case.name", "job.type", "job.accelerators.count", "job.accelerators.type"])
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    case_name = str(raw_case["case"]["name"])

    job_type = canonicalize(raw_case["job"]["type"], _JOB_TYPE_MAP)
    accel_type = canonicalize(raw_case["job"]["accelerators"]["type"], _ACCEL_MAP)
    accel_count = int(raw_case["job"]["accelerators"]["count"])
    min_mem = int(raw_case["job"]["hardware"]["min_memory_gb"])

    latency = canonicalize(raw_case["constraints"]["latency_sensitivity"], _LATENCY_MAP)
    energy = canonicalize(raw_case["constraints"]["energy_preference"], _ENERGY_MAP)
    reliability = canonicalize(raw_case["constraints"]["reliability"], _RELIABILITY_MAP)
    placement = canonicalize(raw_case["placement"]["preference"], _PLACEMENT_MAP)
    network = canonicalize(raw_case["network"]["preference"], _NETWORK_MAP)

    objectives = list(raw_case.get("objectives") or [])
    if not isinstance(objectives, list):
        raise ValueError("objectives must be a list")
    objectives = [str(o) for o in objectives]

    explanations: Dict[str, str] = {}
    if raw_case["job"]["type"] != job_type:
        explanations["job.type"] = f"normalized '{raw_case['job']['type']}' -> '{job_type}'"
    if raw_case["constraints"]["latency_sensitivity"] != latency:
        explanations["constraints.latency_sensitivity"] = (
            f"normalized '{raw_case['constraints']['latency_sensitivity']}' -> '{latency}'"
        )

    if job_type == "inference" and accel_count == 0:
        raise ValueError("inference jobs must request at least 1 accelerator in this prototype")
    if accel_count < 0:
        raise ValueError("accelerator count must be non-negative")
    if accel_type == "none" and accel_count > 0:
        raise ValueError("accelerator type cannot be 'none' when count > 0")

    parsed = ParsedIntent(
        case_name=case_name,
        job_type=job_type,  # type: ignore[arg-type]
        accelerator_count=accel_count,
        accelerator_type=accel_type,  # type: ignore[arg-type]
        min_memory_gb=min_mem,
        objectives=objectives,
        latency_sensitivity=latency,  # type: ignore[arg-type]
        energy_preference=energy,  # type: ignore[arg-type]
        reliability=reliability,  # type: ignore[arg-type]
        placement_preference=placement,  # type: ignore[arg-type]
        network_preference=network,  # type: ignore[arg-type]
        runtime_policy=dict(raw_case.get("runtime_policy") or {}),
        raw_source=raw_case,
        explanations=explanations,
    )

    runtime_event = raw_case.get("runtime_event")
    return parsed, (runtime_event if isinstance(runtime_event, dict) else {})

