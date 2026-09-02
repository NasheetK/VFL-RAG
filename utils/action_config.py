"""
Load NIDS agent ontology and per-attack action configuration from RAG_docs JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENTIC_PATH = _REPO_ROOT / "RAG_docs" / "agentic_features.json"
DEFAULT_ATTACK_OPTIONS_PATH = _REPO_ROOT / "RAG_docs" / "attack_options.json"

LEGACY_DEVICE_MAP = {
    "RAN": "FlowProbe",
    "Edge": "Firewall",
    "Core": "CoreRouter",
}


def load_agentic_features(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_AGENTIC_PATH
    if not p.exists():
        raise FileNotFoundError(f"agentic_features.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def get_agent_order(raw: dict[str, Any] | None = None) -> list[str]:
    data = raw if raw is not None else load_agentic_features()
    order = data.get("agent_order")
    if order:
        return list(order)
    agents = data.get("agents") or {}
    legacy = ["RAN", "Edge", "Core"]
    if any(k in agents for k in legacy):
        return [n for n in legacy if n in agents]
    return list(agents.keys())


def load_attack_options(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_ATTACK_OPTIONS_PATH
    if not p.exists():
        raise FileNotFoundError(f"attack_options.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def get_attack_entry(label: str, attack_opts: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = attack_opts if attack_opts is not None else load_attack_options()
    attacks = opts.get("attacks") or {}
    key = (label or "").strip().upper()
    if key in attacks:
        return attacks[key]
    return attacks.get("OTHERS", {})


def get_allowed_actions_for_label(label: str, attack_opts: dict[str, Any] | None = None) -> list[str]:
    entry = get_attack_entry(label, attack_opts)
    allowed = entry.get("allowed_actions")
    if allowed:
        return list(allowed)
    devices = entry.get("devices") or {}
    seen: list[str] = []
    for acts in devices.values():
        for a in acts or []:
            if a not in seen:
                seen.append(a)
    return seen


def get_device_actions_for_label(
    label: str,
    device: str,
    attack_opts: dict[str, Any] | None = None,
) -> list[str]:
    entry = get_attack_entry(label, attack_opts)
    devices = entry.get("devices") or {}
    return list(devices.get(device) or [])


def format_allowed_actions_context(
    label: str,
    agent_order: list[str] | None = None,
    attack_opts: dict[str, Any] | None = None,
) -> str:
    opts = attack_opts if attack_opts is not None else load_attack_options()
    order = agent_order or get_agent_order()
    entry = get_attack_entry(label, opts)
    devices = entry.get("devices") or {}
    lines = []
    for dev in order:
        acts = devices.get(dev) or []
        if acts:
            lines.append(f"- {dev}: {', '.join(acts)}")
    allowed = entry.get("allowed_actions") or []
    if allowed:
        lines.append(f"- Global allowed: {', '.join(allowed)}")
    return "\n".join(lines) if lines else "No actions configured."


def load_scoring_kb(
    attack_options_path: Path | str | None = None,
    agentic_path: Path | str | None = None,
) -> dict[str, Any]:
    """Knowledge bundle for rerank_plan_scoring.build_reference_text_for_label."""
    attack_opts = load_attack_options(attack_options_path)
    agentic = load_agentic_features(agentic_path)
    return {
        "schema_version": attack_opts.get("schema_version", "2.0"),
        "agent_order": get_agent_order(agentic),
        "attacks": attack_opts.get("attacks") or {},
        "agents": agentic.get("agents") or {},
        "legacy_agent_map": agentic.get("legacy_agent_map") or LEGACY_DEVICE_MAP,
    }


def normalize_device_name(name: str, legacy_map: dict[str, str] | None = None) -> str:
    if not name:
        return name
    m = legacy_map or LEGACY_DEVICE_MAP
    return m.get(name, name)
