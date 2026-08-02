#!/usr/bin/env python3
"""Rename legacy RAN/Edge/Core keys to FlowProbe/Firewall/CoreRouter in JSON artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.action_config import LEGACY_DEVICE_MAP, load_agentic_features

DEVICE_REPLACEMENTS = {
    **LEGACY_DEVICE_MAP,
    "RAN": "FlowProbe",
    "Edge": "Firewall",
    "Core": "CoreRouter",
}
TIER_WORD_REPLACEMENTS = [
    (r"\bRAN tier\b", "FlowProbe"),
    (r"\bEdge tier\b", "Firewall"),
    (r"\bCore tier\b", "CoreRouter"),
    (r"\bRAN\b", "FlowProbe"),
    (r"\bEdge\b", "Firewall"),
    (r"\bCore\b", "CoreRouter"),
]


def _rename_dict_keys(obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.items():
        nk = DEVICE_REPLACEMENTS.get(k, k)
        out[nk] = v
    return out


def _migrate_shap_block(shap: dict[str, Any]) -> None:
    if not shap:
        return
    for key in ("party_contributions", "party_contributions_pct", "feature_contributions"):
        block = shap.get(key)
        if isinstance(block, dict):
            shap[key] = _rename_dict_keys(block)
    dom = shap.get("dominant_agent")
    if dom in DEVICE_REPLACEMENTS:
        shap["dominant_agent"] = DEVICE_REPLACEMENTS[dom]


def _migrate_action_plan(plan: dict[str, Any]) -> None:
    for section in ("primary_actions", "supporting_actions"):
        for item in plan.get(section) or []:
            if not isinstance(item, dict):
                continue
            tier = item.get("network_tier") or item.get("network_device")
            if tier in DEVICE_REPLACEMENTS:
                item["network_tier"] = DEVICE_REPLACEMENTS[tier]
                item["network_device"] = DEVICE_REPLACEMENTS[tier]
    reasoning = plan.get("overall_reasoning") or ""
    if isinstance(reasoning, str):
        s = reasoning
        for pat, repl in TIER_WORD_REPLACEMENTS:
            s = re.sub(pat, repl, s)
        plan["overall_reasoning"] = s


def migrate_sample(sample: dict[str, Any]) -> None:
    shap = sample.get("shap_explanation")
    if isinstance(shap, dict):
        _migrate_shap_block(shap)
    for plan_key in ("action_plan", "action_plan_rag", "action_plan_no_rag"):
        plan = sample.get(plan_key)
        if isinstance(plan, dict):
            _migrate_action_plan(plan)


def migrate_predictions_dir(pred_dir: Path) -> int:
    count = 0
    for fp in pred_dir.glob("*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        for sample in data:
            if isinstance(sample, dict):
                migrate_sample(sample)
                count += 1
        fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return count


def migrate_model_metadata(meta_path: Path) -> None:
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    agentic = load_agentic_features()
    order = agentic.get("agent_order") or list(DEVICE_REPLACEMENTS.values())
    agents = agentic.get("agents") or {}
    meta["agent_names"] = order
    meta["agent_domains"] = [
        (agents.get(n) or {}).get("description", n) for n in order
    ]
    meta["agent_actions"] = [
        ", ".join((agents.get(n) or {}).get("action_capabilities") or [])
        for n in order
    ]
    meta["schema_version"] = agentic.get("schema_version", "2.0")
    meta["nids_device_map"] = DEVICE_REPLACEMENTS
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    pred_dir = ROOT / "RAG_docs" / "predictions"
    n = migrate_predictions_dir(pred_dir)
    migrate_model_metadata(ROOT / "model" / "model_metadata.json")
    print(f"Migrated {n} prediction sample(s) and model_metadata.json")


if __name__ == "__main__":
    main()
