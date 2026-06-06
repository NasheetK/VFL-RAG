#!/usr/bin/env python3
"""End-to-end validation for NIDS architecture overhaul."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.action_config import (
    format_allowed_actions_context,
    get_agent_order,
    get_allowed_actions_for_label,
    load_agentic_features,
    load_attack_options,
)
from utils.rerank_plan_scoring import build_reference_text_for_label, load_scoring_kb

EXPECTED_DEVICES = {"FlowProbe", "Firewall", "CoreRouter"}
LABELS = [
    "BENIGN",
    "DDOS",
    "DOS",
    "SSHPATATOR",
    "FTPPATATOR",
    "PORTSCAN",
    "WEBATTACK",
    "BOT",
    "OTHERS",
]


def main() -> int:
    errors: list[str] = []
    agentic = load_agentic_features()
    order = get_agent_order(agentic)
    if set(order) != EXPECTED_DEVICES:
        errors.append(f"agent_order mismatch: {order}")

    attack_kb, agents_kb = load_scoring_kb(ROOT)
    for label in LABELS:
        ref = build_reference_text_for_label(label, attack_kb, agents_kb, order)
        for dev in order:
            if f"{dev} actions:" not in ref:
                errors.append(f"reference missing {dev} for {label}")
        allowed = get_allowed_actions_for_label(label)
        if not allowed:
            errors.append(f"no allowed_actions for {label}")
        ctx = format_allowed_actions_context(label, order)
        if "No actions" in ctx:
            errors.append(f"empty allowed context for {label}")

    meta_path = ROOT / "model" / "model_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        names = meta.get("agent_names") or []
        if names != order:
            errors.append(f"metadata agent_names {names} != {order}")

    pred_path = ROOT / "RAG_docs" / "predictions" / "sample_predictions.json"
    if pred_path.exists():
        samples = json.loads(pred_path.read_text(encoding="utf-8"))
        if samples:
            shap = samples[0].get("shap_explanation") or {}
            keys = set((shap.get("party_contributions") or {}).keys())
            legacy = keys & {"RAN", "Edge", "Core"}
            if legacy:
                errors.append(f"predictions still use legacy keys: {legacy}")
            if not keys & EXPECTED_DEVICES and not legacy:
                errors.append(f"unexpected SHAP keys: {keys}")

    print("NIDS e2e validation")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return 1
    print("  All checks passed.")
    for label in LABELS[:3]:
        print(f"  [{label}] {build_reference_text_for_label(label, attack_kb, agents_kb, order)[:120]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
