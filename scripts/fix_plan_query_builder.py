#!/usr/bin/env python3
"""Fix Plan.ipynb query builder to use NIDS device ids."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "Plan.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

OLD_TIER_PCT = '''    tier_contributions_pct = {
        "RAN": float(party_pct.get("RAN", 0.0) or 0.0) * 100.0,
        "Edge": float(party_pct.get("Edge", 0.0) or 0.0) * 100.0,
        "Core": float(party_pct.get("Core", 0.0) or 0.0) * 100.0,
    }'''

NEW_TIER_PCT = '''    tier_contributions_pct = {
        "FlowProbe": float(party_pct.get("FlowProbe", party_pct.get("RAN", 0.0)) or 0.0) * 100.0,
        "Firewall": float(party_pct.get("Firewall", party_pct.get("Edge", 0.0)) or 0.0) * 100.0,
        "CoreRouter": float(party_pct.get("CoreRouter", party_pct.get("Core", 0.0)) or 0.0) * 100.0,
    }'''

OLD_TOP = '''        "RAN": _top_features_for_tier(sample, "RAN", top_n=3),
        "Edge": _top_features_for_tier(sample, "Edge", top_n=3),
        "Core": _top_features_for_tier(sample, "Core", top_n=3),'''

NEW_TOP = '''        "FlowProbe": _top_features_for_tier(sample, "FlowProbe", top_n=3),
        "Firewall": _top_features_for_tier(sample, "Firewall", top_n=3),
        "CoreRouter": _top_features_for_tier(sample, "CoreRouter", top_n=3),'''

OLD_TEMPLATE = '''    "Tier contributions: RAN {ran_pct:.0f}%, Edge {edge_pct:.0f}%, Core {core_pct:.0f}%. "
    "Top indicators per tier: RAN: {ran_feats}. Edge: {edge_feats}. Core: {core_feats}. "'''

NEW_TEMPLATE = '''    "Device contributions: FlowProbe {ran_pct:.0f}%, Firewall {edge_pct:.0f}%, CoreRouter {core_pct:.0f}%. "
    "Top indicators per device: FlowProbe: {ran_feats}. Firewall: {edge_feats}. CoreRouter: {core_feats}. "'''

OLD_FMT = '''        ran_pct=float(tier_pct.get("RAN", 0.0) or 0.0),
        edge_pct=float(tier_pct.get("Edge", 0.0) or 0.0),
        core_pct=float(tier_pct.get("Core", 0.0) or 0.0),
        ran_feats=fmt(s["top_features"].get("RAN", [])),
        edge_feats=fmt(s["top_features"].get("Edge", [])),
        core_feats=fmt(s["top_features"].get("Core", [])),'''

NEW_FMT = '''        ran_pct=float(tier_pct.get("FlowProbe", tier_pct.get("RAN", 0.0)) or 0.0),
        edge_pct=float(tier_pct.get("Firewall", tier_pct.get("Edge", 0.0)) or 0.0),
        core_pct=float(tier_pct.get("CoreRouter", tier_pct.get("Core", 0.0)) or 0.0),
        ran_feats=fmt(s["top_features"].get("FlowProbe", s["top_features"].get("RAN", []))),
        edge_feats=fmt(s["top_features"].get("Firewall", s["top_features"].get("Edge", []))),
        core_feats=fmt(s["top_features"].get("CoreRouter", s["top_features"].get("Core", []))),'''

OLD_DOC = '    """Deterministic top-N feature names for a tier (RAN/Edge/Core)."""'
NEW_DOC = '    """Deterministic top-N feature names for a NIDS device (FlowProbe/Firewall/CoreRouter)."""'

REPLACEMENTS = [
    (OLD_TIER_PCT, NEW_TIER_PCT),
    (OLD_TOP, NEW_TOP),
    (OLD_TEMPLATE, NEW_TEMPLATE),
    (OLD_FMT, NEW_FMT),
    (OLD_DOC, NEW_DOC),
    ('tier_contributions_pct = {\n        "RAN"', 'tier_contributions_pct = {\n        "FlowProbe"'),
]

changed = False
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = cell.get("source")
    text = "".join(src) if isinstance(src, list) else (src or "")
    new_text = text
    for old, new in REPLACEMENTS:
        if old in new_text:
            new_text = new_text.replace(old, new)
    if new_text != text:
        cell["source"] = new_text.splitlines(keepends=True)
        changed = True

if changed:
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Plan.ipynb query builder updated")
else:
    print("No changes applied (patterns not found)")
