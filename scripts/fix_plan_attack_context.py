#!/usr/bin/env python3
"""Update Plan.ipynb prompt builder for attack_options v2 and agent capabilities."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "Plan.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

OLD = '''    attack_actions_context = ""
    if attack_actions_data and "attacks" in attack_actions_data:
        attack_type = predicted_label.upper()
        if attack_type in attack_actions_data["attacks"]:
            recommended_actions = attack_actions_data["attacks"][attack_type]
            attack_actions_context = (
                f"\\n\\nAttack-Specific Recommended Actions (from attack_options.json):\\n"
                f"For {predicted_label} attack, recommended actions: {', '.join(recommended_actions)}\\n"
            )
        else:
            attack_actions_context = (
                f"\\n\\nAttack-Specific Actions: No specific recommendations for {predicted_label}.\\n"
            )
'''

NEW = '''    from utils.action_config import format_allowed_actions_context

    attack_actions_context = ""
    if attack_actions_data and "attacks" in attack_actions_data:
        attack_actions_context = (
            f"\\n\\nAllowed actions (from attack_options.json):\\n"
            f"{format_allowed_actions_context(predicted_label, attack_opts=attack_actions_data)}\\n"
        )
'''

OLD_AGENTIC = '''            "\\n\\nAgentic Features and Actions by Network Tier (from agentic_features.json):\\n"
            "Use evidence from all tiers (contributions + top features) to set action priority and sequencing.\\n"
        )
        tf = condensed_evidence.get("top_features_by_tier", {}) or {}
        tier_pct = condensed_evidence.get("tier_contributions_pct", {}) or {}
        for tier in ["FlowProbe", "Firewall", "CoreRouter"]:
            if tier in agentic_features_data:
                tier_data = agentic_features_data[tier]
                actions = tier_data.get("actions", [])
                tier_feats = tf.get(tier, []) if isinstance(tf, dict) else []
                pct = float(tier_pct.get(tier, 0.0) or 0.0)
                agentic_context += f"\\n{tier} Network Tier:\\n"
                agentic_context += f"  - Tier contribution: {pct:.1f}%\\n"
                agentic_context += f"  - Top evidence features (top 3): {', '.join(tier_feats) if tier_feats else 'none'}\\n"
                agentic_context += f"  - Allowed tier actions: {', '.join(actions)}\\n"
'''

NEW_AGENTIC = '''            "\\n\\nAgentic features by NIDS device (from agentic_features.json):\\n"
            "Use evidence from all devices (contributions + top features) to set action priority and sequencing.\\n"
        )
        tf = condensed_evidence.get("top_features_by_tier", {}) or {}
        tier_pct = condensed_evidence.get("tier_contributions_pct", {}) or {}
        for tier in ["FlowProbe", "Firewall", "CoreRouter"]:
            if tier in agentic_features_data:
                tier_data = agentic_features_data[tier]
                actions = tier_data.get("action_capabilities", tier_data.get("actions", []))
                tier_feats = tf.get(tier, []) if isinstance(tf, dict) else []
                pct = float(tier_pct.get(tier, 0.0) or 0.0)
                agentic_context += f"\\n{tier}:\\n"
                agentic_context += f"  - Contribution: {pct:.1f}%\\n"
                agentic_context += f"  - Top evidence features (top 3): {', '.join(tier_feats) if tier_feats else 'none'}\\n"
                agentic_context += f"  - Device action capabilities: {', '.join(actions)}\\n"
'''

changed = False
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    new_text = text
    if OLD in new_text:
        new_text = new_text.replace(OLD, NEW)
        changed = True
    if OLD_AGENTIC in new_text:
        new_text = new_text.replace(OLD_AGENTIC, NEW_AGENTIC)
        changed = True
    if new_text != text:
        cell["source"] = new_text.splitlines(keepends=True)

if changed:
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Plan.ipynb attack/agentic context updated")
else:
    print("Plan.ipynb: no matching blocks")
