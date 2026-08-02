#!/usr/bin/env python3
"""Point Score.ipynb at utils.rerank_plan_scoring for v2 attack_options."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "Score.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

OLD_BLOCK = '''    label = label.strip().upper()
    actions = attack_options_kb.get(label) or attack_options_kb.get("OTHERS", [])
    actions_norm = [_norm_action(a) for a in actions]

    # Assign each action to agents that can execute it (can be multiple).
    agent_to_actions = defaultdict(list)
    unassigned = []
    for a_raw, a in zip(actions, actions_norm):
        capable_agents = action_to_agents.get(a, [])
        if not capable_agents:
            # If you want STRICT no-unassigned, you can drop it or map to "CoreRouter" by policy.
            unassigned.append(a_raw)
            continue

        # Add canonical + synonyms into the agent bucket to maximize token overlap.
        expanded = [a_raw]
        expanded += ACTION_SYNONYMS.get(a, [])
        for ag in capable_agents:
            agent_to_actions[ag].extend(expanded)

    # Stable ordering for reproducible scores
    def uniq(seq):
        seen = set()
        out = []
        for x in seq:
            k = _norm_action(x)
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    ran = ", ".join(uniq(agent_to_actions.get("FlowProbe", []))) or "none"
    edge = ", ".join(uniq(agent_to_actions.get("Firewall", []))) or "none"
    core = ", ".join(uniq(agent_to_actions.get("CoreRouter", []))) or "none"

    # One compact canonical string your metrics will compare against
    ref = (
        f"{label} response: "
        f"FlowProbe actions: {ran}. "
        f"Firewall actions: {edge}. "
        f"CoreRouter actions: {core}."
    )

    if unassigned:
        ref += f" Unassigned actions: {', '.join(unassigned)}."

    return ref'''

NEW_BLOCK = '''    from utils.action_config import get_agent_order, load_agentic_features
    from utils.rerank_plan_scoring import build_reference_text_for_label as _build_ref

    return _build_ref(
        label.strip().upper(),
        attack_options_kb,
        agents_kb,
        get_agent_order(load_agentic_features()),
    )'''

changed = False
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if OLD_BLOCK in text:
        cell["source"] = text.replace(OLD_BLOCK, NEW_BLOCK).splitlines(keepends=True)
        changed = True
        break

if changed:
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Score.ipynb build_reference_text_for_label delegates to utils")
else:
    print("Score.ipynb: block not found")
