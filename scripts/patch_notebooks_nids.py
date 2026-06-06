#!/usr/bin/env python3
"""Replace legacy RAN/Edge/Core strings in notebooks for NIDS device model."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["Train.ipynb", "Predict.ipynb", "Plan.ipynb", "Score.ipynb", "Index.ipynb"]

REPLACEMENTS = [
    ('"RAN", "Edge", "Core"', '"FlowProbe", "Firewall", "CoreRouter"'),
    ("('RAN', 'Edge', 'Core')", "('FlowProbe', 'Firewall', 'CoreRouter')"),
    ('["RAN", "Edge", "Core"]', '["FlowProbe", "Firewall", "CoreRouter"]'),
    ("RAN / Edge / Core", "FlowProbe / Firewall / CoreRouter"),
    ("RAN|Edge|Core", "FlowProbe|Firewall|CoreRouter"),
    ("network tiers (RAN, Edge, Core)", "NIDS devices (FlowProbe, Firewall, CoreRouter)"),
    ("across network tiers (RAN, Edge, Core)", "across NIDS devices (FlowProbe, Firewall, CoreRouter)"),
    ("responder agent (RAN / Edge / Core)", "responder device (FlowProbe / Firewall / CoreRouter)"),
    ('agent_to_actions.get("RAN"', 'agent_to_actions.get("FlowProbe"'),
    ('agent_to_actions.get("Edge"', 'agent_to_actions.get("Firewall"'),
    ('agent_to_actions.get("Core"', 'agent_to_actions.get("CoreRouter"'),
    ('f"RAN actions:', 'f"FlowProbe actions:'),
    ('f"Edge actions:', 'f"Firewall actions:'),
    ('f"Core actions:', 'f"CoreRouter actions:'),
    ("RAN actions:", "FlowProbe actions:"),
    ("Edge actions:", "Firewall actions:"),
    ("Core actions:", "CoreRouter actions:"),
    ('"network_tier": "RAN|Edge|Core"', '"network_tier": "FlowProbe|Firewall|CoreRouter"'),
    ('map to "Core" by policy', 'map to "CoreRouter" by policy'),
    ("RAN tier", "FlowProbe"),
    ("Edge tier", "Firewall"),
    ("Core tier", "CoreRouter"),
    ("Agent 1 (RAN)", "Agent 1 (FlowProbe)"),
    ("Agent 2 (Edge)", "Agent 2 (Firewall)"),
    ("Agent 3 (Core)", "Agent 3 (CoreRouter)"),
    ("(RAN, Edge, Core)", "(FlowProbe, Firewall, CoreRouter)"),
    ("Load agent definitions (RAN, Edge, Core)", "Load agent definitions (FlowProbe, Firewall, CoreRouter)"),
    ("RAN/Edge/Core", "FlowProbe/Firewall/CoreRouter"),
]


def _patch_text(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def patch_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb.get("cells", []):
        src = cell.get("source")
        if src is None:
            continue
        if isinstance(src, list):
            joined = "".join(src)
            new_joined = _patch_text(joined)
            if new_joined != joined:
                cell["source"] = new_joined.splitlines(keepends=True)
                if cell["source"] and not cell["source"][-1].endswith("\n"):
                    pass
                changed = True
        elif isinstance(src, str):
            new_src = _patch_text(src)
            if new_src != src:
                cell["source"] = new_src
                changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return changed


def main() -> None:
    for name in NOTEBOOKS:
        p = ROOT / name
        if p.exists():
            ok = patch_notebook(p)
            print(f"{name}: {'patched' if ok else 'unchanged'}")


if __name__ == "__main__":
    main()
