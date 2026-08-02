#!/usr/bin/env python3
"""Patch Score.ipynb docs for description-based reference + agent_guidance scoring text."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Score.ipynb"

REPLACEMENTS = [
    (
        "a **label-derived reference string** built deterministically from `attack_options.json` (actions per attack label) and `agentic_features.json` (which agents can execute which actions).",
        "a **label-derived reference string** built from `agentic_features.json` (NIDS device roles and descriptions per FlowProbe / Firewall / CoreRouter), without fixed action lists.",
    ),
    (
        "- **Context utilization** (`Util_SBert` / `Util_CRAG`): SBERT cosine and CRAG-style proxy between the top-10 retrieved context and the LLM's `overall_reasoning`. Quantifies how much of the retrieved evidence the LLM actually grounds its answer in (faithfulness / answer-vs-context alignment). Reuses the same adaptive CRAG thresholds as the context-grounding metric.",
        "- **Context utilization** (`Util_SBert` / `Util_CRAG`): SBERT cosine and CRAG-style proxy between the top-10 retrieved context and the LLM's per-device guidance text (concatenated `agent_guidance` paragraphs). Quantifies how much of the retrieved evidence the LLM actually grounds its answer in (faithfulness / answer-vs-context alignment). Reuses the same adaptive CRAG thresholds as the context-grounding metric.",
    ),
    (
        "- **LLM reasoning quality** (`LLM_SBert` / `LLM_CRAG` / `BertscoreF1`): SBERT cosine, CRAG proxy, and `BERTScore-F1` between `overall_reasoning` and the same reference string.",
        "- **LLM reasoning quality** (`LLM_SBert` / `LLM_CRAG` / `BertscoreF1`): SBERT cosine, CRAG proxy, and `BERTScore-F1` between concatenated `agent_guidance` text and the same reference string.",
    ),
    (
        "- Candidate `overall_reasoning` (LLM output): text being evaluated against the reference for the LLM-quality metrics.",
        "- Candidate scoring text (LLM output): concatenated `agent_guidance` paragraphs (stored in the `overall_reasoning` column for compatibility).",
    ),
    (
        "- BERTScore (`roberta-large`): contextual token-level similarity between reference and `overall_reasoning`; reported as F1.",
        "- BERTScore (`roberta-large`): contextual token-level similarity between reference and concatenated agent guidance; reported as F1.",
    ),
    (
        "The reference is generated per sample by `build_reference_text_for_label(predicted_label)` from `attack_options.json` + `agentic_features.json`. It lists, per responder agent (FlowProbe / Firewall / CoreRouter), the subset of the attack's action options that each agent's `action_capabilities` can execute, plus any unassigned actions. Example (PORTSCAN): `PORTSCAN response: FlowProbe actions: block IP, limit rate, update ACL. Firewall actions: block IP, harden ports. CoreRouter actions: update ACL, scan threshold. Unassigned actions: tarpit scan.`",
        "The reference is generated per sample by `build_reference_text_for_label(predicted_label)` from `agentic_features.json` device descriptions. Example (PORTSCAN): `PORTSCAN incident — recommend coordinated response across NIDS devices. FlowProbe: ... Firewall: ... CoreRouter: ...`",
    ),
    (
        "- `build_reference_text_for_label(label)`: builds the per-sample reference string from `attack_options.json` + `agentic_features.json`. Lists the attack's action options grouped by the agent (FlowProbe / Firewall / CoreRouter) capable of executing each action. Quality of this string defines the metric ceiling.",
        "- `build_reference_text_for_label(label)`: builds the per-sample reference string from `agentic_features.json` (device role descriptions). Quality of this string defines the metric ceiling.",
    ),
    (
        '    """Extract overall_reasoning from an action plan."""\n',
        '    """Extract scoring text from agent_guidance (legacy overall_reasoning fallback)."""\n',
    ),
    (
        "    overall_reasoning = action_plan.get(\"overall_reasoning\", \"\").strip()\n",
        "    from utils.plan_prompts import agent_guidance_to_scoring_text\n",
    ),
]

# Second part of extract_overall_reasoning patch - need ordered application
EXTRACT_FN_OLD = '''def extract_overall_reasoning(action_plan: Dict[str, Any]) -> str:
    """Extract scoring text from agent_guidance (legacy overall_reasoning fallback)."""
    if not isinstance(action_plan, dict):
        return ""
    from utils.plan_prompts import agent_guidance_to_scoring_text

    overall_reasoning = action_plan.get("overall_reasoning", "").strip()
    return overall_reasoning if overall_reasoning else ""'''

EXTRACT_FN_NEW = '''def extract_overall_reasoning(action_plan: Dict[str, Any]) -> str:
    """Extract scoring text from agent_guidance (legacy overall_reasoning fallback)."""
    from utils.plan_prompts import agent_guidance_to_scoring_text

    if not isinstance(action_plan, dict):
        return ""
    text = agent_guidance_to_scoring_text(action_plan)
    if text:
        return text
    return (action_plan.get("overall_reasoning") or action_plan.get("reasoning") or "").strip()'''


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        orig = src
        for old, new in REPLACEMENTS:
            src = src.replace(old, new)
        if "def extract_overall_reasoning" in src and "agent_guidance_to_scoring_text" in src:
            # Fix botched double replacement
            if EXTRACT_FN_OLD.split("\n")[0] in src:
                pass
            old_block = '''def extract_overall_reasoning(action_plan: Dict[str, Any]) -> str:
    """Extract scoring text from agent_guidance (legacy overall_reasoning fallback)."""
    if not isinstance(action_plan, dict):
        return ""
    from utils.plan_prompts import agent_guidance_to_scoring_text

    overall_reasoning = action_plan.get("overall_reasoning", "").strip()
    return overall_reasoning if overall_reasoning else ""'''
            if old_block in src:
                src = src.replace(old_block, EXTRACT_FN_NEW)
        if src != orig:
            cell["source"] = [line + "\n" for line in src.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            changed += 1
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Patched {changed} cell(s) in Score.ipynb")


if __name__ == "__main__":
    main()
