#!/usr/bin/env python3
"""Switch Plan.ipynb to plain-text LLM guidance via utils.plan_prompts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Plan.ipynb"

IMPORT_BLOCK = """
from utils.plan_prompts import (
    LLM_SYSTEM_MESSAGE,
    agent_guidance_to_scoring_text,
    build_rag_context_text,
    call_llm_api,
    create_prompt,
    create_prompt_without_RAG,
    extract_sample_summary,
)
"""

PRINT_COMPACT_OLD = '''def _print_llm_response_compact(llm_response: Optional[Dict[str, Any]]) -> None:
    print("\\n--- LLM response (JSON) ---")
    if not llm_response:
        print("Failed to get LLM response")
        return
    print(json.dumps(llm_response, indent=2, ensure_ascii=False))'''

PRINT_COMPACT_NEW = '''def _print_llm_response_compact(llm_response: Optional[Dict[str, Any]]) -> None:
    print("\\n--- LLM guidance (plain text) ---")
    if not llm_response:
        print("Failed to get LLM response")
        return
    text = agent_guidance_to_scoring_text(llm_response)
    print(text or "(empty)")'''

PRINT_SUMMARY_OLD = '''def _print_llm_response_summary(llm_response: Optional[Dict[str, Any]]) -> None:
    if not llm_response:
        print("Failed to get LLM response")
        return
    print("\\n--- LLM response ---")
    print(f"  Threat level: {llm_response.get('threat_level')}")
    print(f"  Execution priority: {llm_response.get('execution_priority')}")
    print(f"  Primary actions: {llm_response.get('primary_actions', [])}")
    reasoning = (llm_response.get("overall_reasoning") or "")[:500]
    if len(llm_response.get("overall_reasoning") or "") > 500:
        reasoning += "..."
    print(f"  Overall reasoning: {reasoning}")'''

PRINT_SUMMARY_NEW = '''def _print_llm_response_summary(llm_response: Optional[Dict[str, Any]]) -> None:
    if not llm_response:
        print("Failed to get LLM response")
        return
    print("\\n--- LLM guidance ---")
    text = agent_guidance_to_scoring_text(llm_response)
    preview = text[:800] + ("..." if len(text) > 800 else "")
    print(preview)'''


def _drop_function_block(lines: list[str], start_prefix: str, next_def_prefixes: tuple[str, ...]) -> list[str]:
    start = None
    for i, line in enumerate(lines):
        if line.startswith(start_prefix):
            start = i
            break
    if start is None:
        return lines
    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j]
        if s.startswith("def ") and not s.startswith(start_prefix):
            for prefix in next_def_prefixes:
                if s.startswith(prefix) or prefix == "def ":
                    end = j
                    break
            if end != len(lines):
                break
        if s.startswith("def ") and start_prefix.startswith("def create_prompt"):
            end = j
            break
    return lines[:start] + lines[end:]


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cell = nb["cells"][5]
    lines = "".join(cell.get("source", [])).splitlines()
    orig_len = len(lines)

    if "from utils.plan_prompts import" not in lines:
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("_RAG_PARENTS = "):
                insert_at = i + 1
                break
        import_lines = [ln for ln in IMPORT_BLOCK.strip().splitlines()]
        lines = lines[:insert_at] + [""] + import_lines + [""] + lines[insert_at:]

    # Remove duplicate helpers now imported from plan_prompts (keep _TEMPLATE_QUERY)
    for start, stop_before in (
        ("def create_prompt(", ("def _normalize_strategy_to_keys", "def _vprint")),
        ("def build_rag_context_text(", ("def _RERANK_MODE_LABELS", "def _attack_type")),
    ):
        start_idx = None
        for i, line in enumerate(lines):
            if line.startswith(start):
                start_idx = i
                break
        if start_idx is None:
            continue
        end_idx = len(lines)
        for j in range(start_idx + 1, len(lines)):
            if lines[j].startswith("def "):
                end_idx = j
                break
        lines = lines[:start_idx] + lines[end_idx:]

    src = "\n".join(lines)
    src = src.replace(PRINT_COMPACT_OLD, PRINT_COMPACT_NEW)
    src = src.replace(PRINT_SUMMARY_OLD, PRINT_SUMMARY_NEW)
    src = src.replace(
        '"content": "You are a cybersecurity expert. Return only valid JSON."',
        '"content": LLM_SYSTEM_MESSAGE',
    )
    src = src.replace("full LLM JSON.", "full LLM guidance text.")
    src = src.replace("LLM response (JSON)", "LLM guidance")

    cell["source"] = [ln + "\n" for ln in src.splitlines()]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
    cell["outputs"] = []
    cell["execution_count"] = None

    # Header markdown
    for c in nb["cells"]:
        if c.get("cell_type") == "markdown":
            s = "".join(c.get("source", []))
            if "grounded JSON plans" in s:
                c["source"] = [
                    x.replace(
                        "grounded JSON plans (threat level, priorities, actions, `knowledge_sources_used`)",
                        "concise plain-text guidance per NIDS device (FlowProbe, Firewall, CoreRouter)",
                    )
                    for x in c["source"]
                ]

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    new_len = len(src.splitlines())
    print(f"Patched Plan.ipynb cell 5: {orig_len} -> {new_len} lines")


if __name__ == "__main__":
    main()
