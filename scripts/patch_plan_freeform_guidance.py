#!/usr/bin/env python3
"""Patch Plan.ipynb for free-form agent_guidance prompts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Plan.ipynb"


def patch_cell_source(src: str) -> str:
    if "def create_prompt(" in src and "format_allowed_actions_context" in src:
        marker_start = "def create_prompt("
        marker_end = "def create_prompt_without_RAG"
        i0 = src.index(marker_start)
        i1 = src.index(marker_end)
        replacement = '''def create_prompt(
    sample_data: Dict[str, Any],
    rag_results: List[Dict[str, Any]],
    attack_actions_data: Optional[Dict[str, Any]] = None,
    agentic_features_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Free-form per-device guidance; attack_actions_data ignored."""
    from utils.plan_prompts import create_prompt as _create_prompt

    return _create_prompt(
        sample_data, rag_results, attack_actions_data, agentic_features_data
    )


'''
        src = src[:i0] + replacement + src[i1:]

    if "def create_prompt_without_RAG" in src and "publicly_available_online_sources" in src:
        i0 = src.index("def create_prompt_without_RAG")
        i1 = src.index("def call_llm_api")
        replacement = '''def create_prompt_without_RAG(
    sample_data: Dict[str, Any],
    agentic_features_data: Optional[Dict[str, Any]] = None,
) -> str:
    from utils.plan_prompts import create_prompt_without_RAG as _create

    return _create(sample_data, agentic_features_data)


'''
        src = src[:i0] + replacement + src[i1:]

    if "def build_rag_context_text(" in src:
        i0 = src.index("def build_rag_context_text(")
        i1 = src.index("def _print_pipeline_prompt_artifacts")
        replacement = '''def build_rag_context_text(
    rag_results: List[Dict[str, Any]],
    *,
    max_sections: int = 10,
) -> str:
    from utils.plan_prompts import build_rag_context_text as _build

    return _build(rag_results, max_sections=max_sections)


'''
        src = src[:i0] + replacement + src[i1:]

    if "def call_llm_api(prompt: str)" in src:
        old = '''def call_llm_api(prompt: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return None

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "You are a cybersecurity expert. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    response_text = response.choices[0].message.content.strip()
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response_text[start:end])
    except Exception as e:
        print(f"Warning: Could not parse JSON: {e}")

    return {
        "threat_level": "Unknown",
        "primary_actions": ["Unable to parse response"],
        "supporting_actions": [],
        "reasoning": response_text[:200],
        "execution_priority": "Standard",
        "knowledge_sources_used": [],
    }'''
        new = '''def call_llm_api(prompt: str) -> Optional[Dict[str, Any]]:
    from utils.plan_prompts import LLM_SYSTEM_MESSAGE, parse_llm_json_response

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return None

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": LLM_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    response_text = (response.choices[0].message.content or "").strip()
    parsed = parse_llm_json_response(response_text)
    if parsed.get("parse_error"):
        print(f"Warning: Could not parse agent_guidance JSON: {parsed.get('parse_error')}")
    return parsed'''
        if old in src:
            src = src.replace(old, new)

    if "def _print_llm_response_summary" in src:
        old = '''def _print_llm_response_summary(llm_response: Optional[Dict[str, Any]]) -> None:
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
        new = '''def _print_llm_response_summary(llm_response: Optional[Dict[str, Any]]) -> None:
    if not llm_response:
        print("Failed to get LLM response")
        return
    print("\\n--- LLM response (per-device guidance) ---")
    guidance = llm_response.get("agent_guidance") or {}
    if isinstance(guidance, dict):
        for dev in ("FlowProbe", "Firewall", "CoreRouter"):
            para = (guidance.get(dev) or "").strip()
            preview = para[:400] + ("..." if len(para) > 400 else "")
            print(f"\\n  [{dev}]\\n  {preview}")
    if llm_response.get("parse_error"):
        print(f"\\n  Parse error: {llm_response.get('parse_error')}")'''
        if old in src:
            src = src.replace(old, new)

    if '_vprint("You are a cybersecurity expert. Return only valid JSON.")' in src:
        src = src.replace(
            '_vprint("You are a cybersecurity expert. Return only valid JSON.")',
            '_vprint("You are a cybersecurity expert advising NIDS operators. Return only valid JSON with agent_guidance.")',
        )

    if "prompt = create_prompt(\n                sample, rag_results, attack_actions_data, agentic_features_data" in src:
        src = src.replace(
            "prompt = create_prompt(\n                sample, rag_results, attack_actions_data, agentic_features_data\n            )",
            "prompt = create_prompt(\n                sample, rag_results, None, agentic_features_data\n            )",
        )
    if "prompt = create_prompt(\n        sample, rag_results, attack_actions_data, agentic_features_data\n    )" in src:
        src = src.replace(
            "prompt = create_prompt(\n        sample, rag_results, attack_actions_data, agentic_features_data\n    )",
            "prompt = create_prompt(\n        sample, rag_results, None, agentic_features_data\n    )",
        )

    return src


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def create_prompt(" not in src and "def call_llm_api" not in src:
            continue
        new_src = patch_cell_source(src)
        if new_src != src:
            cell["source"] = [line + "\n" for line in new_src.splitlines()]
            if cell["source"] and not cell["source"][-1].endswith("\n"):
                cell["source"][-1] += "\n"
            changed += 1
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Patched {changed} cell(s) in Plan.ipynb")


if __name__ == "__main__":
    main()
