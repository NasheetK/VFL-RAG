"""
LLM prompts and response helpers for Plan.ipynb action guidance.

Plain-text, concise per-device recommendations (FlowProbe, Firewall, CoreRouter).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.action_config import get_agent_order

DEFAULT_NIDS_DEVICES = ["FlowProbe", "Firewall", "CoreRouter"]
LEGACY_TIER_ALIASES = {"RAN": "FlowProbe", "Edge": "Firewall", "Core": "CoreRouter"}

LLM_SYSTEM_MESSAGE = (
    "You are a cybersecurity NIDS advisor. Respond in plain, concise prose only — "
    "never JSON, markdown code fences, or structured schemas."
)


def _top_features_for_tier(
    sample: Dict[str, Any],
    tier: str,
    *,
    top_n: int = 3,
    score_key: str = "pct_contribution",
) -> List[str]:
    shap_expl = sample.get("shap_explanation", {}) or {}
    feat_contribs = shap_expl.get("feature_contributions", {}) or {}
    feats = feat_contribs.get(tier, {}) or {}

    scored: List[Tuple[str, float]] = []
    if isinstance(feats, dict):
        for feat_name, meta in feats.items():
            if not isinstance(meta, dict):
                continue
            score = float(meta.get(score_key, 0.0) or 0.0)
            if score > 0.0:
                scored.append((feat_name, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:top_n]]


def extract_sample_summary(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize prediction fields used by query builders and prompts."""
    label = (sample.get("predicted_label") or sample.get("true_label") or "UNKNOWN").strip().upper()
    confidence = float(sample.get("confidence", 0.0) or 0.0)

    shap_expl = sample.get("shap_explanation", {}) or {}
    dominant_agent = (shap_expl.get("dominant_agent") or "").strip()
    dominant_tier = (
        dominant_agent
        if dominant_agent in DEFAULT_NIDS_DEVICES
        else LEGACY_TIER_ALIASES.get(dominant_agent, dominant_agent or "Unknown")
    )
    dominant_pct = float(shap_expl.get("dominant_contribution_pct", 0.0) or 0.0) * 100.0

    party_pct = shap_expl.get("party_contributions_pct", {}) or {}
    tier_contributions_pct = {
        "FlowProbe": float(party_pct.get("FlowProbe", party_pct.get("RAN", 0.0)) or 0.0) * 100.0,
        "Firewall": float(party_pct.get("Firewall", party_pct.get("Edge", 0.0)) or 0.0) * 100.0,
        "CoreRouter": float(party_pct.get("CoreRouter", party_pct.get("Core", 0.0)) or 0.0) * 100.0,
    }

    top_features = {
        "FlowProbe": _top_features_for_tier(sample, "FlowProbe", top_n=3),
        "Firewall": _top_features_for_tier(sample, "Firewall", top_n=3),
        "CoreRouter": _top_features_for_tier(sample, "CoreRouter", top_n=3),
    }

    return {
        "label": label,
        "confidence": confidence,
        "dominant_tier": dominant_tier,
        "dominant_pct": dominant_pct,
        "tier_contributions_pct": tier_contributions_pct,
        "top_features": top_features,
    }


def build_condensed_evidence(
    sample: Dict[str, Any],
    *,
    dominant_tier: str = "",
    dominant_pct: float = 0.0,
) -> Dict[str, Any]:
    s = extract_sample_summary(sample)
    return {
        "predicted_label": str(s.get("label") or sample.get("predicted_label", "UNKNOWN")),
        "confidence": float(s.get("confidence", sample.get("confidence", 0.0)) or 0.0),
        "dominant_tier": s.get("dominant_tier", dominant_tier),
        "dominant_contribution_pct": float(s.get("dominant_pct", dominant_pct) or 0.0),
        "tier_contributions_pct": s.get("tier_contributions_pct", {}),
        "top_features_by_tier": s.get("top_features", {}),
    }


def format_network_agents_context(
    agentic_features_data: Optional[Dict[str, Any]],
    *,
    agent_order: Optional[List[str]] = None,
) -> str:
    """Device identity: observation point + role in the network."""
    if not agentic_features_data:
        return "No NIDS device metadata available."

    order = agent_order or get_agent_order(agentic_features_data)
    agents = agentic_features_data.get("agents") or agentic_features_data

    lines = [
        "This network has three NIDS observation/enforcement points:",
    ]
    for dev in order:
        tier_data = agents.get(dev) if isinstance(agents, dict) else None
        if not isinstance(tier_data, dict):
            if dev in agentic_features_data and isinstance(agentic_features_data.get(dev), dict):
                tier_data = agentic_features_data[dev]
            else:
                continue
        desc = (tier_data.get("description") or "").strip()
        obs = (tier_data.get("observation_point") or "").strip()
        lines.append(f"- {dev}:")
        if obs:
            lines.append(f"    Where it sits: {obs}")
        if desc:
            lines.append(f"    Role: {desc}")

    return "\n".join(lines) if len(lines) > 1 else "No NIDS device metadata available."


def format_inference_decision_context(
    sample_data: Dict[str, Any],
    *,
    dominant_tier: str = "",
    dominant_pct: float = 0.0,
) -> str:
    """Human-readable inference decision and SHAP contribution summary."""
    evidence = build_condensed_evidence(
        sample_data, dominant_tier=dominant_tier, dominant_pct=dominant_pct
    )
    label = evidence["predicted_label"]
    conf = float(evidence["confidence"])
    dom = evidence["dominant_tier"]
    dom_pct = float(evidence["dominant_contribution_pct"])
    tier_pct = evidence["tier_contributions_pct"] or {}
    top_feats = evidence["top_features_by_tier"] or {}

    lines = [
        f"- Predicted incident type: {label} ({conf:.1%} model confidence)",
        f"- Primary SHAP contributor: {dom} ({dom_pct:.1f}% of total attribution)",
        "- SHAP contribution by device:",
    ]
    for dev in DEFAULT_NIDS_DEVICES:
        pct = float(tier_pct.get(dev, 0.0) or 0.0)
        feats = top_feats.get(dev, []) if isinstance(top_feats, dict) else []
        feat_str = ", ".join(feats[:3]) if feats else "none highlighted"
        lines.append(f"    {dev}: {pct:.1f}% — top features: {feat_str}")

    return "\n".join(lines)


def build_rag_context_text(
    rag_results: List[Dict[str, Any]],
    *,
    max_sections: int = 10,
) -> str:
    top = (rag_results or [])[: int(max_sections)]
    if not top:
        return "No relevant knowledge-base passages were retrieved for this query."
    parts = ["The following passages were retrieved from the knowledge base:\n"]
    for idx, result in enumerate(top, 1):
        title = result.get("title", "Unknown")
        src = result.get("source_file") or ""
        src_note = f" (source: {src})" if src else ""
        parts.append(f"\n[{idx}] {title}{src_note}\n")
        parts.append(f"{result.get('text', '')}\n")
    return "".join(parts)


def create_action_guidance_prompt(
    sample_data: Dict[str, Any],
    rag_results: List[Dict[str, Any]],
    agentic_features_data: Optional[Dict[str, Any]] = None,
    *,
    include_rag: bool = True,
    dominant_tier: str = "",
    dominant_pct: float = 0.0,
) -> str:
    """Build the user prompt for concise plain-text per-device guidance."""
    sample_id = sample_data.get("sample_id", 0)
    network_context = format_network_agents_context(agentic_features_data)
    inference_context = format_inference_decision_context(
        sample_data, dominant_tier=dominant_tier, dominant_pct=dominant_pct
    )
    rag_context = (
        build_rag_context_text(rag_results)
        if include_rag
        else "Knowledge-base retrieval was not used for this run."
    )

    return f"""You advise operators of a three-device NIDS deployment (FlowProbe, Firewall, CoreRouter).

NETWORK STRUCTURE
{network_context}

INFERENCE DECISION (sample_id={sample_id})
{inference_context}

KNOWLEDGE BASE
{rag_context}

TASK
Write concise operational guidance for this {inference_context.splitlines()[0].replace('- Predicted incident type: ', '').split(' (')[0]} incident.
For each device — FlowProbe, Firewall, and CoreRouter — state what operators should do in 1–3 short sentences.
Ground each recommendation in the inference decision above and in retrieved knowledge when it supports the action.
Weight urgency by model confidence. Be direct; no filler.

OUTPUT FORMAT
Plain text only. No JSON, no markdown code blocks, no nested schemas.
Use exactly three labeled paragraphs in this order:

FlowProbe: <concise guidance>

Firewall: <concise guidance>

CoreRouter: <concise guidance>

Keep the full response under 250 words."""


def create_prompt(
    sample_data: Dict[str, Any],
    rag_results: List[Dict[str, Any]],
    attack_actions_data: Optional[Dict[str, Any]] = None,
    agentic_features_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Legacy signature; attack_actions_data is ignored."""
    from utils.rag_utils import get_dominant_party_info

    _ = attack_actions_data
    _, dominant_tier, dominant_pct = get_dominant_party_info(sample_data)
    return create_action_guidance_prompt(
        sample_data,
        rag_results,
        agentic_features_data,
        include_rag=True,
        dominant_tier=dominant_tier,
        dominant_pct=dominant_pct,
    )


def create_prompt_without_RAG(
    sample_data: Dict[str, Any],
    agentic_features_data: Optional[Dict[str, Any]] = None,
) -> str:
    from utils.rag_utils import get_dominant_party_info

    _, dominant_tier, dominant_pct = get_dominant_party_info(sample_data)
    return create_action_guidance_prompt(
        sample_data,
        [],
        agentic_features_data,
        include_rag=False,
        dominant_tier=dominant_tier,
        dominant_pct=dominant_pct,
    )


def parse_llm_text_response(response_text: str) -> Dict[str, Any]:
    """Wrap plain-text LLM output for JSON export and Score.ipynb."""
    text = (response_text or "").strip()
    if not text:
        return {"guidance_text": "", "format": "plain_text", "parse_error": "empty response"}
    return {"guidance_text": text, "format": "plain_text"}


def parse_llm_json_response(response_text: str) -> Dict[str, Any]:
    """Backward-compatible alias; prefers plain text, falls back to legacy JSON."""
    txt = (response_text or "").strip()
    if txt.startswith("{") and "agent_guidance" in txt:
        try:
            start = txt.find("{")
            end = txt.rfind("}") + 1
            obj = json.loads(txt[start:end])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return parse_llm_text_response(txt)


def call_llm_api(prompt: str) -> Optional[Dict[str, Any]]:
    """Call OpenAI chat API; return plain-text guidance wrapped in a dict."""
    from openai import OpenAI

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
    return parse_llm_text_response(response_text)


def validate_agent_guidance_response(obj: Any) -> Tuple[bool, str]:
    text = agent_guidance_to_scoring_text(obj if isinstance(obj, dict) else None)
    if not text.strip():
        return False, "Empty guidance text"
    return True, ""


def agent_guidance_to_scoring_text(
    action_plan: Optional[Dict[str, Any]],
    agent_order: Optional[List[str]] = None,
) -> str:
    """Text used by Score.ipynb for SBERT / ROUGE / BERTScore."""
    if not action_plan or not isinstance(action_plan, dict):
        return ""

    plain = action_plan.get("guidance_text")
    if isinstance(plain, str) and plain.strip():
        return plain.strip()

    order = agent_order or DEFAULT_NIDS_DEVICES
    guidance = action_plan.get("agent_guidance")
    if isinstance(guidance, dict) and guidance:
        parts: List[str] = []
        for dev in order:
            text = guidance.get(dev) or guidance.get(LEGACY_TIER_ALIASES.get(dev, ""))
            if isinstance(text, str) and text.strip():
                parts.append(f"{dev}: {text.strip()}")
        if parts:
            return "\n\n".join(parts)

    legacy = (action_plan.get("overall_reasoning") or action_plan.get("reasoning") or "").strip()
    return legacy


def build_reference_text_for_label(
    label: str,
    agents_kb: Dict[str, Any],
    agent_order: Optional[List[str]] = None,
    *,
    attack_options_kb: Optional[Dict[str, Any]] = None,
) -> str:
    _ = attack_options_kb
    label_u = (label or "UNKNOWN").strip().upper()
    order = agent_order or get_agent_order({"agents": agents_kb})

    parts: List[str] = []
    for dev in order:
        info = agents_kb.get(dev) or {}
        if not isinstance(info, dict):
            continue
        desc = (info.get("description") or "").strip()
        obs = (info.get("observation_point") or "").strip()
        segment = desc or obs or "no description"
        if obs and desc:
            segment = f"{desc} (observes at {obs})"
        parts.append(f"{dev}: {segment}")

    body = ". ".join(parts)
    return (
        f"{label_u} incident — recommend coordinated response across NIDS devices. "
        f"{body}."
    )


def _normalize_guidance_words(text: str) -> Tuple[str, ...]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return tuple(sorted(set(words)))


def guidance_signature(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not plan or not isinstance(plan, dict):
        return {"missing": True}
    text = agent_guidance_to_scoring_text(plan)
    if not text.strip():
        return {"missing": True, "empty": True}
    return {
        "missing": False,
        "word_set": _normalize_guidance_words(text),
        "char_len": len(text),
    }
