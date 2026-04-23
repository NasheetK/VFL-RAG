"""Score action-plan ``overall_reasoning`` (rerank comparison JSON) vs label-derived reference text.

Aligns with ``Score.ipynb``: same reference builder (attack_options + agent tiers) and BLEU / ROUGE-1 /
SBERT cosine / BERTScore.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Same synonym map as Score.ipynb (keep in sync if you tune matching) ---
ACTION_SYNONYMS: Dict[str, List[str]] = {
    "rate limiting": ["limit rate", "rate limit"],
    "traffic scrubbing": ["enable scrubbing", "scrub traffic"],
    "blackhole routing": ["blackhole route", "blackhole"],
    "ip blocking": ["block ip", "ip block"],
    "acl update": ["update acl", "acl change"],
    "syn cookies": ["enable syn cookies", "syn cookie"],
    "connection limiting": ["limit connections", "conn limit"],
    "account lockout": ["lock account", "lockout"],
    "mfa enforce": ["enforce mfa", "require mfa"],
    "credential throttle": ["throttle login", "login throttle"],
    "fail2ban block": ["fail2ban", "ban ip"],
    "waf rules": ["waf rule", "apply waf"],
    "virtual patching": ["virtual patch", "hot patch"],
    "isolate service": ["isolate", "quarantine service"],
    "captcha challenge": ["captcha", "challenge"],
    "reputation filter": ["reputation", "filter reputation"],
    "js challenge": ["js challenge", "javascript challenge"],
    "tarpitting": ["tarpit", "slow scan"],
    "port hardening": ["harden ports", "close ports"],
    "scan threshold": ["threshold scan", "scan threshold"],
    "auto scale": ["autoscale", "scale out"],
    "log only": ["log", "logging"],
    "monitor": ["monitoring", "observe"],
}


def _norm_action(s: str) -> str:
    return " ".join(s.lower().strip().split())


def load_scoring_kb(
    repo_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root = repo_root or Path(__file__).resolve().parents[1]
    attack_path = root / "RAG_docs/attack_options.json"
    agents_path = root / "RAG_docs/agentic_features.json"
    with open(attack_path, "r", encoding="utf-8") as f:
        attack_options_kb = json.load(f)["attacks"]
    with open(agents_path, "r", encoding="utf-8") as f:
        agents_kb = json.load(f)["agents"]
    return attack_options_kb, agents_kb


def build_reference_text_for_label(
    label: str,
    attack_options_kb: Dict[str, Any],
    agents_kb: Dict[str, Any],
) -> str:
    label_u = label.strip().upper()
    actions = attack_options_kb.get(label_u) or attack_options_kb.get("OTHERS", [])
    actions_norm = [_norm_action(a) for a in actions]

    action_to_agents: Dict[str, List[str]] = defaultdict(list)
    for agent_name, agent_info in agents_kb.items():
        for a in agent_info.get("action_capabilities", []):
            action_to_agents[_norm_action(a)].append(agent_name)

    agent_to_actions: Dict[str, List[str]] = defaultdict(list)
    unassigned: List[str] = []
    for a_raw, a in zip(actions, actions_norm):
        capable_agents = action_to_agents.get(a, [])
        if not capable_agents:
            unassigned.append(a_raw)
            continue
        expanded = [a_raw] + ACTION_SYNONYMS.get(a, [])
        for ag in capable_agents:
            agent_to_actions[ag].extend(expanded)

    def uniq(seq: List[str]) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for x in seq:
            k = _norm_action(x)
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    ran = ", ".join(uniq(agent_to_actions.get("RAN", []))) or "none"
    edge = ", ".join(uniq(agent_to_actions.get("Edge", []))) or "none"
    core = ", ".join(uniq(agent_to_actions.get("Core", []))) or "none"

    ref = (
        f"{label_u} response: "
        f"RAN actions: {ran}. "
        f"Edge actions: {edge}. "
        f"Core actions: {core}."
    )
    if unassigned:
        ref += f" Unassigned actions: {', '.join(unassigned)}."
    return ref


def extract_overall_reasoning(action_plan: Optional[Dict[str, Any]]) -> str:
    if not action_plan or not isinstance(action_plan, dict):
        return ""
    s = (action_plan.get("overall_reasoning") or action_plan.get("reasoning") or "").strip()
    return s


def load_rerank_comparison_files(
    directory: Path,
    *,
    glob_pattern: str = "rerank_comparison_sample_*.json",
) -> List[Dict[str, Any]]:
    directory = Path(directory)
    files = sorted(directory.glob(glob_pattern))
    out: List[Dict[str, Any]] = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("report_type") != "rerank_ablation_comparison":
            continue
        data["file_path"] = str(file_path)
        data["file_name"] = file_path.name
        out.append(data)
    return out


def compute_lexical_similarity_scores(
    reference: str,
    candidate: str,
    sbert_model: Any,
) -> Dict[str, float]:
    from bert_score import score as bert_score_fn
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer
    from sklearn.metrics.pairwise import cosine_similarity

    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    smoothing = SmoothingFunction().method1
    bleu = float(
        sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    )

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    rouge1 = float(scorer.score(reference, candidate)["rouge1"].fmeasure)

    ref_embedding = sbert_model.encode([reference])
    cand_embedding = sbert_model.encode([candidate])
    sbert_cosine = float(cosine_similarity(ref_embedding, cand_embedding)[0][0])

    p, r, f1 = bert_score_fn([candidate], [reference], lang="en", verbose=False)
    return {
        "bleu": bleu,
        "rouge1": rouge1,
        "sbert_cosine": sbert_cosine,
        "bertscore_precision": float(p.mean().item()),
        "bertscore_recall": float(r.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
    }


def score_rerank_directory(
    input_dir: Path,
    sbert_model: Any,
    attack_options_kb: Dict[str, Any],
    agents_kb: Dict[str, Any],
    *,
    glob_pattern: str = "rerank_comparison_sample_*.json",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rep in load_rerank_comparison_files(input_dir, glob_pattern=glob_pattern):
        file_name = rep["file_name"]
        sample_id = rep.get("sample_id", "unknown")
        pred = rep.get("prediction") or {}
        label = str(pred.get("predicted_label") or "OTHERS")
        reference_text = build_reference_text_for_label(
            label, attack_options_kb, agents_kb
        )
        modes = rep.get("modes") or {}
        if not isinstance(modes, dict):
            continue
        for mode_key, block in modes.items():
            if not isinstance(block, dict):
                continue
            plan = block.get("llm_action_plan")
            text = extract_overall_reasoning(plan if isinstance(plan, dict) else None)
            if not text:
                continue
            scores = compute_lexical_similarity_scores(
                reference_text, text, sbert_model
            )
            rows.append(
                {
                    "file_name": file_name,
                    "sample_id": sample_id,
                    "predicted_label": label,
                    "rerank_mode": str(mode_key),
                    "rerank_label": block.get("label") or mode_key,
                    "reference_text": reference_text,
                    "text": text,
                    **scores,
                }
            )
    return rows
