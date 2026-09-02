"""Score action-plan text (``agent_guidance`` paragraphs) vs description-based reference.

Aligns with ``Score.ipynb``: reference from ``agentic_features.json`` device roles;
LLM text from ``utils.plan_prompts.agent_guidance_to_scoring_text``.
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
    """Load attack entries and agent capabilities for reference building."""
    from utils.action_config import load_agentic_features, load_attack_options

    root = repo_root or Path(__file__).resolve().parents[1]
    attack_opts = load_attack_options(root / "RAG_docs" / "attack_options.json")
    agentic = load_agentic_features(root / "RAG_docs" / "agentic_features.json")
    return attack_opts.get("attacks") or {}, agentic.get("agents") or {}


def build_reference_text_for_label(
    label: str,
    attack_options_kb: Dict[str, Any],
    agents_kb: Dict[str, Any],
    agent_order: Optional[List[str]] = None,
) -> str:
    """Build per-device reference from agent descriptions (attack_options ignored)."""
    from utils.plan_prompts import build_reference_text_for_label as _build_ref

    return _build_ref(
        label,
        agents_kb,
        agent_order=agent_order,
        attack_options_kb=attack_options_kb,
    )


def extract_overall_reasoning(action_plan: Optional[Dict[str, Any]]) -> str:
    """Scoring text: concatenated agent_guidance paragraphs (legacy overall_reasoning fallback)."""
    from utils.plan_prompts import agent_guidance_to_scoring_text

    return agent_guidance_to_scoring_text(action_plan)


def load_rerank_comparison_files(
    directory: Path,
    *,
    glob_pattern: str = "rerank_comparison_sample_*.json",
    recursive: bool = True,
    dedupe_latest: bool = True,
    require_report_type: bool = True,
    target_classes: Optional[Tuple[str, ...]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Load Plan rerank reports from ``output/plans/`` (optionally per-class subdirs).

    When ``dedupe_latest`` is True, keeps the newest file per ``sample_id`` so
    re-runs of Plan.ipynb do not double-count rows.
    """
    import re

    from utils.label_config import TARGET_CLASSES

    directory = Path(directory)
    allowed = tuple(target_classes or TARGET_CLASSES)
    pattern = glob_pattern.lstrip("/")

    if recursive:
        all_paths = list(directory.rglob(pattern))
    else:
        all_paths = list(directory.glob(pattern))

    if dedupe_latest:
        best: Dict[int, Tuple[float, Path]] = {}
        sid_pat = re.compile(r"rerank_comparison_sample_(\d+)_")
        for fp in all_paths:
            m = sid_pat.search(fp.name)
            if not m:
                continue
            sid = int(m.group(1))
            mt = fp.stat().st_mtime
            prev = best.get(sid)
            if prev is None or mt > prev[0]:
                best[sid] = (mt, fp)
        file_paths = [p for _, p in sorted(best.values(), key=lambda x: x[0])]
    else:
        file_paths = sorted(all_paths)

    out: List[Dict[str, Any]] = []
    by_class: Dict[str, int] = defaultdict(int)
    skipped_type = 0
    skipped_label = 0

    from utils.notebook_progress import progress

    for fp in progress(file_paths, desc="Load rerank reports", unit="file"):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        if require_report_type and data.get("report_type") != "rerank_ablation_comparison":
            skipped_type += 1
            continue
        lbl = str((data.get("prediction") or {}).get("predicted_label", "")).strip().upper()
        if lbl not in allowed:
            skipped_label += 1
            continue
        data["file_path"] = str(fp)
        data["file_name"] = fp.name
        data["_source_file"] = fp.name
        data["_plans_class_dir"] = fp.parent.name
        out.append(data)
        by_class[lbl] += 1

    out.sort(
        key=lambda d: (
            str((d.get("prediction") or {}).get("predicted_label", "")),
            int(d.get("sample_id") if d.get("sample_id") is not None else -1),
        )
    )

    if verbose:
        print(f"Found {len(all_paths)} rerank comparison file(s) on disk under {directory}")
        if dedupe_latest:
            print(f"  -> {len(file_paths)} after keeping newest per sample_id")
        print(f"  -> {len(out)} report(s) in TARGET_CLASSES {list(allowed)}")
        if skipped_type:
            print(f"  (skipped {skipped_type} non-ablation JSON)")
        if skipped_label:
            print(f"  (skipped {skipped_label} outside TARGET_CLASSES)")
        for cls in allowed:
            if by_class.get(cls):
                print(f"     [{cls}] {by_class[cls]} sample(s)")
    return out
