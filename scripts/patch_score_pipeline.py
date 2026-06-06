#!/usr/bin/env python3
"""Align Score.ipynb with 5-class pipeline, NIDS devices, and output/plans/."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Score.ipynb"

_scoring_spec = importlib.util.spec_from_file_location(
    "scoring_cell_cyber", ROOT / "scripts" / "scoring_cell_cyber.py"
)
_scoring_mod = importlib.util.module_from_spec(_scoring_spec)
assert _scoring_spec.loader is not None
_scoring_spec.loader.exec_module(_scoring_mod)
SCORING_CELL = _scoring_mod.SCORING_CELL

IMPORTS_CELL = '''# Import required libraries
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import nltk

from utils.action_config import get_agent_order, load_agentic_features
from utils.label_config import TARGET_CLASSES
from utils.rerank_plan_scoring import (
    build_reference_text_for_label as _build_reference_for_label,
    extract_overall_reasoning as _extract_guidance_text,
    load_scoring_kb,
)

_attack_options_kb, agents_kb = load_scoring_kb()
_agentic_data = load_agentic_features()
_agent_order = get_agent_order(_agentic_data)


def build_reference_text_for_label(label: str) -> str:
    """Description-based reference (FlowProbe / Firewall / CoreRouter)."""
    return _build_reference_for_label(
        label.strip().upper(),
        _attack_options_kb,
        agents_kb,
        _agent_order,
    )


def extract_overall_reasoning(action_plan: Dict[str, Any]) -> str:
    """Scoring text from agent_guidance (legacy overall_reasoning fallback)."""
    return _extract_guidance_text(action_plan)


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from utils.compute_device import (
    bert_score_device,
    report_compute_device_status,
    sentence_transformer_device,
)

print("Compute devices:")
print(report_compute_device_status())
print("Loading SBERT model...")
sbert_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=sentence_transformer_device(),
)
print(f"SBERT loaded on device={sentence_transformer_device()}")
print("Libraries imported successfully!")
print(f"Target attack classes: {list(TARGET_CLASSES)}")
'''

CONFIG_CELL = '''# Configuration
#
# Per-sample references come from agentic_features.json (NIDS device descriptions).
# Plan reports are read from output/plans/<CLASS>/ (see load cell below).
from utils.pipeline_paths import PLANS_ROOT, SCORES_ROOT, ensure_pipeline_dirs

ensure_pipeline_dirs()
RAG_DOCS_DIR = PLANS_ROOT

print(f"Plans directory: {RAG_DOCS_DIR} (recursive under output/plans/<CLASS>/)")

print("\\nPer-label reference texts (first 200 chars):")
for _lbl in TARGET_CLASSES:
    _ref = build_reference_text_for_label(_lbl)
    _preview = _ref[:200] + ("..." if len(_ref) > 200 else "")
    print(f"  [{_lbl}] {_preview}")
'''

LOAD_FN_OLD_MARKER = "def load_rerank_comparison_files(directory: Path) -> List[Dict[str, Any]]:"
LOAD_FN_NEW = '''def load_rerank_comparison_files(directory: Path) -> List[Dict[str, Any]]:
    """Load latest rerank ablation JSON per sample_id from output/plans/."""
    from utils.rerank_plan_scoring import load_rerank_comparison_files as _load_plans

    return _load_plans(
        directory,
        recursive=True,
        dedupe_latest=True,
        require_report_type=True,
        verbose=True,
    )'''

LOAD_CELL = '''# Load rerank comparison reports (newest per sample_id under output/plans/)
comparison_data = load_rerank_comparison_files(RAG_DOCS_DIR)

if not comparison_data:
    print("No rerank comparison files found under", RAG_DOCS_DIR.resolve())
else:
    print(f"\\nTotal reports for scoring: {len(comparison_data)}")
    _by_lbl: dict[str, int] = {}
    for _d in comparison_data:
        _l = str((_d.get("prediction") or {}).get("predicted_label", "UNKNOWN")).upper()
        _by_lbl[_l] = _by_lbl.get(_l, 0) + 1
    print("By class:", ", ".join(f"{k}={v}" for k, v in sorted(_by_lbl.items())))
'''


def _replace_load_function(src: str) -> str:
    start = src.find("def load_rerank_comparison_files")
    if start < 0:
        return src
    end = src.find("\n\ndef calculate_bleu_score", start)
    if end < 0:
        end = src.find("\ndef calculate_bleu_score", start)
    if end < 0:
        return src
    return src[:start] + LOAD_FN_NEW + src[end:]


def clear_outputs(nb: dict) -> None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    changed = 0

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        orig = src

        if cell["cell_type"] == "code" and src.startswith("# Import required libraries"):
            cell["source"] = [line + "\n" for line in IMPORTS_CELL.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            changed += 1
            continue

        if cell["cell_type"] == "code" and "# Configuration" in src and "RAG_DOCS_DIR = PLANS_ROOT" in src:
            cell["source"] = [line + "\n" for line in CONFIG_CELL.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            changed += 1
            continue

        if cell["cell_type"] == "code" and LOAD_FN_OLD_MARKER in src:
            src = _replace_load_function(src)
            # Remove duplicate extract_overall_reasoning if we import in cell 2
            old_extract = '''def extract_overall_reasoning(action_plan: Dict[str, Any]) -> str:
    """Extract scoring text from agent_guidance (legacy overall_reasoning fallback)."""
    from utils.plan_prompts import agent_guidance_to_scoring_text

    if not isinstance(action_plan, dict):
        return ""
    text = agent_guidance_to_scoring_text(action_plan)
    if text:
        return text
    return (action_plan.get("overall_reasoning") or action_plan.get("reasoning") or "").strip()


'''
            if old_extract in src:
                src = src.replace(old_extract, "")
            changed += 1

        if cell["cell_type"] == "code" and src.strip().startswith(
            "# Score rerank modes with retrieval"
        ):
            cell["source"] = [line + "\n" for line in SCORING_CELL.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            changed += 1
            continue

        if cell["cell_type"] == "code" and src.strip().startswith("# Load all rerank comparison"):
            cell["source"] = [line + "\n" for line in LOAD_CELL.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            changed += 1
            continue

        if "preferred_order = [lbl for lbl in attack_options_kb.keys()" in src:
            src = src.replace(
                "preferred_order = [lbl for lbl in attack_options_kb.keys() if lbl in observed_labels]",
                "preferred_order = [lbl for lbl in TARGET_CLASSES if lbl in observed_labels]",
            )
            changed += 1

        if 'f"  Overall Reasoning: {reasoning_text' in src:
            src = src.replace("Overall Reasoning", "Agent guidance")
            changed += 1

        if '"top-10 context and overall_reasoning.' in src:
            src = src.replace(
                '"top-10 context and overall_reasoning.',
                '"top-10 context and agent_guidance (overall_reasoning column).',
            )
            src = src.replace(
                '"reasoning": "SBERT cosine + ROUGE-1 + CRAG-style proxy between overall_reasoning and reference text.",',
                '"reasoning": "SBERT cosine + ROUGE-1 + CRAG-style proxy between agent_guidance and reference text.",',
            )
            src = src.replace(
                '"bertscore": "BERTScore-F1 computed between ground-truth reference statement and overall_reasoning.",',
                '"bertscore": "BERTScore-F1 between description-based reference and agent_guidance.",',
            )
            changed += 1

        if cell["cell_type"] == "markdown":
            if "This notebook scores" in src and "five training classes" not in src:
                src = src.replace(
                    "This notebook scores the `rerank_comparison_sample_*.json` reports produced by `Plan.ipynb`.",
                    "This notebook scores the `rerank_comparison_sample_*.json` reports produced by `Plan.ipynb` under **`output/plans/<CLASS>/`** (five training classes: BENIGN, DDOS, DOS, PORTSCAN, FTPPATATOR; typically 50 samples per class). It keeps the **newest** file per `sample_id` when Plan was re-run.",
                )
            if "nine headline metrics" in src and "per-class" not in src.lower():
                src = src.replace(
                    "the nine headline metrics",
                    "the headline metrics",
                )
            if "Util_*` compare retrieved top-10 context against `overall_reasoning`" in src:
                src = src.replace(
                    "`overall_reasoning`",
                    "`agent_guidance` (stored as `overall_reasoning` in rows)",
                )
            if "reflected in the LLM's `overall_reasoning`" in src:
                src = src.replace(
                    "the LLM's `overall_reasoning`",
                    "the LLM's `agent_guidance`",
                )
            if "## Extract Overall Reasoning" in src:
                src = src.replace(
                    "## Extract Overall Reasoning and Calculate Scores",
                    "## Extract agent guidance and calculate scores",
                )

        if src != orig:
            cell["source"] = [line + "\n" for line in src.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"

    clear_outputs(nb)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched Score.ipynb ({changed} cell touch(es)); cleared stale outputs.")


if __name__ == "__main__":
    main()
