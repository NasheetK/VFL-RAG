"""RAG Recall@k / nDCG@k from Plan rerank comparison reports (aligned with Score.ipynb)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.cyber_encoders import (
    GRADING_ENCODER_KEYS,
    grades_for_texts,
    graded_relevance as cyber_graded_relevance,
    metric_suffix,
)
from utils.plan_prompts import build_reference_text_for_label

TOP_K = 10
RELEVANCE_FLOOR = 0.15

MODE_ORDER = (
    "none",
    "bm25",
    "doct5query",
    "crossencoder",
    "bm25_crossencoder",
    "colbert",
    "none_securebert",
    "crossencoder_securebert",
)
MODE_PREFIX = {m: m for m in MODE_ORDER}


def chunk_string(result: Dict[str, Any]) -> str:
    title = str(result.get("title", "") or "")
    text = str(result.get("text", "") or "")
    return f"{title}\n\n{text}".strip()


def graded_relevance(reference: str, chunk_text: str) -> float:
    """MiniLM + ROUGE graded relevance (Score.ipynb baseline)."""
    return cyber_graded_relevance(reference, chunk_text, "minilm")


def recall_at_k(
    ranked_grades: List[float],
    pool_grades: List[float],
    k: int = TOP_K,
    relevance_threshold: float = RELEVANCE_FLOOR,
) -> float:
    """Recall@k: denominator = relevant chunks in candidate_pool (Score.ipynb)."""
    if not pool_grades:
        return 0.0
    total_relevant = sum(1 for g in pool_grades if g >= relevance_threshold)
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(
        1 for g in ranked_grades[:k] if g >= relevance_threshold
    )
    return float(min(1.0, retrieved_relevant / total_relevant))


def ndcg_at_k(ranked_grades: List[float], k: int = TOP_K) -> float:
    if not ranked_grades:
        return 0.0
    k = min(k, len(ranked_grades))
    dcg = sum(
        (2.0**rel - 1.0) / np.log2(i + 1.0)
        for i, rel in enumerate(ranked_grades[:k], start=1)
    )
    ideal = sorted(ranked_grades, reverse=True)[:k]
    idcg = sum(
        (2.0**rel - 1.0) / np.log2(i + 1.0)
        for i, rel in enumerate(ideal, start=1)
    )
    return float(dcg / idcg) if idcg > 0 else 0.0


def _recall_ndcg_columns(
    reference: str,
    ranked_texts: List[str],
    pool_texts: List[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for enc in GRADING_ENCODER_KEYS:
        ranked_g, pool_g = grades_for_texts(reference, ranked_texts, pool_texts, enc)
        suf = metric_suffix(enc)
        out[f"recall_at_10{suf}"] = round(
            recall_at_k(ranked_g, pool_g, k=TOP_K), 4
        )
        out[f"ndcg_at_10{suf}"] = round(ndcg_at_k(ranked_g, k=TOP_K), 4)
    return out


def _pick_latest_per_sample_id(paths: List[Path]) -> Dict[int, Path]:
    best: Dict[int, Tuple[float, Path]] = {}
    pat = re.compile(r"rerank_comparison_sample_(\d+)_")
    for p in paths:
        m = pat.search(p.name)
        if not m:
            continue
        sid = int(m.group(1))
        mt = p.stat().st_mtime
        prev = best.get(sid)
        if prev is None or mt > prev[0]:
            best[sid] = (mt, p)
    return {k: v[1] for k, v in best.items()}


def load_rerank_reports(
    plans_root: Path,
    *,
    sample_ids: Optional[List[int]] = None,
) -> List[Tuple[int, Path]]:
    plans_root = Path(plans_root)
    paths = list(plans_root.rglob("rerank_comparison_sample_*.json"))
    by_sid = _pick_latest_per_sample_id(paths)
    ids = list(sample_ids) if sample_ids is not None else sorted(by_sid.keys())
    if not ids:
        raise FileNotFoundError(
            f"No rerank_comparison_sample_*.json under {plans_root} "
            "(searched recursively under output/plans/<CLASS>/)."
        )
    out: List[Tuple[int, Path]] = []
    for sid in ids:
        p = by_sid.get(sid)
        if p is None:
            raise FileNotFoundError(
                f"No rerank_comparison_sample_{sid}_*.json under {plans_root}."
            )
        out.append((sid, p))
    return sorted(out, key=lambda x: x[0])


def _agents_kb(agentic_features_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(agentic_features_data, dict):
        return {}
    agents = agentic_features_data.get("agents")
    return agents if isinstance(agents, dict) else agentic_features_data


from utils.notebook_progress import progress


def build_rag_scores_dataframe(
    report_paths: List[Tuple[int, Path]],
    agentic_features_data: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    agents_kb = _agents_kb(agentic_features_data)
    rows: List[Dict[str, Any]] = []

    for sid, path in progress(
        report_paths,
        desc="RAG Recall/nDCG",
        unit="report",
    ):
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)
        label = str((rep.get("prediction") or {}).get("predicted_label") or "UNKNOWN")
        ulabel = label.strip().upper()
        reference_text = build_reference_text_for_label(ulabel, agents_kb)

        modes_data = rep.get("modes") or {}
        row: Dict[str, Any] = {
            "attack_group": f"sample_{sid}_{ulabel}",
            "sample_id": sid,
            "predicted_label": ulabel,
        }
        for mk in MODE_ORDER:
            prefix = MODE_PREFIX[mk]
            mode_block = modes_data.get(mk)
            if not isinstance(mode_block, dict):
                for enc in GRADING_ENCODER_KEYS:
                    suf = metric_suffix(enc)
                    row[f"{prefix}_Recall@10{suf}"] = float("nan")
                    row[f"{prefix}_nDCG@10{suf}"] = float("nan")
                continue
            rag_info = mode_block.get("rag_info") or {}
            top = rag_info.get("top_results") or []
            pool = rag_info.get("candidate_pool") or top
            ranked_texts = [chunk_string(d) for d in top]
            pool_texts = [chunk_string(d) for d in pool]

            metrics = _recall_ndcg_columns(reference_text, ranked_texts, pool_texts)
            for enc in GRADING_ENCODER_KEYS:
                suf = metric_suffix(enc)
                row[f"{prefix}_Recall@10{suf}"] = metrics.get(
                    f"recall_at_10{suf}", float("nan")
                )
                row[f"{prefix}_nDCG@10{suf}"] = metrics.get(
                    f"ndcg_at_10{suf}", float("nan")
                )
        rows.append(row)

    col_order = ["attack_group", "sample_id", "predicted_label"]
    for mk in MODE_ORDER:
        p = MODE_PREFIX[mk]
        for enc in GRADING_ENCODER_KEYS:
            suf = metric_suffix(enc)
            col_order.extend([f"{p}_Recall@10{suf}", f"{p}_nDCG@10{suf}"])
    df = pd.DataFrame(rows)
    return df[[c for c in col_order if c in df.columns]]
