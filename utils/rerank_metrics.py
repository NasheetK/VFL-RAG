"""Metrics for rerank ablation JSON reports: query–chunk embedding similarity and LLM output comparison.

Uses the same SentenceTransformer-based embedder as ``utils.rag_utils.load_vector_store`` (manifest model).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.embeddings import Embeddings


def _cosine_vec(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    da = sum(x * x for x in a)
    db = sum(x * x for x in b)
    if da <= 0.0 or db <= 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return float(dot) / float(math.sqrt(da) * math.sqrt(db))


def mean_topk_cosine_query_chunks(
    embeddings: Embeddings,
    query: str,
    chunk_texts: List[str],
    *,
    top_k: Optional[int] = None,
) -> float:
    """
    Mean cosine similarity between the query embedding and each of the first ``top_k`` chunk texts.

    Chunk strings should match what you want to score (e.g. ``title + "\\n\\n" + text``).
    """
    q = (query or "").strip()
    texts = [str(t or "") for t in chunk_texts]
    if top_k is not None:
        texts = texts[: int(top_k)]
    texts = [t for t in texts if t.strip()]
    if not q or not texts:
        return 0.0
    qv = list(embeddings.embed_query(q))
    mat = embeddings.embed_documents(texts)
    sims = [_cosine_vec(qv, list(row)) for row in mat]
    return float(sum(sims) / len(sims)) if sims else 0.0


def anchor_query_from_report(report: Dict[str, Any]) -> str:
    """MMR/rerank anchor query; falls back to legacy ``rag_query`` (including MULTI_QUERY bundles)."""
    m = str(report.get("mmr_rerank_anchor_query") or "").strip()
    if m:
        return m
    rq = str(report.get("rag_query") or "")
    if rq.startswith("MULTI_QUERY"):
        m2 = re.search(r"\[1\]\s*(.+?)(?:\n\n---|\Z)", rq, re.DOTALL)
        if m2:
            return m2.group(1).strip()
    return rq.strip()


def top_result_chunk_texts(
    mode_block: Dict[str, Any],
    *,
    top_k: int = 10,
) -> List[str]:
    tops = (mode_block.get("rag_info") or {}).get("top_results") or []
    out: List[str] = []
    for t in tops[: int(top_k)]:
        title = str((t or {}).get("title") or "")
        body = str((t or {}).get("text") or "")
        out.append(f"{title}\n\n{body}".strip())
    return out


def cyber_encoder_metrics_for_report(
    report: Dict[str, Any],
    *,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Mean query–chunk cosine per rerank mode for CySecBERT, CTI-BERT, SecureBERT."""
    from utils.cyber_encoders import CYBER_ENCODER_KEYS, encode_texts

    q = anchor_query_from_report(report)
    modes = report.get("modes") or {}
    per_mode: Dict[str, Any] = {}
    for mode_key, block in modes.items():
        if not isinstance(block, dict):
            continue
        texts = top_result_chunk_texts(block, top_k=top_k)
        texts = [t for t in texts if (t or "").strip()]
        if not q or not texts:
            per_mode[str(mode_key)] = {
                enc: 0.0 for enc in CYBER_ENCODER_KEYS
            }
            continue
        q_embs: Dict[str, Any] = {}
        mode_scores: Dict[str, float] = {}
        for enc in CYBER_ENCODER_KEYS:
            qv = encode_texts(enc, [q])[0]
            q_embs[enc] = qv
            mat = encode_texts(enc, texts)
            sims = [
                _cosine_vec(list(qv), list(row)) for row in mat
            ]
            mode_scores[enc] = float(sum(sims) / len(sims)) if sims else 0.0
        per_mode[str(mode_key)] = mode_scores
    return {
        "anchor_query_preview": q[:240] + ("..." if len(q) > 240 else ""),
        "top_k": int(top_k),
        "per_mode": per_mode,
    }


def embedding_metrics_for_report(
    report: Dict[str, Any],
    embeddings: Embeddings,
    *,
    top_k: int = 10,
    include_cyber_encoders: bool = True,
) -> Dict[str, Any]:
    q = anchor_query_from_report(report)
    modes = report.get("modes") or {}
    per_mode: Dict[str, Any] = {}
    for mode_key, block in modes.items():
        if not isinstance(block, dict):
            continue
        texts = top_result_chunk_texts(block, top_k=top_k)
        entry: Dict[str, Any] = {
            "mean_top_k_cosine_query_chunk": mean_topk_cosine_query_chunks(
                embeddings, q, texts, top_k=len(texts)
            ),
            "n_chunks_scored": len(texts),
        }
        per_mode[str(mode_key)] = entry
    out: Dict[str, Any] = {
        "anchor_query_preview": q[:240] + ("..." if len(q) > 240 else ""),
        "top_k": int(top_k),
        "per_mode": per_mode,
    }
    if include_cyber_encoders:
        out["cyber_encoders"] = cyber_encoder_metrics_for_report(report, top_k=top_k)
    return out


def _plan_signature(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    from utils.plan_prompts import guidance_signature

    return guidance_signature(plan)


def _jaccard_tuple(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    u = sa | sb
    if not u:
        return 1.0
    return len(sa & sb) / len(u)


def downstream_comparison_for_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pairwise comparison of ``llm_action_plan`` fields across rerank modes in one report.

    Useful for fixed-sample ablations: see where per-device guidance text diverges across modes.
    """
    modes = report.get("modes") or {}
    keys = [str(k) for k in modes.keys()]
    sigs: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        block = modes.get(k) or {}
        sigs[k] = _plan_signature(block.get("llm_action_plan"))

    pairs: List[Dict[str, Any]] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            sa, sb = sigs[a], sigs[b]
            if sa.get("missing") or sb.get("missing"):
                pairs.append(
                    {
                        "mode_a": a,
                        "mode_b": b,
                        "both_present": not (sa.get("missing") or sb.get("missing")),
                        "note": "missing llm_action_plan in one or both modes",
                    }
                )
                continue
            wa = set(sa.get("word_set") or ())
            wb = set(sb.get("word_set") or ())
            pairs.append(
                {
                    "mode_a": a,
                    "mode_b": b,
                    "guidance_word_jaccard": _jaccard_tuple(
                        tuple(sorted(wa)),
                        tuple(sorted(wb)),
                    ),
                    "char_len_a": sa.get("char_len"),
                    "char_len_b": sb.get("char_len"),
                }
            )

    return {
        "sample_id": report.get("sample_id"),
        "mode_keys": keys,
        "pairwise": pairs,
    }


def load_rerank_comparison(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data
