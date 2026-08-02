"""Cybersecurity BERT encoders for retrieval grading and three-layer evaluation metrics."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

from utils.compute_device import (
    hf_encoder_batch_size,
    preferred_torch_device_string,
    report_compute_device_status,
    sentence_transformer_device,
)

# Short keys used in metric column suffixes (e.g. recall_at_10_cysecbert).
CYBER_ENCODER_KEYS: Tuple[str, ...] = ("cysecbert", "ctibert", "securebert")

ENCODER_HF_IDS: Dict[str, str] = {
    "minilm": "all-MiniLM-L6-v2",
    "cysecbert": "markusbayer/CySecBERT",
    "ctibert": "ibm-research/CTI-BERT",
    "securebert": "cisco-ai/SecureBERT2.0-biencoder",
}

# Legacy alias: graded relevance with MiniLM matches Score.ipynb baseline.
GRADING_ENCODER_KEYS: Tuple[str, ...] = ("minilm",) + CYBER_ENCODER_KEYS

_MAX_LENGTH_BERT = 512
_rouge_scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

_st_models: Dict[str, Any] = {}
_hf_models: Dict[str, Any] = {}
_hf_tokenizers: Dict[str, Any] = {}
_device_logged = False


def _encoder_device() -> str:
    override = os.environ.get("CYBER_ENCODER_DEVICE", "").strip()
    if override:
        return override
    return preferred_torch_device_string()


def _log_encoder_device_once(key: str, device: str) -> None:
    global _device_logged
    if _device_logged:
        return
    print(f"[cyber_encoders] {key} on device={device}")
    print(report_compute_device_status())
    _device_logged = True


def rouge1_f(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(_rouge_scorer.score(a, b)["rouge1"].fmeasure)


def _get_sentence_transformer(key: str) -> Any:
    if key not in _st_models:
        from sentence_transformers import SentenceTransformer

        model_id = ENCODER_HF_IDS[key]
        device = sentence_transformer_device()
        _st_models[key] = SentenceTransformer(
            model_id,
            device=device,
        )
        _log_encoder_device_once(key, device)
    return _st_models[key]


def _get_hf_bert(key: str) -> Tuple[Any, Any]:
    if key not in _hf_models:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_id = ENCODER_HF_IDS[key]
        tok = AutoTokenizer.from_pretrained(model_id)
        device = _encoder_device()
        model = AutoModel.from_pretrained(model_id)
        model.to(device)
        model.eval()
        _hf_tokenizers[key] = tok
        _hf_models[key] = model
        _log_encoder_device_once(key, device)
    return _hf_tokenizers[key], _hf_models[key]


def _mean_pool(last_hidden: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(key: str, texts: Sequence[str]) -> np.ndarray:
    """Return (n, dim) float32 embeddings for one encoder key."""
    cleaned = [str(t or "") for t in texts]
    if not cleaned:
        return np.zeros((0, 0), dtype=np.float32)

    if key in ("minilm", "securebert"):
        model = _get_sentence_transformer(key)
        embs = model.encode(
            cleaned,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=sentence_transformer_device(),
        )
        return np.asarray(embs, dtype=np.float32)

    if key in ("cysecbert", "ctibert"):
        import torch

        tok, model = _get_hf_bert(key)
        device = _encoder_device()
        all_vecs: List[np.ndarray] = []
        batch_size = hf_encoder_batch_size()
        with torch.no_grad():
            for i in range(0, len(cleaned), batch_size):
                batch = cleaned[i : i + batch_size]
                enc = tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=_MAX_LENGTH_BERT,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
                all_vecs.append(pooled.cpu().numpy())
        return np.vstack(all_vecs).astype(np.float32)

    raise KeyError(f"Unknown encoder key: {key!r}")


def pairwise_cosine(a: str, b: str, encoder_key: str) -> float:
    if not (a or "").strip() or not (b or "").strip():
        return 0.0
    embs = encode_texts(encoder_key, [a, b])
    if embs.shape[0] < 2:
        return 0.0
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])


def graded_relevance(
    reference: str,
    chunk_text: str,
    encoder_key: str = "minilm",
) -> float:
    """0.5 * ROUGE-1 + 0.5 * encoder cosine (Score.ipynb style)."""
    if not (chunk_text or "").strip():
        return 0.0
    rouge = rouge1_f(reference, chunk_text)
    sem = pairwise_cosine(reference, chunk_text, encoder_key)
    return float(max(0.0, 0.5 * rouge + 0.5 * sem))


def graded_relevance_multi(reference: str, chunk_text: str) -> Dict[str, float]:
    return {k: graded_relevance(reference, chunk_text, k) for k in GRADING_ENCODER_KEYS}


def metric_suffix(encoder_key: str) -> str:
    """Column suffix: minilm -> sbert-style legacy uses empty for recall; cyber uses key."""
    if encoder_key == "minilm":
        return ""
    return f"_{encoder_key}"


def scoring_metric_column_names() -> List[str]:
    """All optional metric columns written by Score.ipynb / save_scores_by_class."""
    base = [
        "recall_at_10",
        "ndcg_at_10",
        "context_sbert_cosine",
        "context_rouge1",
        "context_crag_proxy",
        "utilization_sbert_cosine",
        "utilization_rouge1",
        "utilization_crag_proxy",
        "reasoning_sbert_cosine",
        "reasoning_rouge1",
        "reasoning_crag_proxy",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]
    for enc in CYBER_ENCODER_KEYS:
        base.extend(
            [
                f"recall_at_10_{enc}",
                f"ndcg_at_10_{enc}",
                f"context_{enc}_cosine",
                f"context_{enc}_crag_proxy",
                f"utilization_{enc}_cosine",
                f"utilization_{enc}_crag_proxy",
                f"reasoning_{enc}_cosine",
                f"reasoning_{enc}_crag_proxy",
            ]
        )
    return base


def normalize_rerank_mode(rerank_mode: str) -> Tuple[str, bool]:
    """Return (base_mode, uses_securebert_index)."""
    rm = (rerank_mode or "crossencoder").strip().lower()
    if rm.endswith("_securebert"):
        return rm[: -len("_securebert")], True
    return rm, False


def grades_for_texts(
    reference: str,
    ranked_texts: List[str],
    pool_texts: List[str],
    encoder_key: str,
    *,
    show_progress: bool = False,
) -> Tuple[List[float], List[float]]:
    """Ranked and pool relevance grades for one encoder."""
    ranked_iter: Iterable[str] = ranked_texts
    if show_progress and len(ranked_texts) > 5:
        from utils.notebook_progress import progress

        ranked_iter = progress(
            ranked_texts,
            desc=f"ranked:{encoder_key}",
            leave=False,
            unit="chunk",
        )
    ranked_grades = [
        graded_relevance(reference, t, encoder_key) for t in ranked_iter
    ]
    pool_grades = list(ranked_grades)
    tail = pool_texts[len(ranked_grades) :]
    tail_iter: Iterable[str] = tail
    if show_progress and len(tail) > 5:
        from utils.notebook_progress import progress

        tail_iter = progress(
            tail,
            desc=f"pool:{encoder_key}",
            leave=False,
            unit="chunk",
        )
    for t in tail_iter:
        pool_grades.append(graded_relevance(reference, t, encoder_key))
    return ranked_grades, pool_grades
