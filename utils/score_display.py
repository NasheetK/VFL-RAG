"""Score.ipynb display columns: MiniLM headline + cyber BERT encoder metrics."""

from __future__ import annotations

from typing import Dict, List

from utils.cyber_encoders import CYBER_ENCODER_KEYS

ENCODER_SHORT: Dict[str, str] = {
    "cysecbert": "CySec",
    "ctibert": "CTI",
    "securebert": "SecBERT",
}

HEADLINE_METRIC_COLS: List[str] = [
    "recall_at_10",
    "ndcg_at_10",
    "context_sbert_cosine",
    "context_crag_proxy",
    "utilization_sbert_cosine",
    "utilization_crag_proxy",
    "reasoning_sbert_cosine",
    "reasoning_crag_proxy",
    "bertscore_f1",
]

RERANK_METHOD_ORDER: List[str] = [
    "none",
    "bm25",
    "doct5query",
    "crossencoder",
    "bm25_crossencoder",
    "colbert",
    "none_securebert",
    "crossencoder_securebert",
]

METHOD_DISPLAY: Dict[str, str] = {
    "none": "Dense MMR",
    "bm25": "BM25",
    "doct5query": "docT5",
    "crossencoder": "Dense -> CE",
    "bm25_crossencoder": "BM25 -> CE",
    "colbert": "Dense -> ColBERT",
    "none_securebert": "SecureBERT MMR",
    "crossencoder_securebert": "SecureBERT -> CE",
}


def cyber_metric_columns() -> List[str]:
    cols: List[str] = []
    for enc in CYBER_ENCODER_KEYS:
        cols.extend(
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
    return cols


def all_metric_columns() -> List[str]:
    return HEADLINE_METRIC_COLS + cyber_metric_columns()


# All possible metric column names (before filtering to a DataFrame).
ALL_METRIC_COLS: List[str] = all_metric_columns()


def metric_display_rename() -> Dict[str, str]:
    rename: Dict[str, str] = {
        "recall_at_10": "recall@10",
        "ndcg_at_10": "ndcg@10",
        "context_sbert_cosine": "RI_SBert",
        "context_crag_proxy": "CRAG",
        "utilization_sbert_cosine": "Util_SBert",
        "utilization_crag_proxy": "Util_CRAG",
        "reasoning_sbert_cosine": "LLM_SBert",
        "reasoning_crag_proxy": "LLM_CRAG",
        "bertscore_f1": "BertscoreF1",
    }
    for enc, short in ENCODER_SHORT.items():
        rename[f"recall_at_10_{enc}"] = f"recall@10_{short}"
        rename[f"ndcg_at_10_{enc}"] = f"nDCG@10_{short}"
        rename[f"context_{enc}_cosine"] = f"RI_{short}"
        rename[f"context_{enc}_crag_proxy"] = f"CRAG_{short}"
        rename[f"utilization_{enc}_cosine"] = f"Util_{short}"
        rename[f"utilization_{enc}_crag_proxy"] = f"UtilCRAG_{short}"
        rename[f"reasoning_{enc}_cosine"] = f"LLM_{short}"
        rename[f"reasoning_{enc}_crag_proxy"] = f"LLMCRAG_{short}"
    return rename


def present_metric_columns(df_columns: List[str]) -> List[str]:
    """Metric columns that exist in a DataFrame."""
    colset = set(df_columns)
    return [c for c in all_metric_columns() if c in colset]


def method_order_for_df(methods: List[str]) -> List[str]:
    known = [m for m in RERANK_METHOD_ORDER if m in methods]
    extra = sorted(set(methods) - set(known))
    return known + extra
