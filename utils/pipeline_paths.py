"""
Pipeline I/O layout: input CSVs → inference → plans → scoring.

    input/
      <CLASS>/samples.csv   # Predict.ipynb — 50 rows per class from training data
    output/
      inference/<CLASS>/    # Predict JSON outputs
      plans/<CLASS>/        # Plan.ipynb
      scores/<CLASS>/       # Score.ipynb
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.label_config import TARGET_CLASSES

_REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = _REPO_ROOT / "input"
OUTPUT_ROOT = _REPO_ROOT / "output"
INFERENCE_ROOT = OUTPUT_ROOT / "inference"
PLANS_ROOT = OUTPUT_ROOT / "plans"
SCORES_ROOT = OUTPUT_ROOT / "scores"

# Default CSV name inside each input/<CLASS>/ folder
INPUT_SAMPLES_FILENAME = "samples.csv"
INPUT_ROWS_PER_CLASS = 50

# RAG index / KB stay under RAG_docs (not run outputs)
VECTOR_STORE_DIR = _REPO_ROOT / "RAG_docs" / "vector_store"
SECUREBERT_VECTOR_STORE_DIR = _REPO_ROOT / "RAG_docs" / "vector_store_securebert"
ATTACK_OPTIONS_PATH = _REPO_ROOT / "RAG_docs" / "attack_options.json"
AGENTIC_FEATURES_PATH = _REPO_ROOT / "RAG_docs" / "agentic_features.json"


def normalize_class_label(label: str) -> str:
    key = (label or "UNKNOWN").strip().upper()
    if key not in TARGET_CLASSES:
        raise ValueError(
            f"Label {key!r} is not in TARGET_CLASSES {list(TARGET_CLASSES)}. "
            "Filter predictions before saving to the pipeline layout."
        )
    return key


def input_dir_for(label: str) -> Path:
    return INPUT_ROOT / normalize_class_label(label)


def input_csv_for(label: str) -> Path:
    return input_dir_for(label) / INPUT_SAMPLES_FILENAME


def inference_dir_for(label: str) -> Path:
    return INFERENCE_ROOT / normalize_class_label(label)


def plans_dir_for(label: str) -> Path:
    return PLANS_ROOT / normalize_class_label(label)


def scores_dir_for(label: str) -> Path:
    return SCORES_ROOT / normalize_class_label(label)


def label_from_sample(sample: dict[str, Any]) -> str:
    return normalize_class_label(str(sample.get("predicted_label") or ""))


def ensure_input_dirs() -> None:
    """Create input/<CLASS>/ folders."""
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for cls in TARGET_CLASSES:
        input_dir_for(cls).mkdir(parents=True, exist_ok=True)


def ensure_pipeline_dirs() -> None:
    """Create input + output stage folders and per-class subfolders."""
    ensure_input_dirs()
    for root in (OUTPUT_ROOT, INFERENCE_ROOT, PLANS_ROOT, SCORES_ROOT):
        root.mkdir(parents=True, exist_ok=True)
    for cls in TARGET_CLASSES:
        inference_dir_for(cls).mkdir(parents=True, exist_ok=True)
        plans_dir_for(cls).mkdir(parents=True, exist_ok=True)
        scores_dir_for(cls).mkdir(parents=True, exist_ok=True)


def load_predict_input_dataframe(
    *,
    filename: str = INPUT_SAMPLES_FILENAME,
    verbose: bool = True,
) -> "pandas.DataFrame":
    """Load and concatenate per-class CSVs from input/<CLASS>/."""
    import pandas as pd

    ensure_input_dirs()
    frames: list[pd.DataFrame] = []
    for cls in TARGET_CLASSES:
        path = input_dir_for(cls) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input CSV for {cls}: {path}\n"
                f"Run: python scripts/build_input_class_csvs.py"
            )
        part = pd.read_csv(path)
        part["input_attack_class"] = cls
        part["input_row_index"] = range(len(part))
        frames.append(part)
        if verbose:
            print(f"  {cls}: {len(part)} rows from {path}")

    combined = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"Total inference rows: {len(combined)} ({len(TARGET_CLASSES)} classes)")
    return combined


def save_inference_by_class(
    results: list[dict[str, Any]],
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Write prediction JSON lists under output/inference/<CLASS>/."""
    ensure_pipeline_dirs()
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_class[label_from_sample(row)].append(row)

    written: dict[str, Path] = {}
    for cls in TARGET_CLASSES:
        rows = by_class.get(cls, [])
        if not rows:
            continue
        path = inference_dir_for(cls) / f"predictions_{ts}.json"
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        written[cls] = path

    manifest = {
        "timestamp": ts,
        "classes": {cls: str(p) for cls, p in written.items()},
        "total_samples": len(results),
    }
    (INFERENCE_ROOT / f"manifest_{ts}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return written


def save_scores_by_class(
    scoring_rows: list[dict[str, Any]],
    *,
    timestamp: str | None = None,
    adaptive_thresholds: dict[str, Any] | None = None,
    metric_notes: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write scoring_results.json per class under output/scores/<CLASS>/."""
    import pandas as pd

    ensure_pipeline_dirs()
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scoring_rows:
        lbl = str(row.get("predicted_label") or "UNKNOWN").strip().upper()
        if lbl in TARGET_CLASSES:
            by_class[lbl].append(row)

    from utils.cyber_encoders import scoring_metric_column_names

    written: dict[str, Path] = {}
    metric_cols = scoring_metric_column_names()

    for cls, rows in by_class.items():
        if not rows:
            continue
        out_dir = scores_dir_for(cls)
        df = pd.DataFrame(rows)
        average_scores: dict[str, dict[str, float]] = {}
        if not df.empty and "method" in df.columns:
            for method in sorted(df["method"].unique()):
                sub = df[df["method"] == method]
                average_scores[method] = {
                    m: float(sub[m].mean()) for m in metric_cols if m in sub.columns
                }

        payload: dict[str, Any] = {
            "predicted_label": cls,
            "timestamp": ts,
            "individual_scores": rows,
            "average_scores": average_scores,
        }
        if adaptive_thresholds is not None:
            payload["adaptive_thresholds_used"] = adaptive_thresholds
        if metric_notes is not None:
            payload["metric_notes"] = metric_notes

        out_path = out_dir / "scoring_results.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written[cls] = out_path

        if not df.empty and "method" in df.columns:
            csv_path = out_dir / "rerank_method_averages.csv"
            _method_display = {
                "none": "Dense MMR",
                "bm25": "BM25",
                "doct5query": "docT5",
                "crossencoder": "Dense -> CE",
                "bm25_crossencoder": "BM25 -> CE",
                "colbert": "Dense -> ColBERT",
                "none_securebert": "SecureBERT MMR",
                "crossencoder_securebert": "SecureBERT -> CE",
            }
            agg_rows = []
            for method in sorted(df["method"].unique()):
                sub = df[df["method"] == method]
                agg_rows.append(
                    {
                        "Method": _method_display.get(method, method),
                        "Recall@10": round(float(sub["recall_at_10"].mean()), 6),
                        "nDCG@10": round(float(sub["ndcg_at_10"].mean()), 6),
                        "BertscoreF1": round(float(sub["bertscore_f1"].mean()), 6),
                    }
                )
            pd.DataFrame(agg_rows).to_csv(csv_path, index=False)

            cyber_csv = out_dir / "rerank_method_averages_cyber.csv"
            cyber_rows = []
            for method in sorted(df["method"].unique()):
                sub = df[df["method"] == method]
                row_c: dict[str, Any] = {
                    "Method": _method_display.get(method, method),
                }
                for enc in ("cysecbert", "ctibert", "securebert"):
                    for stem, col in (
                        ("Recall@10", f"recall_at_10_{enc}"),
                        ("nDCG@10", f"ndcg_at_10_{enc}"),
                        ("CtxCosine", f"context_{enc}_cosine"),
                        ("ReasonCosine", f"reasoning_{enc}_cosine"),
                    ):
                        if col in sub.columns:
                            row_c[f"{enc}_{stem}"] = round(float(sub[col].mean()), 6)
                cyber_rows.append(row_c)
            if cyber_rows:
                pd.DataFrame(cyber_rows).to_csv(cyber_csv, index=False)

    return written
