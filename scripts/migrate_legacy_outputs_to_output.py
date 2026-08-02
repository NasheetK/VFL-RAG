#!/usr/bin/env python3
"""Copy legacy RAG_docs predictions/plans into output/<stage>/<CLASS>/ layout."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.pipeline_paths import (
    INFERENCE_ROOT,
    PLANS_ROOT,
    ensure_pipeline_dirs,
    label_from_sample,
    plans_dir_for,
    inference_dir_for,
)


def migrate_predictions(src: Path) -> int:
    n = 0
    for fp in src.glob("*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [data]
        by_class: dict[str, list] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            cls = label_from_sample(row)
            by_class.setdefault(cls, []).append(row)
        for cls, items in by_class.items():
            dest = inference_dir_for(cls) / fp.name
            dest.write_text(json.dumps(items, indent=2), encoding="utf-8")
            n += 1
    return n


def migrate_plans(src: Path) -> int:
    n = 0
    for fp in src.glob("rerank_comparison_sample_*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        lbl = str(data.get("predicted_label") or data.get("prediction", {}).get("predicted_label") or "")
        if not lbl:
            print(f"Skip (no label): {fp.name}")
            continue
        dest = plans_dir_for(lbl) / fp.name
        shutil.copy2(fp, dest)
        n += 1
    return n


def main() -> None:
    ensure_pipeline_dirs()
    pred_src = ROOT / "RAG_docs" / "predictions"
    plan_src = ROOT / "RAG_docs" / "action_plans"
    if pred_src.exists():
        print(f"Migrated {migrate_predictions(pred_src)} prediction file(s)")
    if plan_src.exists():
        print(f"Migrated {migrate_plans(plan_src)} plan file(s)")
    print(f"Inference root: {INFERENCE_ROOT}")
    print(f"Plans root: {PLANS_ROOT}")


if __name__ == "__main__":
    main()
