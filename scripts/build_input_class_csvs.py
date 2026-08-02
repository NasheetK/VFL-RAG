#!/usr/bin/env python3
"""Populate input/<CLASS>/samples.csv with first N rows per class from training CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.label_config import DEFAULT_TRAINING_CSV, TARGET_CLASSES, filter_dataframe_to_target_classes
from utils.pipeline_paths import (
    INPUT_ROWS_PER_CLASS,
    INPUT_SAMPLES_FILENAME,
    ensure_input_dirs,
    input_csv_for,
)
from utils.vfl_utils import simplify_label


def main() -> None:
    src = ROOT / "datasets" / DEFAULT_TRAINING_CSV
    if not src.exists():
        raise FileNotFoundError(f"Training dataset not found: {src}")

    print(f"Reading {src.name}...")
    df = pd.read_csv(src)
    df["label_simplified"] = df["label"].apply(simplify_label)
    df, dropped = filter_dataframe_to_target_classes(df, "label_simplified")
    print(f"Dropped {dropped} rows outside TARGET_CLASSES")

    ensure_input_dirs()
    n_per_class = INPUT_ROWS_PER_CLASS

    for cls in TARGET_CLASSES:
        subset = df[df["label_simplified"] == cls].head(n_per_class)
        if len(subset) < n_per_class:
            print(f"  WARNING: {cls} has only {len(subset)} rows (requested {n_per_class})")
        out = input_csv_for(cls)
        subset.to_csv(out, index=False)
        print(f"  Wrote {len(subset)} rows -> {out}")

    print("Done.")


if __name__ == "__main__":
    main()
