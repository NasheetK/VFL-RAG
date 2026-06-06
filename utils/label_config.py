"""
Training label policy: fixed 5-class set derived from prior 9-class VFL evaluation.

Dropped classes (low precision / poor separability on CIC2017 test split):
  BOT, WEBATTACK, SSHPATATOR, OTHERS (+ rare types not in CIC2017 undersample).

Kept (required + top F1 among remainder):
  BENIGN, DDOS, DOS, PORTSCAN, FTPPATATOR
"""

from __future__ import annotations

# Primary training corpus (avoid concatenating multi-dataset CSVs → 37+ classes)
DEFAULT_TRAINING_CSV = "undersampled_CIC2017_dataset.csv"

# Five classes for VFL head and downstream RAG/action planning
TARGET_CLASSES: tuple[str, ...] = (
    "BENIGN",
    "DDOS",
    "DOS",
    "PORTSCAN",
    "FTPPATATOR",
)

DROPPED_CLASSES: tuple[str, ...] = (
    "BOT",
    "WEBATTACK",
    "SSHPATATOR",
    "OTHERS",
    "INFILTRATION",
    "HEARTBLEED",
)


def build_label_mapping(classes: tuple[str, ...] | list[str] | None = None) -> tuple[dict[str, int], dict[int, str]]:
    order = list(classes or TARGET_CLASSES)
    forward = {name: idx for idx, name in enumerate(order)}
    reverse = {idx: name for name, idx in forward.items()}
    return forward, reverse


def filter_dataframe_to_target_classes(df, label_col: str = "label_simplified"):
    """Keep only rows whose simplified label is in TARGET_CLASSES."""
    import pandas as pd

    if label_col not in df.columns:
        raise KeyError(f"Column {label_col!r} not in dataframe")
    mask = df[label_col].isin(TARGET_CLASSES)
    dropped = int((~mask).sum())
    out = df.loc[mask].copy()
    return out, dropped
