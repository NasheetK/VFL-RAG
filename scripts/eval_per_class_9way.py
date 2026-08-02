#!/usr/bin/env python3
"""Evaluate saved 9-class VFL model per-class on CIC2017 test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.compute_device import preferred_torch_device
from utils.model_utils import VFLModel
from utils.vfl_utils import load_agent_definitions, simplify_label, split_features_by_agent_definitions


def main() -> None:
    meta = json.loads((ROOT / "model" / "model_metadata.json").read_text(encoding="utf-8"))
    old_classes = [meta["label_mapping"][str(i)] for i in range(meta["num_classes"])]

    df = pd.read_csv(ROOT / "datasets" / "undersampled_CIC2017_dataset.csv")
    df = df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Timestamp"], errors="ignore")
    df["label_simplified"] = df["label"].apply(simplify_label)
    rev = meta["label_mapping_reverse"]
    df = df[df["label_simplified"].isin(rev.keys())]
    df["label_numeric"] = df["label_simplified"].map(rev)

    defs = load_agent_definitions(ROOT / "RAG_docs" / "agentic_features.json")
    all_feats = meta["agent1_features"] + meta["agent2_features"] + meta["agent3_features"]
    a1, a2, a3, _ = split_features_by_agent_definitions(all_feats, defs)
    X1, X2, X3 = df[a1].values.astype(np.float32), df[a2].values.astype(np.float32), df[a3].values.astype(np.float32)
    y = df["label_numeric"].values

    _, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=42)
    X1t, X2t, X3t, yt = X1[test_idx], X2[test_idx], X3[test_idx], y[test_idx]

    s1, s2, s3 = joblib.load(ROOT / "model/scaler1.pkl"), joblib.load(ROOT / "model/scaler2.pkl"), joblib.load(ROOT / "model/scaler3.pkl")
    X1t, X2t, X3t = s1.transform(X1t), s2.transform(X2t), s3.transform(X3t)

    device = preferred_torch_device()
    model = VFLModel(input_dims=[len(a1), len(a2), len(a3)], embed_dim=64, hidden_dim=128, num_classes=9).to(device)
    ckpt = torch.load(ROOT / "model/vfl_model_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(
            model([torch.tensor(X1t, device=device), torch.tensor(X2t, device=device), torch.tensor(X3t, device=device)]),
            dim=1,
        ).cpu().numpy()

    rep = classification_report(yt, pred, target_names=old_classes, zero_division=0, output_dict=True)
    print("Per-class metrics (9-class model, CIC2017 test 20%):")
    rows = []
    for c in old_classes:
        d = rep[c]
        rows.append((c, d["precision"], d["recall"], d["f1-score"], int(d["support"])))
        print(f"  {c:12} prec={d['precision']:.3f} rec={d['recall']:.3f} f1={d['f1-score']:.3f} sup={int(d['support'])}")
    rows.sort(key=lambda x: x[3])
    print("\nLowest F1 (drop candidates):", [r[0] for r in rows[:4]])
    print("Highest F1 (keep candidates):", [r[0] for r in rows[-5:]])


if __name__ == "__main__":
    main()
