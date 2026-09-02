#!/usr/bin/env python3
"""Patch Score.ipynb and Plan.ipynb to load encoders on GPU when available."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCORE_IMPORT_OLD = '''print("Loading SBERT model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Libraries imported successfully!")'''

SCORE_IMPORT_NEW = '''from utils.compute_device import (
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
print("Libraries imported successfully!")'''

SCORE_BERTSCORE_OLD = '''    P, R, F1 = bert_score(candidates, references, lang='en', verbose=False)'''

SCORE_BERTSCORE_NEW = '''    P, R, F1 = bert_score(
        candidates,
        references,
        lang="en",
        verbose=False,
        device=bert_score_device(),
    )'''

SCORE_SBERT_COSINE_OLD = '''    ref_embedding = sbert_model.encode([reference])
    cand_embedding = sbert_model.encode([candidate])'''

SCORE_SBERT_COSINE_NEW = '''    encode_device = sentence_transformer_device()
    ref_embedding = sbert_model.encode([reference], device=encode_device)
    cand_embedding = sbert_model.encode([candidate], device=encode_device)'''

PLAN_SBERT_OLD = '_sbert_model = SentenceTransformer("all-MiniLM-L6-v2")'

PLAN_SBERT_NEW = '''from utils.compute_device import report_compute_device_status, sentence_transformer_device

print("Compute devices (Plan scoring cell):")
print(report_compute_device_status())
_sbert_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=sentence_transformer_device(),
)'''

PLAN_ENCODE_OLD = '    embs = _sbert_model.encode([reference, candidate])'
PLAN_ENCODE_NEW = '    embs = _sbert_model.encode([reference, candidate], device=sentence_transformer_device())'


def _patch_nb(path: Path, replacements: list[tuple[str, str]]) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new = src
        for old, new_block in replacements:
            if old in new:
                new = new.replace(old, new_block, 1)
        if new != src:
            cell["source"] = [line + "\n" for line in new.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            changed += 1
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    score_changes = _patch_nb(
        ROOT / "Score.ipynb",
        [
            (SCORE_IMPORT_OLD, SCORE_IMPORT_NEW),
            (SCORE_BERTSCORE_OLD, SCORE_BERTSCORE_NEW),
            (SCORE_SBERT_COSINE_OLD, SCORE_SBERT_COSINE_NEW),
        ],
    )
    plan_changes = _patch_nb(
        ROOT / "Plan.ipynb",
        [
            (PLAN_SBERT_OLD, PLAN_SBERT_NEW),
            (PLAN_ENCODE_OLD, PLAN_ENCODE_NEW),
        ],
    )
    print(f"Patched Score.ipynb: {score_changes} cell(s)")
    print(f"Patched Plan.ipynb: {plan_changes} cell(s)")


if __name__ == "__main__":
    main()
