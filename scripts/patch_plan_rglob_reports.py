#!/usr/bin/env python3
"""Patch Plan.ipynb rerank report loader: rglob + all sample IDs by default."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Plan.ipynb"

OLD_IDS_BLOCK = """    ids = sample_ids if sample_ids is not None else list(range(9))
    out: List[Tuple[int, Path]] = []"""

NEW_IDS_BLOCK = """    if sample_ids is not None:
        ids = list(sample_ids)
    else:
        ids = sorted(by_sid.keys())
    if not ids:
        raise FileNotFoundError(
            f"No rerank_comparison_sample_*.json under {action_dir} "
            f"(searched recursively under output/plans/<CLASS>/)."
        )
    out: List[Tuple[int, Path]] = []"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def load_nine_rerank_reports" not in src:
            continue
        new = src
        new = new.replace(
            'paths = list(action_dir.glob("rerank_comparison_sample_*.json"))',
            'paths = list(action_dir.rglob("rerank_comparison_sample_*.json"))',
        )
        if OLD_IDS_BLOCK in new:
            new = new.replace(OLD_IDS_BLOCK, NEW_IDS_BLOCK)
        old_err = 'f"No rerank_comparison_sample_{sid}_*.json under {action_dir}"'
        new_err = (
            'f"No rerank_comparison_sample_{sid}_*.json under {action_dir} "\n'
            '                f"(searched recursively, including output/plans/<CLASS>/)."'
        )
        if old_err in new and "searched recursively" not in new:
            new = new.replace(old_err, new_err)
        new = new.replace(
            "_reports = load_nine_rerank_reports(_action_dir)\n",
            "_reports = load_nine_rerank_reports(_action_dir)\n"
            "print(f\"Loaded {len(_reports)} rerank comparison report(s).\")\n",
            1,
        )
        if new != src:
            cell["source"] = [line + "\n" for line in new.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            changed += 1
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Patched {changed} cell(s)")


if __name__ == "__main__":
    main()
