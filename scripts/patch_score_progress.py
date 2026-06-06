#!/usr/bin/env python3
"""Enhance Score.ipynb progress: load cell + finer Pass 1 grading feedback."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Score.ipynb"

LOAD_OLD = """# Load rerank comparison reports (newest per sample_id under output/plans/)
comparison_data = load_rerank_comparison_files(RAG_DOCS_DIR)"""

LOAD_NEW = """# Load rerank comparison reports (newest per sample_id under output/plans/)
from utils.notebook_progress import log_progress

log_progress("Loading rerank comparison reports from output/plans/ ...")
comparison_data = load_rerank_comparison_files(RAG_DOCS_DIR)"""

GRADES_OLD = """        encoder_grades: Dict[str, Dict[str, List[float]]] = {}
        for enc in GRADING_ENCODER_KEYS:
            rg, pg = grades_for_texts(
                reference_text, ranked_chunk_texts, pool_chunk_texts, enc
            )
            encoder_grades[enc] = {\"ranked\": rg, \"pool\": pg}"""

GRADES_NEW = """        encoder_grades: Dict[str, Dict[str, List[float]]] = {}
        for enc in GRADING_ENCODER_KEYS:
            _pass1_bar.set_postfix(
                sample=sample_id, mode=mode_key, enc=enc, refresh=True
            )
            rg, pg = grades_for_texts(
                reference_text,
                ranked_chunk_texts,
                pool_chunk_texts,
                enc,
                show_progress=True,
            )
            encoder_grades[enc] = {\"ranked\": rg, \"pool\": pg}"""

UPDATE_OLD = """        _pass1_bar.update(1)
        _pass1_bar.set_postfix(sample=sample_id, mode=mode_key, refresh=False)

        pass1_rows.append({"""

UPDATE_NEW = """        _pass1_bar.update(1)
        if _pass1_bar.n == 1 or _pass1_bar.n % 25 == 0 or _pass1_bar.n == _pass1_total:
            log_progress(
                f\"  Pass 1: {_pass1_bar.n}/{_pass1_total} pairs "
                f"(sample={sample_id}, mode={mode_key})\"
            )

        pass1_rows.append({"""

IMPORT_OLD = "from utils.notebook_progress import progress, progress_bar"
IMPORT_NEW = "from utils.notebook_progress import log_progress, progress, progress_bar"

PASS2_OLD = 'print(f"Pass 2: Recall/nDCG, CRAG, BERTScore for {len(pass1_rows)} row(s)...")'
PASS2_NEW = """log_progress(
    f\"Pass 2: Recall/nDCG, CRAG, BERTScore for {len(pass1_rows)} row(s)...\"
)"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new = src
        for old, repl in (
            (LOAD_OLD, LOAD_NEW),
            (IMPORT_OLD, IMPORT_NEW),
            (GRADES_OLD, GRADES_NEW),
            (UPDATE_OLD, UPDATE_NEW),
            (PASS2_OLD, PASS2_NEW),
        ):
            if old in new:
                new = new.replace(old, repl, 1)
        if new != src:
            cell["source"] = [line + "\n" for line in new.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            changed += 1
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Patched Score.ipynb: {changed} cell(s)")


if __name__ == "__main__":
    main()
