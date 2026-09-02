#!/usr/bin/env python3
"""Add tqdm progress bars to long-running Score.ipynb and Plan.ipynb cells."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Score.ipynb: two-pass scoring cell ---
SCORE_PASS1_OLD = """pass1_rows: List[Dict[str, Any]] = []
all_pool_grades: List[float] = []
all_context_sberts: List[float] = []
all_context_rouges: List[float] = []

for comp_data in comparison_data:"""

SCORE_PASS1_NEW = """from utils.notebook_progress import progress, progress_bar

pass1_rows: List[Dict[str, Any]] = []
all_pool_grades: List[float] = []
all_context_sberts: List[float] = []
all_context_rouges: List[float] = []

_pass1_total = sum(
    1
    for _cd in comparison_data
    if isinstance(_cd.get("modes"), dict)
    for _mk in RERANK_MODES
    if isinstance((_cd.get("modes") or {}).get(_mk), dict)
)
print(f"Pass 1: grading up to {_pass1_total} (sample, mode) pairs...")
_pass1_bar = progress_bar(total=_pass1_total, desc="Pass 1 grade", unit="pair")

for comp_data in comparison_data:"""

SCORE_PASS1_APPEND = """        pass1_rows.append({
            \"file_name\": file_name,"""

SCORE_PASS1_APPEND_NEW = """        pass1_rows.append({
            \"file_name\": file_name,"""

# Insert pbar update before pass1_rows.append - need unique anchor
SCORE_PASS1_UPDATE_ANCHOR = """        if retrieved_context:
            all_context_sberts.append(context_sbert)
            all_context_rouges.append(context_rouge1)

        pass1_rows.append({"""

SCORE_PASS1_UPDATE_NEW = """        if retrieved_context:
            all_context_sberts.append(context_sbert)
            all_context_rouges.append(context_rouge1)

        _pass1_bar.update(1)
        _pass1_bar.set_postfix(sample=sample_id, mode=mode_key, refresh=False)

        pass1_rows.append({"""

SCORE_PASS1_CLOSE = """adaptive = derive_adaptive_thresholds(
    per_chunk_grades=all_pool_grades,"""

SCORE_PASS1_CLOSE_NEW = """_pass1_bar.close()
print(f"Pass 1 complete: {len(pass1_rows)} row(s) collected.")

adaptive = derive_adaptive_thresholds(
    per_chunk_grades=all_pool_grades,"""

SCORE_PASS2_OLD = """scoring_results: List[Dict[str, Any]] = []
current_file = None

for row in pass1_rows:"""

SCORE_PASS2_NEW = """scoring_results: List[Dict[str, Any]] = []
current_file = None

print(f"Pass 2: Recall/nDCG, CRAG, BERTScore for {len(pass1_rows)} row(s)...")
for row in progress(pass1_rows, desc="Pass 2 metrics", unit="row"):"""

SCORE_PASS2_PRINT_OLD = """    if file_name != current_file:
        current_file = file_name
        print(f\"\\n{'='*60}\")
        print(f\"File: {file_name} (Sample ID: {sample_id}, Label: {predicted_label})\")
        print(f\"{'='*60}\")

    recall_ndcg:"""

SCORE_PASS2_PRINT_NEW = """    recall_ndcg:"""

SCORE_PASS2_MODE_PRINT_OLD = """    print(
        f\"Mode={mode_key:22s} | Recall@10={recall_ndcg['recall_at_10']:.4f} | \"
        f\"nDCG@10={recall_ndcg['ndcg_at_10']:.4f} | \"
        f\"Ctx-SBERT={context_sbert:.4f} | Reason-SBERT={reasoning_sbert:.4f} | \"
        f\"BERT-F1={bertscore_reasoning['bertscore_f1']:.4f}\"
    )

    result_row:"""

SCORE_PASS2_MODE_PRINT_NEW = """    result_row:"""

# --- Plan.ipynb: docT5 loop ---
PLAN_DOCT5_OLD = """        for i, chunk in enumerate(missing, 1):
            key = _doct5query_cache_key(chunk)
            text = str(chunk.get(\"chunk_text\") or \"\")
            try:
                qs = _generate_synthetic_queries(text, _DOC2QUERY_NUM_QUERIES)
            except Exception as e:
                err = str(e)
                if i == 1 and (
                    isinstance(e, ImportError)
                    or \"SentencePiece\" in err
                    or \"sentencepiece\" in err.lower()
                ):
                    raise RuntimeError(
                        \"docT5 failed on the first chunk (missing dependency?). \"
                        \"In the notebook kernel: pip install sentencepiece, restart kernel, \"
                        \"delete RAG_docs/rag_parents_doc2query.json if it contains only empty expansions.\"
                    ) from e
                print(f\"  [{i}/{len(missing)}] doc2query failure ({e}); skipping cache for this chunk.\")
                qs = []
            if qs:
                cached[key] = \" \".join(qs)
            if i % 10 == 0 or i == len(missing):
                print(f\"  expanded {i}/{len(missing)} chunks\")
        _save_doct5query_cache(cached)"""

PLAN_DOCT5_NEW = """        from utils.notebook_progress import progress

        for i, chunk in enumerate(
            progress(missing, desc=\"docT5 expand\", unit=\"chunk\"), 1
        ):
            key = _doct5query_cache_key(chunk)
            text = str(chunk.get(\"chunk_text\") or \"\")
            try:
                qs = _generate_synthetic_queries(text, _DOC2QUERY_NUM_QUERIES)
            except Exception as e:
                err = str(e)
                if i == 1 and (
                    isinstance(e, ImportError)
                    or \"SentencePiece\" in err
                    or \"sentencepiece\" in err.lower()
                ):
                    raise RuntimeError(
                        \"docT5 failed on the first chunk (missing dependency?). \"
                        \"In the notebook kernel: pip install sentencepiece, restart kernel, \"
                        \"delete RAG_docs/rag_parents_doc2query.json if it contains only empty expansions.\"
                    ) from e
                tqdm.write(f\"  [{i}/{len(missing)}] doc2query failure ({e}); skipping cache.\")
                qs = []
            if qs:
                cached[key] = \" \".join(qs)
        _save_doct5query_cache(cached)"""

# Fix PLAN_DOCT5_NEW - use print instead of tqdm.write to avoid importing tqdm in function
PLAN_DOCT5_NEW = PLAN_DOCT5_NEW.replace("                tqdm.write(", "                print(")

# --- Plan.ipynb: driver loop ---
PLAN_DRIVER_OLD = """# Execute: RAG + LLM for each queued sample.
for sample in samples_for_actions:"""

PLAN_DRIVER_NEW = """# Execute: RAG + LLM for each queued sample.
from utils.notebook_progress import progress

for sample in progress(
    samples_for_actions,
    desc=\"Plan samples\",
    unit=\"sample\",
):"""

# --- Plan.ipynb: build_rag_scores_dataframe loop ---
PLAN_RAGSCORES_LOOP_OLD = """    rows: List[Dict[str, Any]] = []

    for sid, path in report_paths:"""

PLAN_RAGSCORES_LOOP_NEW = """    from utils.notebook_progress import progress

    rows: List[Dict[str, Any]] = []

    for sid, path in progress(report_paths, desc=\"RAG Recall/nDCG\", unit=\"report\"):"""

# --- Plan.ipynb: utilization metrics loop ---
PLAN_UTIL_OLD = """for _sid, _path in _reports_final:"""

PLAN_UTIL_NEW = """from utils.notebook_progress import progress

for _sid, _path in progress(_reports_final, desc=\"Context utilization\", unit=\"report\"):"""


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
    score_n = _patch_nb(
        ROOT / "Score.ipynb",
        [
            (SCORE_PASS1_OLD, SCORE_PASS1_NEW),
            (SCORE_PASS1_UPDATE_ANCHOR, SCORE_PASS1_UPDATE_NEW),
            (SCORE_PASS1_CLOSE, SCORE_PASS1_CLOSE_NEW),
            (SCORE_PASS2_OLD, SCORE_PASS2_NEW),
            (SCORE_PASS2_PRINT_OLD, SCORE_PASS2_PRINT_NEW),
            (SCORE_PASS2_MODE_PRINT_OLD, SCORE_PASS2_MODE_PRINT_NEW),
        ],
    )
    plan_n = _patch_nb(
        ROOT / "Plan.ipynb",
        [
            (PLAN_DOCT5_OLD, PLAN_DOCT5_NEW),
            (PLAN_DRIVER_OLD, PLAN_DRIVER_NEW),
            (PLAN_RAGSCORES_LOOP_OLD, PLAN_RAGSCORES_LOOP_NEW),
            (PLAN_UTIL_OLD, PLAN_UTIL_NEW),
        ],
    )
    print(f"Patched Score.ipynb: {score_n} cell(s)")
    print(f"Patched Plan.ipynb: {plan_n} cell(s)")


if __name__ == "__main__":
    main()
