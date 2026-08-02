#!/usr/bin/env python3
"""Patch Predict, Plan, Score notebooks for output/{inference,plans,scores}/<CLASS> layout."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_notebook(path: Path, replacements: list[tuple[str, str]]) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell.get("source", "")
        new_src = src
        for old, new in replacements:
            if old in new_src:
                new_src = new_src.replace(old, new)
        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return changed


PLAN_OLD = '''VECTOR_STORE_DIR = Path("RAG_docs/vector_store")
PREDICTIONS_DIR = Path("RAG_docs/predictions")
RESULTS_DIR = Path("RAG_docs/action_plans")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)'''

PLAN_NEW = '''from utils.pipeline_paths import (
    INFERENCE_ROOT,
    PLANS_ROOT,
    SCORES_ROOT,
    VECTOR_STORE_DIR,
    ensure_pipeline_dirs,
    plans_dir_for,
)

ensure_pipeline_dirs()
PREDICTIONS_DIR = INFERENCE_ROOT
PLANS_ROOT.mkdir(parents=True, exist_ok=True)'''

PLAN_LOAD = "predictions_data = load_predictions(PREDICTIONS_DIR, verbose=True)"
PLAN_LOAD_NEW = "predictions_data = load_predictions_from_class_folders(INFERENCE_ROOT, verbose=True)"

PLAN_EXEC_OLD = '''for sample in samples_for_actions:
    run_agent_pipeline_for_sample(
        sample,
        vector_store,
        attack_actions_data,
        agentic_features_data,
        RESULTS_DIR,'''

PLAN_EXEC_NEW = '''for sample in samples_for_actions:
    _class_plans_dir = plans_dir_for(label_from_sample(sample))
    run_agent_pipeline_for_sample(
        sample,
        vector_store,
        attack_actions_data,
        agentic_features_data,
        _class_plans_dir,'''

PLAN_DONE = 'print("Done. Action plans in:", RESULTS_DIR)'
PLAN_DONE_NEW = 'print("Done. Action plans under:", PLANS_ROOT)'

SCORE_RAG = 'RAG_DOCS_DIR = Path("RAG_docs/action_plans")'
SCORE_RAG_NEW = '''from utils.pipeline_paths import PLANS_ROOT, SCORES_ROOT, ensure_pipeline_dirs

ensure_pipeline_dirs()
PLANS_ROOT = PLANS_ROOT  # noqa: PLR004 — notebook alias
RAG_DOCS_DIR = PLANS_ROOT'''

SCORE_LOAD_FN = '''def load_rerank_comparison_files(directory: Path) -> List[Dict[str, Any]]:
    """Load rerank ablation JSON files from the directory."""
    comparison_files = sorted(directory.glob("rerank_comparison_sample_*.json"))'''

SCORE_LOAD_FN_NEW = '''def load_rerank_comparison_files(directory: Path) -> List[Dict[str, Any]]:
    """Load rerank ablation JSON files from the directory (class subfolders when recursive)."""
    from utils.rerank_plan_scoring import load_rerank_comparison_files as _load_rerank

    return _load_rerank(directory, recursive=True)'''

# Simpler: just patch RAG_DOCS_DIR and comparison_data load + save block

SCORE_SAVE_OLD = '''    scoring_dir = Path("RAG_docs/scoring")
    scoring_dir.mkdir(parents=True, exist_ok=True)

    output_file = scoring_dir / "scoring_results.json"'''

SCORE_SAVE_NEW = '''    from utils.pipeline_paths import save_scores_by_class

    _written = save_scores_by_class(
        scoring_results,
        adaptive_thresholds=adaptive,
        metric_notes=output_data.get("metric_notes"),
    )
    print("Per-class score files:")
    for _cls, _p in sorted(_written.items()):
        print(f"  {_cls}: {_p}")
    output_file = SCORES_ROOT / "scoring_results_all_classes.json"'''

# The Score save block builds output_data then writes - need more careful patch
# Read Score save cell and do surgical replace

PREDICT_SAVE_MARKER = '# 4. Save Results to Output Folder'
PREDICT_SAVE_INSERT = '''# 4. Save Results to Output Folder (per-class under output/inference/)
# -----------------------------
from utils.pipeline_paths import save_inference_by_class, INFERENCE_ROOT, ensure_pipeline_dirs

ensure_pipeline_dirs()
'''


def patch_predict():
    path = ROOT / "Predict.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if "OUTPUT_DIR = Path(\"outputs\")" in src and "save_inference_by_class" not in src:
            src = src.replace(
                'OUTPUT_DIR = Path("outputs")\nOUTPUT_DIR.mkdir(exist_ok=True)',
                'from utils.pipeline_paths import INFERENCE_ROOT, ensure_pipeline_dirs, save_inference_by_class\n\nensure_pipeline_dirs()\nOUTPUT_DIR = INFERENCE_ROOT  # legacy alias in this cell',
            )
            # Replace detailed json save with per-class save
            old_block = '''# Save detailed JSON
detailed_file = OUTPUT_DIR / f"predictions_detailed_{timestamp}.json"
with open(detailed_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✓ Detailed results saved to: {detailed_file}")'''
            new_block = '''# Save per-class prediction JSON under output/inference/<CLASS>/
_written = save_inference_by_class(results, timestamp=timestamp)
for _cls, _path in sorted(_written.items()):
    print(f"✓ {_cls}: {_path}")
detailed_file = INFERENCE_ROOT / f"predictions_all_{timestamp}.json"
with open(detailed_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✓ Combined detailed JSON: {detailed_file}")'''
            if old_block in src:
                src = src.replace(old_block, new_block)
            cell["source"] = src.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("Predict.ipynb patched")
    return changed


def patch_plan():
    ok = patch_notebook(
        ROOT / "Plan.ipynb",
        [
            (PLAN_OLD, PLAN_NEW),
            (PLAN_LOAD, PLAN_LOAD_NEW),
            (PLAN_EXEC_OLD, PLAN_EXEC_NEW),
            (PLAN_DONE, PLAN_DONE_NEW),
            (
                "- `VECTOR_STORE_DIR` / `PREDICTIONS_DIR` / `RESULTS_DIR`: I/O contract for loading FAISS + predictions and saving action plans.",
                "- `VECTOR_STORE_DIR` / `INFERENCE_ROOT` / `PLANS_ROOT`: load FAISS + `output/inference/<CLASS>/` predictions; save plans under `output/plans/<CLASS>/`.",
            ),
            (
                "**This notebook** loads the **saved vector store only** (no re-reading `RAG_docs/knowledge/`) and writes JSON under `RAG_docs/action_plans/` (including optional rerank comparison reports).",
                "**This notebook** loads predictions from `output/inference/<CLASS>/` and writes action plans under `output/plans/<CLASS>/` (including rerank comparison reports).",
            ),
        ],
    )
    print(f"Plan.ipynb: {'patched' if ok else 'unchanged'}")
    return ok


def patch_score():
    path = ROOT / "Score.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
        new_src = src

        if 'RAG_DOCS_DIR = Path("RAG_docs/action_plans")' in new_src and "PLANS_ROOT" not in new_src:
            new_src = new_src.replace(
                'RAG_DOCS_DIR = Path("RAG_docs/action_plans")\n\nprint(f"Action plans directory: {RAG_DOCS_DIR}")',
                'from utils.pipeline_paths import PLANS_ROOT, SCORES_ROOT, ensure_pipeline_dirs\n\nensure_pipeline_dirs()\nRAG_DOCS_DIR = PLANS_ROOT\n\nprint(f"Action plans directory: {RAG_DOCS_DIR} (recursive per-class)")',
            )

        if 'comparison_files = sorted(directory.glob("rerank_comparison_sample_*.json"))' in new_src:
            new_src = new_src.replace(
                'comparison_files = sorted(directory.glob("rerank_comparison_sample_*.json"))',
                'comparison_files = sorted(directory.rglob("rerank_comparison_sample_*.json"))',
            )

        if 'scoring_dir = Path("RAG_docs/scoring")' in new_src and "save_scores_by_class" not in new_src:
            # First scoring_dir block (main save)
            if 'output_file = scoring_dir / "scoring_results.json"' in new_src:
                new_src = new_src.replace(
                    '''    scoring_dir = Path("RAG_docs/scoring")
    scoring_dir.mkdir(parents=True, exist_ok=True)

    output_file = scoring_dir / "scoring_results.json"

    output_data = {''',
                    '''    from utils.pipeline_paths import save_scores_by_class, SCORES_ROOT

    output_data = {''',
                )
                new_src = new_src.replace(
                    '''    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")''',
                    '''    _class_paths = save_scores_by_class(
        scoring_results,
        adaptive_thresholds=adaptive,
        metric_notes=output_data.get("metric_notes"),
    )
    output_file = SCORES_ROOT / "scoring_results_combined.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Combined summary: {output_file}")
    print("Per-class score files:")
    for _cls, _p in sorted(_class_paths.items()):
        print(f"  {_cls}: {_p}")''',
                )

        if 'csv_path = scoring_dir / "rerank_method_averages.csv"' in new_src:
            new_src = new_src.replace(
                '''    scoring_dir = Path("RAG_docs/scoring")
    scoring_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scoring_dir / "rerank_method_averages.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\\nAverage-by-method CSV written to: {csv_path}")''',
                '''    combined_csv = SCORES_ROOT / "rerank_method_averages_all.csv"
    summary_df.to_csv(combined_csv, index=False)
    print(f"\\nCombined average-by-method CSV: {combined_csv}")
    print("(Per-class CSVs are under output/scores/<CLASS>/rerank_method_averages.csv)")''',
            )

        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True)
            changed = True

    if changed:
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("Score.ipynb patched")
    return changed


def main():
    patch_predict()
    patch_plan()
    patch_score()
    ensure_script = ROOT / "scripts" / "ensure_output_tree.py"
    ensure_script.write_text(
        '''#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pipeline_paths import ensure_pipeline_dirs
ensure_pipeline_dirs()
print("Created output/inference|plans|scores class folders")
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
