#!/usr/bin/env python3
"""Point Plan.ipynb at output/inference/<CLASS>/ and output/plans/<CLASS>/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Plan.ipynb"

CELL3_OLD = '''VECTOR_STORE_DIR = Path("RAG_docs/vector_store")
PREDICTIONS_DIR = Path("RAG_docs/predictions")
RESULTS_DIR = Path("RAG_docs/action_plans")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# From FAISS manifest; persisted on rerank comparison JSON for offline metrics.
_INDEX_EMBED_MODEL = read_manifest(VECTOR_STORE_DIR).get("embed_model")'''

CELL3_NEW = '''from utils.pipeline_paths import (
    INFERENCE_ROOT,
    PLANS_ROOT,
    SECUREBERT_VECTOR_STORE_DIR,
    VECTOR_STORE_DIR,
    ensure_pipeline_dirs,
    label_from_sample,
    plans_dir_for,
)

ensure_pipeline_dirs()

# From FAISS manifest; persisted on rerank comparison JSON for offline metrics.
_INDEX_EMBED_MODEL = read_manifest(VECTOR_STORE_DIR).get("embed_model")'''

CELL4_LOAD_OLD = "predictions_data = load_predictions(PREDICTIONS_DIR, verbose=True)"
CELL4_LOAD_NEW = (
    "predictions_data = load_predictions_from_class_folders(INFERENCE_ROOT, verbose=True)"
)

IMPORTS_ADD = "    load_predictions_from_class_folders,\n"

EXEC_OLD = '''for sample in samples_for_actions:
    run_agent_pipeline_for_sample(
        sample,
        vector_store,
        attack_actions_data,
        agentic_features_data,
        RESULTS_DIR,'''

EXEC_NEW = '''for sample in samples_for_actions:
    run_agent_pipeline_for_sample(
        sample,
        vector_store,
        attack_actions_data,
        agentic_features_data,
        plans_dir_for(label_from_sample(sample)),'''


def _set_src(cell: dict, src: str) -> None:
    cell["source"] = [line + "\n" for line in src.splitlines()]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
    cell["outputs"] = []
    cell["execution_count"] = None


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    changed = 0

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        orig = src

        if cell.get("cell_type") == "markdown" and "PREDICTIONS_DIR` / `RESULTS_DIR" in src:
            src = src.replace(
                "`VECTOR_STORE_DIR` / `PREDICTIONS_DIR` / `RESULTS_DIR`: I/O contract for loading FAISS + predictions and saving action plans.",
                "`VECTOR_STORE_DIR`: FAISS indexes. Predictions load from `output/inference/<CLASS>/` only (five class folders; root combined files ignored). Plans save to `output/plans/<CLASS>/`.",
            )
            changed += 1

        if "from utils.rag_utils import (" in src and "load_predictions_from_class_folders" not in src:
            src = src.replace(
                "    load_predictions,\n",
                "    load_predictions,\n    load_predictions_from_class_folders,\n",
            )
            changed += 1

        if CELL3_OLD in src:
            src = src.replace(CELL3_OLD, CELL3_NEW)
            changed += 1

        if CELL4_LOAD_OLD in src:
            src = src.replace(CELL4_LOAD_OLD, CELL4_LOAD_NEW)
            changed += 1

        if 't in agentic_features_data for t in ("RAN", "Edge", "Core")' in src:
            src = src.replace(
                't in agentic_features_data for t in ("RAN", "Edge", "Core")',
                't in agentic_features_data for t in ("FlowProbe", "Firewall", "CoreRouter")',
            )
            changed += 1

        if EXEC_OLD in src:
            src = src.replace(EXEC_OLD, EXEC_NEW)
            changed += 1

        if 'print("Done. Action plans in:", RESULTS_DIR)' in src:
            src = src.replace(
                'print("Done. Action plans in:", RESULTS_DIR)',
                'print("Done. Action plans under:", PLANS_ROOT)',
            )
            changed += 1

        if 'print(f"  RESULTS_DIR = {RESULTS_DIR}")' in src:
            src = src.replace(
                'print(f"  RESULTS_DIR = {RESULTS_DIR}")',
                'print(f"  INFERENCE_ROOT = {INFERENCE_ROOT}")\nprint(f"  PLANS_ROOT = {PLANS_ROOT}")',
            )
            changed += 1

        if "RESULTS_DIR," in src and "run_agent_pipeline_for_sample" in src:
            src = src.replace(
                "        RESULTS_DIR,",
                "        plans_dir_for(label_from_sample(sample)),",
            )
            changed += 1

        if src != orig:
            _set_src(cell, src)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched Plan.ipynb ({changed} touch(es)).")


if __name__ == "__main__":
    main()
