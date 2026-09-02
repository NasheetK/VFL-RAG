"""Fix Plan.ipynb RAG scores cell (Recall@10) and last analysis cells."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Plan.ipynb"


def set_cell_source(cells: list, idx: int, source: str) -> None:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cells[idx]["source"] = lines
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None


RAG_SCORES_CELL = """# RAG retrieval-quality scores (rerank comparison JSON under output/plans/)
# Requires: PLANS_ROOT, agentic_features_data from earlier cells.
# Reference = description-based (utils.plan_prompts); Recall@10 uses candidate_pool
# like Score.ipynb (denominator over ~100 pool chunks, RELEVANCE_FLOOR=0.15).

from pathlib import Path

from utils.rag_retrieval_scores import build_rag_scores_dataframe, load_rerank_reports

_reports = load_rerank_reports(Path(PLANS_ROOT))
print(f"Loaded {len(_reports)} rerank comparison report(s).")
_df_scores = build_rag_scores_dataframe(
    _reports,
    agentic_features_data if isinstance(agentic_features_data, dict) else None,
)
_csv_out = Path(PLANS_ROOT) / "rag_context_rerank_keyword_scores.csv"
_df_scores.to_csv(_csv_out, index=False)
print(f"Wrote {_csv_out} ({len(_df_scores)} rows).")
try:
    from IPython.display import display

    display(_df_scores)
except Exception:
    print(_df_scores.to_string())
"""

OPTIONAL_METRICS_MD = """## Rerank report metrics (optional)

After generating `rerank_comparison_sample_*.json`, run the next cell (needs `vector_store` from the load cell).

The next cell reads reports from **`PLANS_ROOT`** (`output/plans/`, including per-class subfolders via `rglob`).

Reports include `modes.*.rag_info.candidate_pool` (**100 sections**) for Recall@k in the RAG scores cell above."""

OPTIONAL_METRICS_CELL = """# Optional: mean cosine(query, top-k chunks) + pairwise LLM field comparison per report.
try:
    from utils.rag_utils import embeddings_from_vector_store
except ImportError:
    def embeddings_from_vector_store(vector_store):
        for attr in ("embedding_function", "embeddings"):
            e = getattr(vector_store, attr, None)
            if e is not None and hasattr(e, "embed_query") and hasattr(e, "embed_documents"):
                return e
        raise ValueError(
            "Vector store has no LangChain-compatible Embeddings instance."
        )

from utils.rerank_metrics import downstream_comparison_for_report, embedding_metrics_for_report

RERANK_METRICS_REPORT_DIR = Path(PLANS_ROOT)

_MET_TOPK = 10
_emb = embeddings_from_vector_store(vector_store)
_paths = sorted(RERANK_METRICS_REPORT_DIR.rglob("rerank_comparison_sample_*.json"))
if not _paths:
    print("No rerank_comparison_sample_*.json under", RERANK_METRICS_REPORT_DIR.resolve())
else:
    _cos_acc: dict[str, list[float]] = {}
    for _p in _paths[:20]:
        import json as _json

        _rep = _json.loads(_p.read_text(encoding="utf-8"))
        if _rep.get("report_type") != "rerank_ablation_comparison":
            continue
        _e = embedding_metrics_for_report(_rep, _emb, top_k=_MET_TOPK)
        _d = downstream_comparison_for_report(_rep)
        print(f"\\n{_p.name} | sample_id={_rep.get('sample_id')}")
        _cy_modes = (_e.get("cyber_encoders") or {}).get("per_mode") or {}
        for _mk, _st in (_e.get("per_mode") or {}).items():
            v = float(_st.get("mean_top_k_cosine_query_chunk", 0.0))
            _cos_acc.setdefault(_mk, []).append(v)
            print(f"  cosine_mean_top{_MET_TOPK} [{_mk}] (index embedder): {v:.4f}")
            for _enc, _ev in (_cy_modes.get(_mk) or {}).items():
                print(f"    cyber_{_enc}: {float(_ev):.4f}")
        for _row in (_d.get("pairwise") or [])[:3]:
            print("  pair:", _row)
    print("\\n--- aggregate (up to 20 files) ---")
    for _mk, vs in sorted(_cos_acc.items()):
        print(f"  [{_mk}] mean {sum(vs)/len(vs):.4f} over {len(vs)} reports")
"""

FINAL_MD = """### Context utilization — LLM guidance vs retrieved context

Uses `agent_guidance` paragraphs (via `utils.rerank_plan_scoring.extract_overall_reasoning`) compared to the top-10 retrieved chunks per rerank mode. Reports are loaded from **`PLANS_ROOT`** (all discovered `sample_id`s).

Run after the RAG scores cell (or any cell that defines `PLANS_ROOT` and loads `agentic_features_data`)."""

FINAL_CELL = """# Context utilization — LLM agent_guidance vs retrieved top-10 context, per rerank mode.
import json
from pathlib import Path

import pandas as pd

from utils.rerank_plan_scoring import extract_overall_reasoning
from utils.rag_retrieval_scores import (
    MODE_ORDER,
    TOP_K,
    chunk_string,
    load_rerank_reports,
    rouge1_f,
    sbert_cosine,
)

_MODE_DISPLAY = {
    "none": "Dense MMR",
    "bm25": "BM25",
    "doct5query": "docT5",
    "crossencoder": "Dense -> CE",
    "bm25_crossencoder": "BM25 -> CE",
    "colbert": "Dense -> ColBERT",
}
_CRAG_SBERT_THRESHOLD = 0.55
_CRAG_ROUGE1_THRESHOLD = 0.12


def _crag_style_truthfulness(sbert: float, rouge1: float, has_answer: bool) -> float:
    if not has_answer:
        return 0.5
    if sbert >= _CRAG_SBERT_THRESHOLD or rouge1 >= _CRAG_ROUGE1_THRESHOLD:
        return 1.0
    return 0.0


_reports_final = load_rerank_reports(Path(PLANS_ROOT))

_per_mode_sbert: dict[str, list[float]] = {mk: [] for mk in MODE_ORDER}
_per_mode_crag: dict[str, list[float]] = {mk: [] for mk in MODE_ORDER}

for _sid, _path in _reports_final:
    with open(_path, encoding="utf-8") as _f:
        _rep = json.load(_f)
    _modes = _rep.get("modes") or {}
    for _mk in MODE_ORDER:
        _block = _modes.get(_mk) or {}
        _top = ((_block.get("rag_info") or {}).get("top_results")) or []
        _ctx = "\\n\\n".join(chunk_string(d) for d in _top[:TOP_K]).strip()
        _llm_text = extract_overall_reasoning(_block.get("llm_action_plan"))
        if not _ctx or not _llm_text:
            continue
        _sb = sbert_cosine(_llm_text, _ctx)
        _r1 = rouge1_f(_llm_text, _ctx)
        _crag = _crag_style_truthfulness(_sb, _r1, has_answer=True)
        _per_mode_sbert[_mk].append(_sb)
        _per_mode_crag[_mk].append(_crag)

_rows_out = []
for _mk in MODE_ORDER:
    _sb_list = _per_mode_sbert[_mk]
    _crag_list = _per_mode_crag[_mk]
    _rows_out.append(
        {
            "Method": _MODE_DISPLAY[_mk],
            "Util_SBert": round(sum(_sb_list) / len(_sb_list), 4) if _sb_list else float("nan"),
            "Util_CRAG": round(sum(_crag_list) / len(_crag_list), 4) if _crag_list else float("nan"),
        }
    )

df_reasoning_vs_context = pd.DataFrame(_rows_out).set_index("Method")
_n = len(_reports_final)

print(f"Context Utilization — agent_guidance vs retrieved top-10 (mean over samples in PLANS_ROOT, n={_n})")
print("Util_SBert = SBERT cosine(retrieved_context, agent_guidance)")
print("Util_CRAG  = 3-valued truthfulness proxy in [0, 1]")
print(df_reasoning_vs_context.to_string())

try:
    from IPython.display import display

    display(df_reasoning_vs_context)
except Exception:
    pass
"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    to_remove = []
    for i, c in enumerate(cells):
        s = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and (
            "9 × 6" in s or "LLM vs reference" in s and "sample_for_analysis" in s
        ):
            to_remove.append(i)

    for i in reversed(to_remove):
        cells.pop(i)

    for i, c in enumerate(cells):
        s = "".join(c.get("source", []))
        if c["cell_type"] == "code" and "RAG retrieval-quality scores" in s:
            set_cell_source(cells, i, RAG_SCORES_CELL)
        elif c["cell_type"] == "markdown" and "Rerank report metrics (optional)" in s:
            set_cell_source(cells, i, OPTIONAL_METRICS_MD)
        elif c["cell_type"] == "code" and "RERANK_METRICS_REPORT_DIR" in s:
            set_cell_source(cells, i, OPTIONAL_METRICS_CELL)
        elif c["cell_type"] == "markdown" and (
            "Context utilization" in s or "LLM vs reference" in s
        ):
            set_cell_source(cells, i, FINAL_MD)
        elif c["cell_type"] == "code" and (
            "Context utilization" in s or "overall_reasoning vs retrieved" in s
        ):
            set_cell_source(cells, i, FINAL_CELL)

    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Patched", NB)


if __name__ == "__main__":
    main()
