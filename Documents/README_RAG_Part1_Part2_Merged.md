## RAG Workflow (Merged) — Part 1 + Part 2

This document combines the end-to-end RAG workflow:

- **Part 1**: build the vector store and parent store (`Index.ipynb`)
- **Part 2**: retrieve + rank context and generate action plans (`Plan.ipynb`)

---

## Part 1 — Build Vector Store (`Index.ipynb`)

### Purpose

Create a **persistent FAISS vector store** and a **parent store** from `RAG_docs/knowledge/`.

### Inputs

- `RAG_docs/knowledge/` (knowledge base sources ingested by the notebook)

### Outputs

- `RAG_docs/vector_store/`
  - `index.faiss`, `index.pkl` (FAISS index)
  - `rag_manifest.json` (index metadata)
  - `rag_parents.json` (parent store for later expansion)

### When to rerun Part 1

Rerun whenever you change:
- KB files under `RAG_docs/knowledge/`
- embedding model or chunking / index settings

If Part 1 is rerun, the cached doc2query corpus (`RAG_docs/rag_parents_doc2query.json`) is also stale and will be regenerated on the next `Plan.ipynb` run.

---

## Part 2 — Action Planning (`Plan.ipynb`)

### Purpose

For each non-benign sample, produce an LLM action plan grounded in:
- condensed prediction evidence (every agent's top-3 SHAP features), and
- retrieved knowledge context from the Part 1 index, assembled under **six retrieval / reranking modes** so `Score.ipynb` can ablate them head-to-head.

### Prerequisites

- `RAG_docs/vector_store/` exists (run Part 1 first)
- `RAG_docs/predictions/*.json` exists
- `.env` contains `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`)
- Optional configs: `RAG_docs/attack_options.json`, `RAG_docs/agentic_features.json`

### Output

- `RAG_docs/action_plans/rerank_comparison_sample_<id>_<timestamp>.json` — one JSON per sample containing, for each of the six modes, the top-10 context, the 100-chunk candidate pool, and the LLM action plan.

---

## Part 2 retrieval / reranking topology (six modes)

All two-stage modes use a **100-chunk Stage-1 shortlist** and select the **top 10** for the LLM. Single-stage modes produce 10 directly and report their ranked top-100 as the pool.

| Mode | Stage 1 (100) | Stage 2 (10) |
|---|---|---|
| `none` (Dense MMR) | Dense FAISS multi-query | MMR |
| `bm25` | BM25 over full child corpus | truncate |
| `doct5query` | BM25 over doc2query-expanded corpus (`castorini/doc2query-t5-base-msmarco`, 20 synth queries / chunk, cached to `RAG_docs/rag_parents_doc2query.json`) | truncate |
| `crossencoder` (Dense → CE) | Raw dense FAISS top-100 (no MMR) | CrossEncoder |
| `bm25_crossencoder` (BM25 → CE) | BM25 top-100 | CrossEncoder |
| `colbert` (Dense → ColBERT) | Raw dense FAISS top-100 (no MMR) | ColBERT v2 late-interaction |

### What runs before the LLM (per sample, per mode)

1. **Multi-query construction** — `QUERY_STRATEGY` can be a string or list of strategies.
2. **Stage 1** — dense modes run FAISS over the multi-query bundle (deduped by `(parent_id, child_index)`); lexical modes run `rank_bm25.BM25Okapi` over the appropriate corpus.
3. **Stage 2** — MMR / CrossEncoder / ColBERT / lexical truncation depending on the mode.
4. **Child → parent expansion** — top-10 child chunks expand to full parent sections via `rag_parents.json`.
5. **LLM call + report persistence** — each mode's 10 parent sections go into the prompt; the comparison report (top-10 + 100-chunk pool + LLM output) is persisted for `Score.ipynb`.

### Why it improves the system

- **Stage-1 / Stage-2 isolation** — swapping the retriever with the reranker fixed (and vice versa) cleanly attributes precision / recall gains to the right component.
- **Comparable Recall@10 / nDCG@10** — every two-stage mode uses a 100-chunk pool as the evaluation denominator, so the scoring notebook's recall numbers are on the same footing across modes.
- **BEIR-aligned** — this gives a like-for-like comparison against BEIR and CRAG-style benchmarks (lexical, dense, doc2query, late-interaction, cross-encoder).

---

## Recommended run order

1. Run Part 1: `Index.ipynb`
2. Ensure predictions exist: `RAG_docs/predictions/*.json`
3. Run Part 2: `Plan.ipynb`
4. Score the resulting comparison reports: `Score.ipynb`
