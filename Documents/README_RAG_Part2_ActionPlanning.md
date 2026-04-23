## RAG Part 2 — Action Planning (`Plan.ipynb`)

### Purpose

Generate **LLM action plans** for non-benign samples by combining:
- prediction signals (label + confidence + condensed SHAP: all agents, top-3 features each), and
- **retrieved knowledge context** from the vector store built in Part 1, assembled under six different retrieval / reranking configurations so the downstream scoring notebook can ablate them.

### Prerequisites

- Run Part 1 first so `RAG_docs/vector_store/` exists:
  - FAISS index (`index.faiss`, `index.pkl`)
  - parent store (`rag_parents.json`)
- Predictions exist under `RAG_docs/predictions/`
- Optional configs:
  - `RAG_docs/attack_options.json`
  - `RAG_docs/agentic_features.json`
- `.env` contains `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`)

### Inputs

- `RAG_docs/predictions/*.json`
- `RAG_docs/vector_store/` (from Part 1)

### Outputs

- `RAG_docs/action_plans/rerank_comparison_sample_<id>_<timestamp>.json` — one file per sample, containing the top-10 context, 100-chunk candidate pool, and LLM action plan for every retrieval / reranking mode.
- `RAG_docs/rag_parents_doc2query.json` — cached doc2query-expanded corpus (rebuilt whenever the parent store changes).

### Retrieval / reranking topology (six modes)

All two-stage modes use a **100-chunk Stage-1 shortlist** and select the **top 10** for the LLM. Single-stage modes produce 10 directly and report their ranked top-100 as the pool.

| Mode | Stage 1 (100) | Stage 2 (10) |
|---|---|---|
| `none` (Dense MMR) | Dense FAISS multi-query | MMR |
| `bm25` | BM25 over full child corpus | truncate |
| `doct5query` | BM25 over doc2query-expanded corpus (`castorini/doc2query-t5-base-msmarco`, 20 synth queries / chunk) | truncate |
| `crossencoder` (Dense → CE) | Raw dense FAISS top-100 (no MMR) | CrossEncoder |
| `bm25_crossencoder` (BM25 → CE) | BM25 top-100 | CrossEncoder |
| `colbert` (Dense → ColBERT) | Raw dense FAISS top-100 (no MMR) | ColBERT v2 late-interaction |

### What runs before the LLM (per sample, per mode)

1. **Multi-query support** — `QUERY_STRATEGY` can be a string (single strategy) or a list (multi-query).
2. **Stage-1 candidate pool (100 chunks)** — dense modes run FAISS over the multi-query bundle and dedupe by `(parent_id, child_index)`; lexical modes run BM25 over the appropriate corpus.
3. **Stage-2 top-10** — MMR, CrossEncoder, ColBERT, or lexical truncation depending on the mode.
4. **Child → parent expansion** — top-10 child chunks are expanded into full parent sections via `rag_parents.json`.
5. **LLM call** — the 10 parent sections are injected into the prompt; OpenAI is called once per mode per sample.
6. **Report persistence** — each sample's full comparison (top-10 + 100-chunk candidate pool + LLM output for every mode) is written via `utils.rag_utils.save_rerank_comparison_report`.

### Why it improves the system

- **Stage-1 isolation** — holding the Stage-2 reranker fixed while swapping the retriever (dense vs BM25 vs doc2query-expanded) exposes how much each retriever contributes to final ranking quality.
- **Stage-2 isolation** — holding the Stage-1 retriever fixed while swapping MMR / CE / ColBERT shows the precision delta from each reranker.
- **Shared pool size** — every two-stage mode uses 100 chunks in Stage 1, so `Score.ipynb`'s Recall@10 / nDCG@10 numbers are comparable across modes.
- **Faithfulness evaluation** — because the candidate pool (100) is persisted alongside the top-10, `Score.ipynb` can compute Recall@10 over the pool rather than only over what the LLM saw.

### Configuration

Top-of-notebook parameters in `Plan.ipynb`:

- `QUERY_STRATEGY = "template"` — or a list, e.g. `["template", "rephrase", "llm_concise"]`.
- `TOP_K = 10` — LLM context size.
- `_RERANK_CANDIDATE_POOL_SIZE = 100` — Stage-1 shortlist / reported pool size.
- `_DOC2QUERY_NUM_QUERIES = 20` — synthetic queries per chunk for `doct5query`.
- `RERANK_COMPARISON_MODES = ["none", "bm25", "doct5query", "crossencoder", "bm25_crossencoder", "colbert"]`.
