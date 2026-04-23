## RAG Retrieval Pipeline (Part 2)

This project's action-planning notebook (`Plan.ipynb`) runs a **six-mode retrieval / reranking ablation** for every non-benign sample. Each mode assembles an independent top-10 context list (and a wider 100-chunk candidate pool for evaluation) so that `Score.ipynb` can compare them head-to-head on Recall@10, nDCG@10, SBERT cosine, CRAG proxy, and BERTScore F1.

### Six retrieval / reranking modes

All two-stage modes use a 100-chunk Stage-1 shortlist and select the top 10 in Stage 2. Single-stage modes produce 10 directly and report their pre-truncation top-100 as the candidate pool.

| Mode key | Stage 1 (100) | Stage 2 (10) | Notes |
|---|---|---|---|
| `none` | Dense FAISS over multi-query bundle | **MMR** diversification | Reference "no reranker" baseline; MMR is the only diversity step. |
| `bm25` | **BM25Okapi** over full child corpus | — (truncate to top 10) | Single-stage lexical retriever, BEIR-style. |
| `doct5query` | **BM25 over doc2query-expanded corpus** | — (truncate to top 10) | `castorini/doc2query-t5-base-msmarco` generates 20 synthetic queries per parent chunk; expanded corpus is cached to `RAG_docs/rag_parents_doc2query.json`. |
| `crossencoder` | Raw dense FAISS top-100 (MMR bypassed) | **CrossEncoder** (`sentence-transformers`) | Classic dense → CE reranker. |
| `bm25_crossencoder` | BM25 top-100 over full corpus | **CrossEncoder** | Lexical-first hybrid; isolates the effect of the Stage-1 retriever on CE reranking. |
| `colbert` | Raw dense FAISS top-100 (MMR bypassed) | **ColBERT v2** late-interaction | Token-level MaxSim reranker (RAGatouille + patched Windows kernel). |

### What runs before the LLM (per mode)

1. **Query construction**
   - Builds one or more retrieval queries per sample from prediction evidence (template, rephrase, optional LLM variants) via `QUERY_STRATEGY`.
2. **Stage 1 — candidate pool (100 chunks)**
   - Dense modes issue a multi-query FAISS search and deduplicate by `(parent_id, child_index)`.
   - BM25 modes run `rank_bm25.BM25Okapi` over every child chunk in the FAISS docstore.
   - `doct5query` runs the same BM25 retriever but over the T5-expanded corpus (original text + 20 synthetic queries per chunk).
3. **Stage 2 — final top-10**
   - `none` keeps the MMR-diversified dense top-10.
   - `bm25` / `doct5query` simply truncate their ranked list.
   - `crossencoder` / `bm25_crossencoder` rescore the 100 shortlisted chunks with the cross-encoder and keep the top 10.
   - `colbert` rescores with ColBERT late-interaction and keeps the top 10.
4. **Parent expansion** — top-10 child chunks are expanded to their full parent sections via `rag_parents.json` before being injected into the prompt.
5. **Report persistence** — every mode's top-10 and 100-chunk candidate pool are persisted together under `RAG_docs/action_plans/rerank_comparison_sample_<id>_<ts>.json` via `utils.rag_utils.save_rerank_comparison_report`, so the scoring notebook can compute Recall@10 / nDCG@10 denominators over the wider pool.

### Why this matters

- **Six-way ablation** isolates the effect of the Stage-1 retriever (dense / BM25 / doc2query-expanded BM25) from the Stage-2 reranker (MMR / CrossEncoder / ColBERT).
- **Shared 100-chunk pool** across two-stage modes keeps Recall@10 and nDCG@10 comparable; `Score.ipynb` uses the pool as the denominator for Recall@10, not just the 10 chunks the LLM saw.
- **Raw-dense Stage 1 for reranker modes** (no MMR) gives the cross-encoder / ColBERT the cleanest bi-encoder shortlist; MMR is reserved for the `none` baseline.
- **Full-text parent expansion** happens after ranking, so reranker decisions remain chunk-accurate while the LLM still sees richer context.

### Configuration

In `Plan.ipynb`:

- `QUERY_STRATEGY`:
  - String: `"template" | "rephrase" | "llm_concise" | "llm_expanded"`
  - List: `["template", "rephrase", ...]` (multi-query)
- `TOP_K = 10` — number of chunks passed to the LLM (Stage 2 output).
- `_RERANK_CANDIDATE_POOL_SIZE = 100` — Stage 1 shortlist size for every two-stage mode (and the reported pool size for single-stage modes).
- `_DOC2QUERY_MODEL_NAME = "castorini/doc2query-t5-base-msmarco"`, `_DOC2QUERY_NUM_QUERIES = 20` — doc2query settings.
- `RERANK_COMPARISON_MODES = ["none", "bm25", "doct5query", "crossencoder", "bm25_crossencoder", "colbert"]` — modes driven per sample.
