## RAG Part 2 — Action Planning (`Plan.ipynb`)

### Purpose

Generate **LLM action plans** for non-benign samples by combining:
- prediction signals (label + confidence + condensed SHAP features), and
- a **retrieved knowledge context** from the vector store built in Part 1.

### Prerequisites

- Run Part 1 first so `RAG_docs/vector_store/` exists:
  - FAISS index (`index.faiss`, `index.pkl`)
  - parent store (`rag_parents.json`)
- Predictions exist under: `RAG_docs/predictions/`
- Optional configs:
  - `RAG_docs/attack_options.json`
  - `RAG_docs/agentic_features.json`
- `.env` contains `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`)

### Inputs

- `RAG_docs/predictions/*.json`
- `RAG_docs/vector_store/` (from Part 1)

### Outputs

- `RAG_docs/action_plans/action_plan_sample_<id>_<timestamp>.json`

### Key improvements (retrieval before the LLM)

Part 2 uses a multi-stage retrieval/ranking pipeline to produce **top-5 unique full-text contextual sections** for the prompt:

1) **Multi-query support**
- `QUERY_STRATEGY` can be a string (single query strategy) or a list (multi-query).

2) **Per-query retrieval (20/query)**
- Fetches 20 **child chunks** per query from FAISS.

3) **Merge + dedupe**
- Merges candidates from all queries.
- Dedupe prefers `(parent_id, child_index)` when available (chunk identity).

4) **MMR (diversity)**
- Selects a diverse subset to reduce redundancy and increase topical coverage.

5) **Reranking (precision)**
- **REQUIRED CrossEncoder** reranking.
- **REQUIRED ColBERT** reranking.
- No fallback: the pipeline raises a clear error if reranker libraries/models are missing.

6) **Child → parent expansion**
- Expands only the top-ranked child chunks into their full parent text via `rag_parents.json`.

7) **Final context**
- Dedupes contextual sections and passes **top 5 full sections** (no truncation) into the LLM call.

### Why these improvements help

- **Better relevance**: reranking improves semantic precision over plain vector similarity.
- **Less repetition**: merge/dedupe + MMR prevent the prompt from being dominated by near-duplicate chunks.
- **More coverage**: diversified chunk selection yields broader mitigation/detection/response context.
- **Cleaner prompting**: prediction evidence sent to the LLM is condensed (top features per tier), avoiding large SHAP payloads.

### Configuration

In the bottom cell of `Plan.ipynb`:

- `QUERY_STRATEGY = "template"`
  - or: `QUERY_STRATEGY = ["template", "rephrase", "llm_concise"]`

The notebook defaults to passing **top 5** contextual sections into the LLM.

