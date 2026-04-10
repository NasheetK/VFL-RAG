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
- embedding model or chunking/index settings

---

## Part 2 — Action Planning (`Plan.ipynb`)

### Purpose

For each non-benign sample, produce an LLM action plan grounded in:
- condensed prediction evidence (top features per tier), and
- retrieved knowledge context from the Part 1 index.

### Prerequisites

- `RAG_docs/vector_store/` exists (run Part 1 first)
- `RAG_docs/predictions/*.json` exists
- `.env` contains `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`)
- Optional configs:
  - `RAG_docs/attack_options.json`
  - `RAG_docs/agentic_features.json`

### Output

- `RAG_docs/action_plans/action_plan_sample_<id>_<timestamp>.json`

---

## Part 2 retrieval improvements (what runs before the LLM)

Part 2 uses a multi-stage pipeline to produce **top-5 unique full-text contextual sections**:

1) **Multi-query support**
- `QUERY_STRATEGY` can be a string (single) or a list (multi).

2) **Per-query retrieval**
- Fetch **20 child chunks per query** from FAISS.

3) **Merge + dedupe**
- Merge candidates across queries.
- Dedupe prefers stable chunk identity: **`(parent_id, child_index)`** when available.

4) **MMR (diversity)**
- Maximal Marginal Relevance reduces redundancy and increases coverage.

5) **Reranking (precision)**
- **REQUIRED CrossEncoder** reranking (`sentence-transformers`)
- No fallback: the pipeline raises a clear error if the reranker dependency or model is missing.

6) **Child → parent expansion**
- Expand only top-ranked child chunks into their parent text using `rag_parents.json`.

7) **Final context to LLM**
- De-duplicate expanded sections and pass **top 5 full sections** (no truncation) into the LLM prompt.

### Why it improves the system

- **Higher relevance**: rerankers improve semantic precision.
- **Lower redundancy**: merge/dedupe + MMR avoid repeated context.
- **Better coverage**: diversified selection yields mitigation/detection/response breadth.
- **Cleaner prompting**: avoids sending full SHAP payload; uses condensed top features per tier.

---

## Recommended run order

1) Run Part 1: `Index.ipynb`
2) Ensure predictions exist: `RAG_docs/predictions/*.json`
3) Run Part 2: `Plan.ipynb`

