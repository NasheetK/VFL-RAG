## RAG Retrieval Pipeline (Part 2)

This project's action-planning notebook (`Plan.ipynb`) runs an **eight-mode retrieval / reranking ablation** for every non-benign sample. Six modes use the **MiniLM FAISS** index (`all-MiniLM-L6-v2`); two modes use a parallel **SecureBERT 2.0 bi-encoder** index (`cisco-ai/SecureBERT2.0-biencoder`). Each mode assembles an independent top-10 context list (and a wider 100-chunk candidate pool for evaluation) so that `Score.ipynb` can compare them head-to-head.

### Retrieval / reranking modes

All two-stage modes use a 100-chunk Stage-1 shortlist and select the top 10 in Stage 2. Single-stage modes produce 10 directly and report their pre-truncation top-100 as the candidate pool.

| Mode key | FAISS index | Stage 1 (100) | Stage 2 (10) | Notes |
|---|---|---|---|---|
| `none` | MiniLM | Dense FAISS over multi-query bundle | **MMR** diversification | Reference dense baseline. |
| `bm25` | — (docstore) | **BM25Okapi** over full child corpus | — (truncate to top 10) | Single-stage lexical retriever. |
| `doct5query` | — (expanded corpus) | **BM25 over doc2query-expanded corpus** | — (truncate to top 10) | `castorini/doc2query-t5-base-msmarco`; cache `RAG_docs/rag_parents_doc2query.json`. |
| `crossencoder` | MiniLM | Raw dense FAISS top-100 (MMR bypassed) | **CrossEncoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | Dense → CE. |
| `bm25_crossencoder` | — | BM25 top-100 | **CrossEncoder** | Lexical → CE. |
| `colbert` | MiniLM | Raw dense FAISS top-100 | **ColBERT v2** | Late-interaction rerank. |
| `none_securebert` | SecureBERT | SecureBERT FAISS + MMR | MMR → top 10 | Cyber bi-encoder dense baseline. |
| `crossencoder_securebert` | SecureBERT | SecureBERT FAISS top-100 | **CrossEncoder** (same msmarco CE) | Compare SecureBERT embeddings vs MiniLM at Stage 1. |

### Dual FAISS indexes (`Index.ipynb`)

1. **MiniLM** — `RAG_docs/vector_store/` (`all-MiniLM-L6-v2`), default for modes without `_securebert` suffix.
2. **SecureBERT 2.0 bi-encoder** — `RAG_docs/vector_store_securebert/` (`cisco-ai/SecureBERT2.0-biencoder`). Run the second build cell in `Index.ipynb` after the MiniLM build (reuses the same `rag_documents` / `parent_store`).

`Plan.ipynb` loads both stores and routes `*_securebert` modes through `vector_store_securebert`.

### Three evaluation layers (`Score.ipynb`)

| Layer | What is compared | Baseline columns | Cyber encoder columns |
|---|---|---|---|
| **1. Retrieval** | Reference vs ranked / pool chunks | `recall_at_10`, `ndcg_at_10` (MiniLM-grade) | `recall_at_10_{cysecbert,ctibert,securebert}`, matching nDCG |
| **2. Retrieval → LLM** | Reference vs retrieved context; retrieved vs `agent_guidance` | `context_*`, `utilization_*` (SBERT + CRAG) | `context_{enc}_cosine`, `utilization_{enc}_cosine`, CRAG proxies |
| **3. LLM alone** | Reference vs `agent_guidance` | `reasoning_sbert_cosine`, `reasoning_crag_proxy`, `bertscore_f1` | `reasoning_{enc}_cosine`, CRAG proxies |

Graded relevance: `0.5 × ROUGE-1 + 0.5 × encoder cosine` per encoder family (`utils/cyber_encoders.py`).

Encoder HF ids:

- CySecBERT — `markusbayer/CySecBERT`
- CTI-BERT — `ibm-research/CTI-BERT`
- SecureBERT 2.0 bi-encoder — `cisco-ai/SecureBERT2.0-biencoder`

### What runs before the LLM (per mode)

1. **Query construction** — prediction evidence templates / optional LLM variants (`QUERY_STRATEGY`).
2. **Stage 1 — candidate pool (100 chunks)** — dense FAISS (MiniLM or SecureBERT), BM25, or doc2query-expanded BM25.
3. **Stage 2 — final top-10** — MMR, truncate, CrossEncoder, or ColBERT per mode table above.
4. **Parent expansion** — top-10 children → full parent sections via `rag_parents.json`.
5. **Report persistence** — `output/plans/<CLASS>/rerank_comparison_sample_<id>_<ts>.json` with `embedding_backbone` / `index_embed_model` per mode when available.

### Configuration

In `Plan.ipynb`:

- `RERANK_COMPARISON_MODES` default list includes all eight keys above.
- `_DOC2QUERY_MODEL_NAME = "castorini/doc2query-t5-base-msmarco"`, `_DOC2QUERY_NUM_QUERIES = 20`.
- `TOP_K = 10`, `_RERANK_CANDIDATE_POOL_SIZE = 100`.

In `Score.ipynb`:

- Uses `utils/cyber_encoders` for parallel encoder metrics; writes `output/scores/<CLASS>/rerank_method_averages_cyber.csv`.
