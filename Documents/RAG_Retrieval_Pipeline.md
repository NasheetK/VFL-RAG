## RAG Retrieval Pipeline (Part 2)

This project’s action-planning notebook (`RAG_part2_agent_actions.ipynb`) uses a **multi-stage retrieval + ranking** pipeline to assemble better LLM context from the knowledge base.

### What runs before the LLM

For each sample, the notebook builds one or more retrieval queries (configurable via `QUERY_STRATEGY`) and runs:

1) **Per-query child retrieval (20/query)**
- Pulls candidate **child chunks** from FAISS.
- Keeps results at the *chunk* level so ranking is precise and fast.

2) **Merge + dedupe (child identity)**
- Combines candidates from all queries.
- Removes duplicates while keeping the most relevant candidate.
- Prefers a stable child identity: **`(parent_id, child_index)`** when available.

3) **MMR (diversity)**
- Applies **Maximal Marginal Relevance** to avoid selecting near-duplicate chunks.
- Improves coverage: instead of 5 chunks repeating the same paragraph, you get different aspects (mitigation, detection, response, tier-specific details).

4) **Reranking (precision)**
- **REQUIRED CrossEncoder** reranking: stronger semantic matching than bi-encoder similarity.
- No fallback: the pipeline raises a clear error if `sentence-transformers` / the cross-encoder model is missing.

5) **Top ranked child chunks**
- After reranking, only the best chunks are retained.
- This keeps the next step (parent expansion) focused and reduces noisy context.

6) **Parent document expansion**
- Expands the winning child chunks back to their full parent sections using `rag_parents.json`.
- Expansion happens **after** reranking so the ranking remains chunk-accurate.

7) **Final context (Top 5 sections)**
- Produces **top 5 unique contextual sections** (full text) and injects them into the LLM prompt.

### Why this improves the system

- **Higher relevance**: CrossEncoder reranking is better at precise matching than vector similarity alone.
- **Lower redundancy**: merge/dedupe + MMR prevent the prompt from being filled with repeated or overlapping text.
- **Better coverage**: diversified chunks + parent expansion yields broader, tier-aware context (RAN/Edge/Core) without overwhelming the LLM with unrelated pages.
- **More stable outputs**: a consistent retrieval pipeline reduces run-to-run variability in action plans.
- **Cost/latency control**: expensive steps run only on a smaller candidate pool (post-merge/dedupe; post-MMR).

### Configuration

In `RAG_part2_agent_actions.ipynb`:

- **`QUERY_STRATEGY`**:
  - String: `"template" | "rephrase" | "llm_concise" | "llm_expanded"`
  - List: `["template", "rephrase", ...]` (multi-query)

Defaults (defined in the notebook retrieval cell):
- **20** retrieved child chunks per query
- **Top 5** contextual sections passed to the LLM

