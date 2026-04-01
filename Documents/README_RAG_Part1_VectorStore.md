## RAG Part 1 — Build Vector Store (`RAG_part1_build_vector_store.ipynb`)

### Purpose

Create a **persistent FAISS vector store** and a **parent store** from the knowledge base in `RAG_docs/knowledge/`. This is the indexing step that enables fast retrieval in Part 2.

### Inputs

- **Knowledge base**: `RAG_docs/knowledge/` (PDFs / other ingested documents used by the notebook)
- **Index configuration**: embedding model + chunking settings (defined in the notebook and helpers)

### Outputs

- **FAISS index**: `RAG_docs/vector_store/`
  - `index.faiss`
  - `index.pkl`
  - `rag_manifest.json`
- **Parent store**: `RAG_docs/vector_store/rag_parents.json`
  - Contains the “parent” text blocks and retrieval titles used for parent expansion in Part 2.

### What it does (high level)

- Loads and normalizes KB documents from `RAG_docs/knowledge/`
- Splits content into **parents** (longer coherent sections) and **children** (retrieval chunks)
- Embeds **child chunks** and builds FAISS
- Saves the FAISS index + `rag_parents.json` so Part 2 can retrieve child chunks and expand to parent context

### When to rerun Part 1

Rerun Part 1 whenever:
- KB files in `RAG_docs/knowledge/` change
- You change chunking or embedding settings

