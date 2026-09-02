#!/usr/bin/env python3
"""Add SecureBERT 2.0 bi-encoder FAISS build cell to Index.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Index.ipynb"

SECUREBERT_CELL = '''# SecureBERT 2.0 bi-encoder index (parallel to all-MiniLM-L6-v2 above)
# Reuses the same parent_store / rag_documents from the MiniLM build cell.
from utils.pipeline_paths import SECUREBERT_VECTOR_STORE_DIR

SB_EMBED_MODEL = "cisco-ai/SecureBERT2.0-biencoder"
sb_embeddings = SentenceTransformerEmbeddings(
    model_name=SB_EMBED_MODEL,
    model_kwargs=sentence_transformer_model_kwargs(),
    encode_kwargs={"batch_size": EMBED_ENCODE_BATCH},
)
sb_vector_store = faiss_from_documents_batched(rag_documents, sb_embeddings)
save_vector_store(
    sb_vector_store,
    SECUREBERT_VECTOR_STORE_DIR,
    knowledge_expanded,
    embed_model=SB_EMBED_MODEL,
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
    n_chunks=len(rag_documents),
    extra_manifest={
        "indexing": "parent_child_semantic",
        "n_parents": len(parent_store),
        "title_mode": TITLE_MODE,
        "max_llm_parents": MAX_LLM_PARENTS,
        "embedding_backbone": "securebert",
    },
)
save_parent_store(SECUREBERT_VECTOR_STORE_DIR, parent_store)
print(
    f"SecureBERT index: model={SB_EMBED_MODEL}, "
    f"{len(rag_documents)} vectors -> {SECUREBERT_VECTOR_STORE_DIR}"
)
'''


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "SECUREBERT_VECTOR_STORE_DIR" in src and "SB_EMBED_MODEL" in src:
            print("Index.ipynb already has SecureBERT build cell; skipping.")
            return

    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "save_parent_store(VECTOR_STORE_DIR, parent_store)" in src:
            insert_idx = i + 1
            break
    if insert_idx is None:
        raise RuntimeError("Could not find MiniLM FAISS build cell in Index.ipynb")

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in SECUREBERT_CELL.splitlines()],
    }
    if new_cell["source"]:
        new_cell["source"][-1] = new_cell["source"][-1].rstrip("\n") + "\n"

    nb["cells"].insert(insert_idx, new_cell)

    for cell in nb["cells"]:
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            if "**This notebook**" in src and "vector_store_securebert" not in src:
                cell["source"] = [
                    s.replace(
                        "`RAG_docs/vector_store/`",
                        "`RAG_docs/vector_store/` (MiniLM) and `RAG_docs/vector_store_securebert/` (SecureBERT 2.0 bi-encoder)",
                    )
                    for s in cell["source"]
                ]

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Inserted SecureBERT build cell at index {insert_idx}.")


if __name__ == "__main__":
    main()
