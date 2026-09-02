#!/usr/bin/env python3
"""Add SecureBERT FAISS modes and dual vector-store load to Plan.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Plan.ipynb"


def _set_src(cell: dict, src: str) -> None:
    cell["source"] = [line + "\n" for line in src.splitlines()]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
    cell["outputs"] = []
    cell["execution_count"] = None


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    changed = 0

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        orig = src

        if cell["cell_type"] == "markdown" and "`rerank_mode` selects one of **six" in src:
            src = src.replace(
                "`rerank_mode` selects one of **six retrieval / reranking topologies**",
                "`rerank_mode` selects one of **eight retrieval / reranking topologies** (six MiniLM FAISS + two SecureBERT FAISS)",
            )
            src = src.replace(
                "| `colbert` | Dense FAISS top-100 | ColBERT MaxSim rerank → top-10 | Two-stage dense + late-interaction |\n",
                "| `colbert` | Dense FAISS top-100 | ColBERT MaxSim rerank → top-10 | Two-stage dense + late-interaction |\n"
                "| `none_securebert` | SecureBERT FAISS top-100 | MMR diversify → top-10 | Dense cyber bi-encoder + MMR |\n"
                "| `crossencoder_securebert` | SecureBERT FAISS top-100 | msmarco CrossEncoder rerank → top-10 | SecureBERT index + CE rerank |\n",
            )
            changed += 1

        if "from utils.pipeline_paths import (" in src and "SECUREBERT_VECTOR_STORE_DIR" not in src:
            src = src.replace(
                "    VECTOR_STORE_DIR,\n",
                "    VECTOR_STORE_DIR,\n    SECUREBERT_VECTOR_STORE_DIR,\n",
            )
            changed += 1

        if src.strip().startswith("vector_store = load_vector_store(VECTOR_STORE_DIR)"):
            src = src.replace(
                "vector_store = load_vector_store(VECTOR_STORE_DIR)\n",
                "vector_store = load_vector_store(VECTOR_STORE_DIR)\n"
                "vector_store_securebert = load_vector_store(SECUREBERT_VECTOR_STORE_DIR)\n"
                "_SB_INDEX_EMBED_MODEL = read_manifest(SECUREBERT_VECTOR_STORE_DIR).get(\n"
                '    "embed_model", "cisco-ai/SecureBERT2.0-biencoder"\n'
                ")\n",
            )
            src = src.replace(
                "f\"Vector store: {'loaded' if vector_store is not None else 'missing'}\"",
                "f\"Vector stores: MiniLM={'loaded' if vector_store is not None else 'missing'}, "
                "SecureBERT={'loaded' if vector_store_securebert is not None else 'missing'}\"",
            )
            changed += 1

        if "def retrieve_rag_context_multi(" in src:
            if "rm.endswith(\"_securebert\")" not in src:
                src = src.replace(
                    "    rm = (rerank_mode or \"crossencoder\").strip().lower()\n"
                    "    anchor_query = qs[0]\n",
                    "    rm = (rerank_mode or \"crossencoder\").strip().lower()\n"
                    "    if rm.endswith(\"_securebert\"):\n"
                    "        rm = rm[: -len(\"_securebert\")]\n"
                    "    anchor_query = qs[0]\n",
                )
            if '"none_securebert"' not in src:
                src = src.replace(
                    '    """Retrieve RAG sections under one of six retrieval / reranking topologies.',
                    '    """Retrieve RAG sections under retrieval / reranking topologies (incl. *_securebert).',
                )
            changed += 1

        if '"crossencoder_colbert": "CE + ColBERT (sum)",' in src:
            src = src.replace(
                '"crossencoder_colbert": "CE + ColBERT (sum)",\n',
                '"crossencoder_colbert": "CE + ColBERT (sum)",\n'
                '    "none_securebert": "SecureBERT dense + MMR",\n'
                '    "crossencoder_securebert": "SecureBERT dense -> CE",\n',
            )
            changed += 1

        if '            "colbert",\n        ]' in src and '"none_securebert"' not in src:
            src = src.replace(
                '            "colbert",\n        ]',
                '            "colbert",\n            "none_securebert",\n'
                '            "crossencoder_securebert",\n        ]',
            )
            changed += 1

        if "def run_agent_pipeline_for_sample(" in src and "_store_for_rerank_mode" not in src:
            helper = '''

def _store_for_rerank_mode(mode: str):
    """MiniLM FAISS by default; SecureBERT FAISS for *_securebert modes."""
    if str(mode).strip().lower().endswith("_securebert"):
        return vector_store_securebert
    return vector_store


def _embed_model_for_rerank_mode(mode: str) -> str:
    if str(mode).strip().lower().endswith("_securebert"):
        return str(_SB_INDEX_EMBED_MODEL or "cisco-ai/SecureBERT2.0-biencoder")
    return str(_INDEX_EMBED_MODEL or "")


'''
            src = src.replace(
                "def run_agent_pipeline_for_sample(",
                helper + "def run_agent_pipeline_for_sample(",
                1,
            )
            changed += 1

        if "for mode_i, mode in enumerate(modes):" in src and "_vs = _store_for_rerank_mode" not in src:
            src = src.replace(
                "            rag_candidate_pool, pq = retrieve_rag_context_multi(\n"
                "                vector_store,\n",
                "            _vs = _store_for_rerank_mode(mode)\n"
                "            rag_candidate_pool, pq = retrieve_rag_context_multi(\n"
                "                _vs,\n",
            )
            changed += 1

        if '            modes_payload[str(mode)] = {\n                "label": label,\n                "rag_results": rag_results,\n                "candidate_pool": rag_candidate_pool,\n                "llm_response": llm_response,\n            }\n' in src:
            src = src.replace(
                '            modes_payload[str(mode)] = {\n'
                '                "label": label,\n'
                '                "rag_results": rag_results,\n'
                '                "candidate_pool": rag_candidate_pool,\n'
                '                "llm_response": llm_response,\n'
                "            }\n",
                '            modes_payload[str(mode)] = {\n'
                '                "label": label,\n'
                '                "rag_results": rag_results,\n'
                '                "candidate_pool": rag_candidate_pool,\n'
                '                "llm_response": llm_response,\n'
                '                "embedding_backbone": (\n'
                '                    "securebert"\n'
                '                    if str(mode).endswith("_securebert")\n'
                '                    else "minilm"\n'
                '                ),\n'
                '                "index_embed_model": _embed_model_for_rerank_mode(mode),\n'
                "            }\n",
            )
            changed += 1

        if src != orig:
            _set_src(cell, src)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched Plan.ipynb ({changed} touch(es)).")


if __name__ == "__main__":
    main()
