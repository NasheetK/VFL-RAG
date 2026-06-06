#!/usr/bin/env python3
"""Harden Plan.ipynb docT5: require sentencepiece, ignore empty cache, fail fast."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Plan.ipynb"

ENSURE_OLD = '''def _ensure_doct5query_model() -> Tuple[Any, Any]:
    """Lazy-load the T5 doc2query model + tokenizer."""
    global _DOC2QUERY_T5, _DOC2QUERY_TOKENIZER, _DOC2QUERY_CONFIRMED
    if _DOC2QUERY_T5 is not None and _DOC2QUERY_TOKENIZER is not None:
        return (_DOC2QUERY_T5, _DOC2QUERY_TOKENIZER)
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "docT5 requires `transformers` and `sentencepiece`. "
            "Install: pip install transformers sentencepiece"
        ) from e'''

ENSURE_NEW = '''def _ensure_doct5query_model() -> Tuple[Any, Any]:
    """Lazy-load the T5 doc2query model + tokenizer."""
    global _DOC2QUERY_T5, _DOC2QUERY_TOKENIZER, _DOC2QUERY_CONFIRMED
    if _DOC2QUERY_T5 is not None and _DOC2QUERY_TOKENIZER is not None:
        return (_DOC2QUERY_T5, _DOC2QUERY_TOKENIZER)
    try:
        import sentencepiece  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "docT5 requires `sentencepiece` in the Jupyter kernel environment. "
            "Install: pip install sentencepiece  then restart the kernel."
        ) from e
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "docT5 requires `transformers` and `sentencepiece`. "
            "Install: pip install transformers sentencepiece"
        ) from e'''

LOAD_OLD = '''    exp = payload.get("expansions") or {}
    return {str(k): str(v) for k, v in exp.items()}'''

LOAD_NEW = '''    exp = payload.get("expansions") or {}
    # Ignore empty expansions (e.g. from a failed run without sentencepiece).
    return {str(k): str(v) for k, v in exp.items() if str(v).strip()}'''

BUILD_LOOP_OLD = '''            try:
                qs = _generate_synthetic_queries(text, _DOC2QUERY_NUM_QUERIES)
            except Exception as e:
                print(f"  [{i}/{len(missing)}] doc2query failure ({e}); storing empty expansion.")
                qs = []
            cached[key] = " ".join(qs)'''

BUILD_LOOP_NEW = '''            try:
                qs = _generate_synthetic_queries(text, _DOC2QUERY_NUM_QUERIES)
            except Exception as e:
                err = str(e)
                if i == 1 and (
                    isinstance(e, ImportError)
                    or "SentencePiece" in err
                    or "sentencepiece" in err.lower()
                ):
                    raise RuntimeError(
                        "docT5 failed on the first chunk (missing dependency?). "
                        "In the notebook kernel: pip install sentencepiece, restart kernel, "
                        "delete RAG_docs/rag_parents_doc2query.json if it contains only empty expansions."
                    ) from e
                print(f"  [{i}/{len(missing)}] doc2query failure ({e}); skipping cache for this chunk.")
                qs = []
            if qs:
                cached[key] = " ".join(qs)'''


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cell = nb["cells"][5]
    src = "".join(cell["source"])
    for old, new, label in (
        (ENSURE_OLD, ENSURE_NEW, "_ensure_doct5query_model"),
        (LOAD_OLD, LOAD_NEW, "_load_doct5query_cache"),
        (BUILD_LOOP_OLD, BUILD_LOOP_NEW, "_build_doct5query loop"),
    ):
        if old not in src:
            raise SystemExit(f"patch_plan_doct5query: block not found ({label})")
        src = src.replace(old, new, 1)
    cell["source"] = [line + "\n" for line in src.splitlines()]
    if cell["source"] and not cell["source"][-1].endswith("\n"):
        cell["source"][-1] += "\n"
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Patched Plan.ipynb docT5 helpers.")


if __name__ == "__main__":
    main()
