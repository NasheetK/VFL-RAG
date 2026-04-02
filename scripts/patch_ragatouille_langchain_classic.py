"""Patch installed RAGatouille for LangChain 1.x (one-time per venv, or after upgrading ragatouille).

RAGatouille 0.0.9 imports ``langchain.retrievers.document_compressors.base``; that path was
removed in ``langchain>=1``. The same ``BaseDocumentCompressor`` lives under ``langchain_classic``.

Requires: ``pip install langchain-classic`` (see requirements.txt).

Usage:
    python scripts/patch_ragatouille_langchain_classic.py
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

OLD = "from langchain.retrievers.document_compressors.base import BaseDocumentCompressor"
NEW = "from langchain_classic.retrievers.document_compressors.base import BaseDocumentCompressor"

FILES = ("RAGPretrainedModel.py", Path("integrations") / "_langchain.py")


def main() -> int:
    spec = find_spec("ragatouille")
    if spec is None or not spec.origin:
        print("ragatouille is not installed; nothing to patch.", file=sys.stderr)
        return 1

    root = Path(spec.origin).resolve().parent
    changed = 0
    for rel in FILES:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        if NEW in text:
            print(f"skip (already patched): {path}")
            continue
        if OLD not in text:
            print(f"skip (unexpected content): {path}", file=sys.stderr)
            continue
        path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
        print(f"patched: {path}")
        changed += 1
    if changed:
        print(f"Done. Patched {changed} file(s). Re-run if you reinstall/upgrade ragatouille.")
    else:
        print("No files changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
