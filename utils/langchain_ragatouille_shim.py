"""
RAGatouille imports ``langchain.retrievers.document_compressors.base`` (LangChain <1.0).
LangChain 1.x moved that API to ``langchain_classic``. Register the legacy module names
in ``sys.modules`` so ``from ragatouille import ...`` works with the project's LC stack.

Import this module once before importing ``ragatouille`` (see ``Plan.ipynb``).
"""

from __future__ import annotations

import importlib
import sys
import types


def _install() -> None:
    key = "langchain.retrievers.document_compressors.base"
    if key in sys.modules:
        return

    base = importlib.import_module(
        "langchain_classic.retrievers.document_compressors.base"
    )

    doc_compressors = types.ModuleType("langchain.retrievers.document_compressors")
    doc_compressors.__path__ = []
    retrievers = types.ModuleType("langchain.retrievers")
    retrievers.__path__ = []

    sys.modules[key] = base
    sys.modules["langchain.retrievers.document_compressors"] = doc_compressors
    sys.modules["langchain.retrievers"] = retrievers


_install()
