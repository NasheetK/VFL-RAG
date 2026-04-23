"""
Replace ColBERT's ``segmented_maxsim.cpp`` on Windows: upstream includes ``pthread.h``,
which MSVC does not provide. The repo ships a ``std::thread`` implementation.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

_PATCH_NAME = "colbert_segmented_maxsim.cpp"


def _clear_segmented_maxsim_torch_build_dirs() -> None:
    """Stale ninja outputs keep failing after we replace the .cpp source."""
    roots = []
    la = os.environ.get("LOCALAPPDATA")
    if la:
        roots.append(Path(la) / "torch_extensions")
    tmp = os.environ.get("TMP") or os.environ.get("TEMP")
    if tmp:
        roots.append(Path(tmp) / "torch_extensions")
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for child in root.iterdir():
                if child.is_dir() and "segmented_maxsim" in child.name.lower():
                    shutil.rmtree(child, ignore_errors=True)
        except OSError:
            pass


def apply_colbert_segmented_maxsim_patch() -> None:
    if sys.platform != "win32":
        return

    spec = importlib.util.find_spec("colbert")
    if spec is None or not spec.origin:
        return

    colbert_dir = Path(spec.origin).resolve().parent
    target = colbert_dir / "modeling" / "segmented_maxsim.cpp"
    if not target.is_file():
        return

    patch_src = Path(__file__).resolve().parent / "patches" / _PATCH_NAME
    if not patch_src.is_file():
        return

    if target.read_text(encoding="utf-8", errors="replace") == patch_src.read_text(
        encoding="utf-8", errors="replace"
    ):
        return

    shutil.copyfile(patch_src, target)
    _clear_segmented_maxsim_torch_build_dirs()
