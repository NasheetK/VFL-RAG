"""Jupyter-friendly progress bars for long notebook loops.

Uses ``tqdm.notebook`` in IPython kernels when available, else ``tqdm.auto``.
Set ``VFL_NO_PROGRESS=1`` to disable bars entirely.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

_TQDM_CLS: Any = None


def progress_enabled() -> bool:
    return os.environ.get("VFL_NO_PROGRESS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def _tqdm_class():
    global _TQDM_CLS
    if _TQDM_CLS is not None:
        return _TQDM_CLS
    if not progress_enabled():
        return None
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None and hasattr(ip, "kernel"):
            from tqdm.notebook import tqdm as tqdm_cls

            _TQDM_CLS = tqdm_cls
            return _TQDM_CLS
    except Exception:
        pass
    from tqdm.auto import tqdm as tqdm_cls

    _TQDM_CLS = tqdm_cls
    return _TQDM_CLS


def progress(
    iterable: Iterable[T],
    *,
    desc: str = "",
    total: Optional[int] = None,
    unit: str = "it",
    leave: bool = True,
    mininterval: float = 0.3,
    disable: bool = False,
    **kwargs: Any,
) -> Iterator[T]:
    """Wrap *iterable* with a tqdm bar when progress is enabled."""
    if disable or not progress_enabled():
        return iter(iterable)
    tqdm_cls = _tqdm_class()
    if tqdm_cls is None:
        return iter(iterable)
    return tqdm_cls(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        mininterval=mininterval,
        dynamic_ncols=True,
        **kwargs,
    )


def progress_bar(
    *,
    total: int,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    mininterval: float = 0.3,
    disable: bool = False,
    **kwargs: Any,
):
    """Return a tqdm instance (use as context manager or call ``.update()`` manually)."""
    if disable or not progress_enabled():

        class _DummyBar:
            n = 0

            def update(self, n: int = 1) -> None:
                self.n += n
                return None

            def set_postfix(self, *args: Any, **kw: Any) -> None:
                return None

            def close(self) -> None:
                return None

            def __enter__(self):
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        return _DummyBar()

    tqdm_cls = _tqdm_class()
    if tqdm_cls is None:
        return progress_bar(total=total, desc=desc, unit=unit, disable=True)

    return tqdm_cls(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        mininterval=mininterval,
        dynamic_ncols=True,
        **kwargs,
    )


def log_progress(msg: str) -> None:
    """Print a one-line status message that flushes immediately in notebooks."""
    print(msg, flush=True)
