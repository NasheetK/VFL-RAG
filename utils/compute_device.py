"""Shared compute device selection: prefer CUDA when available.

Notebooks and ``utils`` modules import this so behavior is consistent.
Set ``VFL_FORCE_CPU=1`` to force CPU even when CUDA is present.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def _force_cpu() -> bool:
    return os.environ.get("VFL_FORCE_CPU", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def cuda_is_available() -> bool:
    if _force_cpu():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def preferred_torch_device_string() -> str:
    return "cuda" if cuda_is_available() else "cpu"


def preferred_torch_device():
    import torch

    return torch.device(preferred_torch_device_string())


def sentence_transformer_model_kwargs(**extra: Any) -> Dict[str, Any]:
    """Arguments for ``SentenceTransformer`` via LangChain ``model_kwargs``."""
    out: Dict[str, Any] = dict(extra)
    if cuda_is_available():
        out.setdefault("device", "cuda")
    return out


def transformers_pipeline_device_int() -> int:
    """Hugging Face ``pipeline(..., device=...)``: first GPU or CPU."""
    return 0 if cuda_is_available() else -1


def cross_encoder_device_string() -> str:
    """``sentence_transformers.CrossEncoder(..., device=...)``."""
    return preferred_torch_device_string()
