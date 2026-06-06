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


def sentence_transformer_device() -> str:
    """``SentenceTransformer(..., device=...)`` and ``model.encode(..., device=...)``."""
    return preferred_torch_device_string()


def bert_score_device() -> str:
    """``bert_score.score(..., device=...)``."""
    return preferred_torch_device_string()


def hf_encoder_batch_size() -> int:
    """Default batch size for HuggingFace BERT encoders in ``utils.cyber_encoders``."""
    if os.environ.get("HF_ENCODER_BATCH_SIZE", "").strip().isdigit():
        return int(os.environ["HF_ENCODER_BATCH_SIZE"].strip())
    return 32 if cuda_is_available() else 8


def sentence_transformer_client_kwargs(**extra: Any) -> Dict[str, Any]:
    """Keyword args forwarded to ``SentenceTransformer(...)`` (includes ``device``).

    LangChain ``SentenceTransformerEmbeddings(model_kwargs=...)`` passes these
    through to the underlying client constructor.
    """
    out: Dict[str, Any] = dict(extra)
    out.setdefault("device", sentence_transformer_device())
    return out


def sentence_transformer_model_kwargs(**extra: Any) -> Dict[str, Any]:
    """Backward-compatible alias for ``sentence_transformer_client_kwargs``."""
    return sentence_transformer_client_kwargs(**extra)


def transformers_pipeline_device_int() -> int:
    """Hugging Face ``pipeline(..., device=...)``: first GPU or CPU."""
    return 0 if cuda_is_available() else -1


def cross_encoder_device_string() -> str:
    """``sentence_transformers.CrossEncoder(..., device=...)``."""
    return preferred_torch_device_string()


def report_compute_device_status() -> str:
    """Human-readable summary for notebook startup logs."""
    lines = [
        f"Preferred PyTorch device: {preferred_torch_device_string()}",
        f"CUDA available: {cuda_is_available()}",
    ]
    if _force_cpu():
        lines.append("VFL_FORCE_CPU=1 (GPU use disabled)")
    if cuda_is_available():
        try:
            import torch

            lines.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    return "\n".join(lines)
