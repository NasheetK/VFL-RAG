"""Load the gitignored ``.env`` at the repo root (e.g. ``OPENAI_API_KEY``)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_DOTENV = _REPO_ROOT / ".env"


def load_project_environment() -> None:
    """
    Load ``VFL-RAG/.env`` (always the same file relative to this package, regardless of
    Jupyter's current working directory). Then load ``.env`` from the process cwd if
    different — dotenv does not override keys already set.
    """
    load_dotenv(_REPO_DOTENV)
    load_dotenv()
