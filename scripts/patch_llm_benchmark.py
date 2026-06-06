#!/usr/bin/env python3
"""Trim LLM_Runtime_Benchmark.ipynb to four generative models only."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "LLM_Runtime_Benchmark.ipynb"

DROP_MODELS = (
    "SecureBERT",
    "DarkBERT",
    "CySecBERT",
    "CTI-BERT",
    "CyBERT",
    "PentestGPT",
)

INTRO = """# LLM runtime benchmark (generative models only)

Measures wall-clock runtime for **decoder-only** cybersecurity LLMs on one inference sample.
Encoder models (SecureBERT, CySecBERT, CTI-BERT, etc.) are evaluated in **Score.ipynb** / **Plan.ipynb**, not here.

Configure model IDs via environment variables (see `utils/llm_benchmark.py`).
"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    new_cells = []
    removed = 0

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and (
            src.strip().startswith("# LLM runtime") or src.strip().startswith("# LLM Runtime")
        ):
            cell["source"] = [line + "\n" for line in INTRO.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            cell["outputs"] = []
            cell["execution_count"] = None
            new_cells.append(cell)
            continue

        if cell["cell_type"] == "code":
            drop = any(f'benchmark_results["{name}"]' in src for name in DROP_MODELS)
            drop = drop or any(f"benchmark_results['{name}']" in src for name in DROP_MODELS)
            if drop and any(
                x in src
                for x in ("_result", "run_single_model", "benchmark_model")
            ):
                removed += 1
                continue
            # Drop cells that only define encoder benchmark runs
            if any(name in src for name in DROP_MODELS) and (
                "_result" in src or "run_single" in src.lower()
            ):
                removed += 1
                continue

        new_cells.append(cell)

    nb["cells"] = new_cells
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Removed {removed} encoder/framework cell(s).")


if __name__ == "__main__":
    main()
