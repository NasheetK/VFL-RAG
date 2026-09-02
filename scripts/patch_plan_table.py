#!/usr/bin/env python3
import json
from pathlib import Path

nb = json.loads(Path("Plan.ipynb").read_text(encoding="utf-8"))
needle = "| `colbert` | FAISS dense top-100 (raw, no MMR) | ColBERT MaxSim"
insert = (
    "| `none_securebert` | SecureBERT FAISS top-100 | MMR diversify -> top-10 | Cyber bi-encoder + MMR |\n"
    "| `crossencoder_securebert` | SecureBERT FAISS top-100 | CrossEncoder rerank -> top-10 | SecureBERT + msmarco CE |\n"
)
for cell in nb["cells"]:
    if cell.get("cell_type") != "markdown":
        continue
    src = "".join(cell.get("source", []))
    if needle in src and "none_securebert" not in src:
        src = src.replace(needle, insert + needle, 1)
        cell["source"] = [line + "\n" for line in src.splitlines()]
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
        print("added SecureBERT table rows")
Path("Plan.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)
