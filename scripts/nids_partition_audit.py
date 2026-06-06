#!/usr/bin/env python3
"""Audit feature partition in agentic_features.json vs model metadata deployable columns."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.action_config import get_agent_order, load_agentic_features
from utils.vfl_utils import load_agent_definitions, split_features_by_agent_definitions

REPORT_PATH = ROOT / "RAG_docs" / "nids_partition_audit_report.txt"


def _all_deployable_columns(metadata_path: Path) -> list[str]:
    if not metadata_path.exists():
        return []
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    cols: list[str] = []
    for key in ("agent1_features", "agent2_features", "agent3_features"):
        cols.extend(meta.get(key) or [])
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def main() -> int:
    agentic_path = ROOT / "RAG_docs" / "agentic_features.json"
    meta_path = ROOT / "model" / "model_metadata.json"
    raw = load_agentic_features(agentic_path)
    order = get_agent_order(raw)
    defs = load_agent_definitions(agentic_path)
    deployable = _all_deployable_columns(meta_path)
    if not deployable:
        deployable = sorted(
            {f for feats in defs["agent_features"] for f in feats}
        )

    a1, a2, a3, _ = split_features_by_agent_definitions(deployable, defs)
    resolved = {order[i]: lst for i, lst in enumerate([a1, a2, a3]) if i < len(order)}

    feat_to_devices: dict[str, list[str]] = {}
    for dev, feats in resolved.items():
        for f in feats:
            feat_to_devices.setdefault(f, []).append(dev)

    overlaps = {f: devs for f, devs in feat_to_devices.items() if len(devs) > 1}
    assigned = set(feat_to_devices)
    unassigned = [c for c in deployable if c not in assigned]
    json_only = []
    for dev, feats in zip(order, defs["agent_features"]):
        for f in feats:
            if f not in deployable:
                json_only.append((f, dev))

    lines = [
        "NIDS feature partition audit",
        f"agent_order: {order}",
        f"deployable_columns: {len(deployable)}",
        "",
        "Per-device resolved counts:",
    ]
    for dev in order:
        lines.append(f"  {dev}: {len(resolved.get(dev, []))}")
    lines.extend([
        "",
        f"Overlapping features ({len(overlaps)}):",
    ])
    for f, devs in sorted(overlaps.items()):
        lines.append(f"  {f}: {', '.join(devs)}")
    lines.extend([
        "",
        f"Unassigned deployable columns ({len(unassigned)}):",
    ])
    for c in unassigned:
        lines.append(f"  {c}")
    if json_only:
        lines.extend(["", f"JSON features not in metadata ({len(json_only)}):"])
        for f, dev in json_only[:20]:
            lines.append(f"  {f} ({dev})")
        if len(json_only) > 20:
            lines.append(f"  ... and {len(json_only) - 20} more")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {REPORT_PATH}")
    return 1 if unassigned else 0


if __name__ == "__main__":
    raise SystemExit(main())
