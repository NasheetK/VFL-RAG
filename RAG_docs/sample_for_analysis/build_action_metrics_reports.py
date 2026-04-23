"""
Compute action-plan metrics vs RAG_docs/attack_options.json for each
rerank_comparison_sample_*.json in this directory.

Writes (under reports/):
- action_metrics_vs_attack_options.json
- action_metrics_per_sample_tables.md
- action_metrics_explanations.md
"""
from __future__ import annotations

import json
from pathlib import Path

RAG_DOCS = Path(__file__).resolve().parent.parent
ATTACK_OPTIONS = RAG_DOCS / "attack_options.json"
AGENTIC_FEATURES = RAG_DOCS / "agentic_features.json"
SAMPLE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SAMPLE_DIR / "reports"
MODES = [
    "none",
    "bm25",
    "doct5query",
    "crossencoder",
    "bm25_crossencoder",
    "colbert",
]
AGENT_ORDER = ["RAN", "Edge", "Core"]


def norm(s: str) -> str:
    return (s or "").strip().lower()


def _actions_from_plan(plan: dict) -> tuple[list[str], list[str], list[str]]:
    """Returns (from_all_actions, from_primary, from_supporting) as raw strings."""
    aa = plan.get("all_actions") or []
    if not isinstance(aa, list):
        aa = []
    from_all = [str(x) for x in aa if x and str(x).strip()]

    def from_rows(key: str) -> list[str]:
        out: list[str] = []
        for row in plan.get(key) or []:
            if isinstance(row, dict):
                a = row.get("action")
                if a is not None and str(a).strip():
                    out.append(str(a).strip())
        return out

    return (from_all, from_rows("primary_actions"), from_rows("supporting_actions"))


def _normalize_agent_tier(raw) -> str | None:
    """Map free-text network_tier to RAN / Edge / Core when possible."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if "edge" in s:
        return "Edge"
    if "core" in s:
        return "Core"
    if "ran" in s:
        return "RAN"
    return None


def _agent_capability_sets(agents_block: dict) -> dict[str, set[str]]:
    return {
        name: {norm(x) for x in (agents_block.get(name) or {}).get("action_capabilities") or []}
        for name in AGENT_ORDER
        if name in agents_block
    }


def _agents_with_capability(
    act_n: str, caps_by_agent: dict[str, set[str]]
) -> list[str]:
    """Agents (RAN/Edge/Core order) whose action_capabilities include this action."""
    if not act_n:
        return []
    return [a for a in AGENT_ORDER if act_n in caps_by_agent.get(a, set())]


def _strict_tier_action_match(
    act_n: str, tier_n: str | None, caps_by_agent: dict[str, set[str]]
) -> bool:
    """True iff LLM network_tier agent lists this action in agentic action_capabilities."""
    if not act_n or not tier_n or tier_n not in caps_by_agent:
        return False
    return act_n in caps_by_agent[tier_n]


def compute_agentic_multi_layer(
    plan: dict, agents_block: dict
) -> tuple[bool, list[str], list[dict]]:
    """
    Multi-layer iff >=2 distinct agents (RAN/Edge/Core) each have at least one
    primary/supporting row where network_tier (normalized) matches an agent name
    and that agent's action_capabilities (from agentic_features.json) contains
    the same action string (normalized).

    all_actions entries are not counted toward layers (no LLM network_tier);
    they are still audited for which agents list the action in capabilities.
    """
    caps = _agent_capability_sets(agents_block)
    agents_used: set[str] = set()
    audits: list[dict] = []

    def audit_ps_row(source: str, row: dict) -> None:
        if not isinstance(row, dict):
            return
        act_raw = str(row.get("action") or "").strip()
        act_n = norm(act_raw)
        if not act_n:
            return
        tier_raw = row.get("network_tier")
        tier_n = _normalize_agent_tier(tier_raw)
        holders = _agents_with_capability(act_n, caps)
        match = _strict_tier_action_match(act_n, tier_n, caps)
        audits.append(
            {
                "source": source,
                "action": act_raw or None,
                "network_tier_raw": str(tier_raw).strip() if tier_raw is not None else None,
                "network_tier_normalized": tier_n,
                "agentic_match": match,
                "action_capabilities_agents": holders,
            }
        )
        if match and tier_n:
            agents_used.add(tier_n)

    for row in plan.get("primary_actions") or []:
        audit_ps_row("primary_actions", row)
    for row in plan.get("supporting_actions") or []:
        audit_ps_row("supporting_actions", row)

    from_all, _, _ = _actions_from_plan(plan)
    for act in from_all:
        na = norm(str(act).strip())
        audits.append(
            {
                "source": "all_actions",
                "action": str(act).strip(),
                "network_tier_raw": None,
                "network_tier_normalized": None,
                "agentic_match": None,
                "action_capabilities_agents": _agents_with_capability(na, caps),
                "note": "no network_tier; excluded from multi_layer agent count",
            }
        )

    ordered = [a for a in AGENT_ORDER if a in agents_used]
    return (len(agents_used) >= 2, ordered, audits)


def summarize_ps_agentic_alignment(audits: list[dict]) -> dict:
    """
    Among primary/supporting rows with an action: fraction where LLM network_tier
    agent lists that action in agentic_features action_capabilities.
    100% iff every such row matches.
    """
    ps = [
        a
        for a in audits
        if a.get("source") in ("primary_actions", "supporting_actions") and a.get("action")
    ]
    total = len(ps)
    if total == 0:
        return {
            "agentic_ps_row_count": 0,
            "agentic_ps_rows_matched": 0,
            "agentic_tier_action_alignment_pct": None,
            "agentic_tier_action_alignment_display": "0/0",
        }
    matched = sum(1 for a in ps if a.get("agentic_match") is True)
    pct = 100.0 * matched / total
    return {
        "agentic_ps_row_count": total,
        "agentic_ps_rows_matched": matched,
        "agentic_tier_action_alignment_pct": round(pct, 2),
        "agentic_tier_action_alignment_display": f"{matched}/{total}",
    }


def analyze_mode(plan: dict, allowed_raw: list[str], agents_block: dict) -> dict:
    allowed_set = {norm(a) for a in allowed_raw}
    n_allowed = len(allowed_raw)
    from_all, from_pri, from_sup = _actions_from_plan(plan)
    # Mitigation coverage & validity: union of all three sources (per ranking mode)
    emitted = [norm(x) for x in (from_all + from_pri + from_sup) if x]
    matched_norm = [a for a in emitted if a in allowed_set]
    distinct_covered_norm = sorted(set(matched_norm))
    invalid = sorted({a for a in emitted if a not in allowed_set})
    covered_n = len(distinct_covered_norm)
    coverage_pct = (100.0 * covered_n / n_allowed) if n_allowed else 0.0
    display = f"{covered_n}/{n_allowed}"
    matched_labels = [a for a in allowed_raw if norm(a) in distinct_covered_norm]
    missing_labels = [a for a in allowed_raw if norm(a) not in distinct_covered_norm]
    # Total action slots listed (can exceed distinct if repeated across fields)
    actions_in_plan_count = len(from_all) + len(from_pri) + len(from_sup)
    unique_actions_any = len(set(emitted))
    action_scenario_pct = (
        (100.0 * unique_actions_any / actions_in_plan_count) if actions_in_plan_count else None
    )

    multi_layer, agentic_layers, multi_layer_audits = compute_agentic_multi_layer(
        plan, agents_block
    )
    align = summarize_ps_agentic_alignment(multi_layer_audits)
    raw_tiers: list[str] = []
    for key in ("primary_actions", "supporting_actions"):
        for row in plan.get(key) or []:
            if isinstance(row, dict) and row.get("network_tier"):
                raw_tiers.append(str(row["network_tier"]).strip())

    return {
        "mitigation_coverage_pct": round(coverage_pct, 2),
        "mitigation_coverage_display": display,
        "allowed_action_count": n_allowed,
        "distinct_allowed_covered": covered_n,
        "actions_in_plan_count": actions_in_plan_count,
        "unique_valid_action_types": covered_n,
        "invalid_actions": invalid,
        "all_known_mitigations_mentioned": len(invalid) == 0,
        "multi_layer": multi_layer,
        "agentic_layers_involved": agentic_layers,
        "multi_layer_row_audits": multi_layer_audits,
        "agentic_ps_row_count": align["agentic_ps_row_count"],
        "agentic_ps_rows_matched": align["agentic_ps_rows_matched"],
        "agentic_tier_action_alignment_pct": align["agentic_tier_action_alignment_pct"],
        "agentic_tier_action_alignment_display": align["agentic_tier_action_alignment_display"],
        "action_scenario_unique": unique_actions_any,
        "action_scenario_total": actions_in_plan_count,
        "action_scenario_pct": (round(action_scenario_pct, 2) if action_scenario_pct is not None else None),
        "plan_network_tier_raw": sorted(set(raw_tiers)),
        "matched_actions": matched_labels,
        "missing_allowed_actions": missing_labels,
        "all_actions": from_all,
        "primary_actions_list": from_pri,
        "supporting_actions_list": from_sup,
    }


def _fmt_pct(p: float) -> str:
    if abs(p - round(p)) < 1e-6:
        return f"{int(round(p))}%"
    t = f"{p:.2f}".rstrip("0").rstrip(".")
    return f"{t}%"


def _fmt_agentic_alignment(m: dict) -> str:
    pct = m.get("agentic_tier_action_alignment_pct")
    disp = m.get("agentic_tier_action_alignment_display", "0/0")
    if pct is None:
        return f"n/a ({disp})"
    return f"{_fmt_pct(pct)} ({disp})"


def _fmt_action_scenario(m: dict) -> str:
    u = m.get("action_scenario_unique", 0)
    t = m.get("action_scenario_total", 0)
    p = m.get("action_scenario_pct")
    if p is None:
        return f"n/a ({u}/{t})"
    return f"{_fmt_pct(p)} ({u}/{t})"


def _multi_layer_text(m: dict) -> str:
    parts = m.get("agentic_layers_involved") or []
    if m["multi_layer"]:
        return "Yes (" + " + ".join(parts) + ")"
    if len(parts) == 1:
        return f"No ({parts[0]} only)"
    if not parts:
        return "No (no tier/action agentic match)"
    return "No (" + ", ".join(parts) + ")"


def build_per_sample_markdown(documents: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Action metrics vs `attack_options.json` (per sample)\n")
    lines.append(
        "Mitigation coverage = distinct allowed actions found in **`all_actions` ∪ "
        "primary_actions[].action ∪ supporting_actions[].action** "
        "/ count of allowed actions for `predicted_label`. "
        "Avg. actions = total slots: len(all_actions) + #primary + #supporting. "
        "**Multi-layer** = ≥2 agents where a **primary/supporting** row has "
        "`network_tier` → RAN|Edge|Core and that agent's `action_capabilities` "
        "(in `agentic_features.json`) contains the **same** action. "
        "`all_actions` only gets an audit row (which agents list the action); "
        "it does not add layers (no tier on those entries). "
        "**Agentic tier–action alignment** = %% of primary/supporting rows whose "
        "action is listed under that row's agent in `action_capabilities` "
        "(100%% = all such rows match).\n"
    )
    for doc in documents:
        sid = doc["sample_id"]
        label = doc["predicted_label"]
        fn = doc["file_name"]
        lines.append(f"## Sample {sid} — **{label}** — `{fn}`\n")
        lines.append(
            "| Metric | Dense MMR | BM25 | docT5 | Dense -> CE "
            "| BM25 -> CE | Dense -> ColBERT |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        modes = doc["modes"]
        row_cov = "| Mitigation coverage |"
        row_matched = "| Matched actions |"
        row_known = "| All known mitigations |"
        row_scn = "| Action scenario (unique/total) |"
        row_align = "| Agentic tier–action alignment |"
        row_ml = "| Multi-layer strategies |"
        for key in MODES:
            m = modes[key]
            cov = f"{_fmt_pct(m['mitigation_coverage_pct'])} ({m['mitigation_coverage_display']})"
            row_cov += f" {cov} |"
            row_matched += f" {', '.join(m.get('matched_actions') or []) or '—'} |"
            row_known += f" {'YES' if m.get('all_known_mitigations_mentioned') else 'NO'} |"
            row_scn += f" {_fmt_action_scenario(m)} |"
            row_align += f" {_fmt_agentic_alignment(m)} |"
            row_ml += f" {_multi_layer_text(m)} |"
        lines.append(row_cov)
        lines.append(row_matched)
        lines.append(row_known)
        lines.append(row_scn)
        lines.append(row_align)
        lines.append(row_ml)
        lines.append("")
    return "\n".join(lines)


def build_explanations_markdown(documents: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Explanations (per sample, per mode)\n")
    lines.append(
        "- **Mitigation coverage**: which allowed actions (from `attack_options.json`) were hit/missed.\n"
        "- **All known mitigations**: YES iff all LLM action strings are a subset of the allowed list for that attack.\n"
        "- **Action scenario (unique/total)**: diversity of action strings (unique distinct / total slots).\n"
        "- **Agentic tier–action alignment**: for each primary/supporting row, whether the row's "
        "`network_tier` agent lists that `action` in `agentic_features.json` `action_capabilities`.\n"
        "- **Multi-layer**: requires ≥2 distinct agents with at least one primary/supporting row where "
        "`agentic_match == true`.\n"
    )
    mode_title = {
        "none": "Dense MMR",
        "bm25": "BM25",
        "doct5query": "docT5",
        "crossencoder": "Dense -> CE",
        "bm25_crossencoder": "BM25 -> CE",
        "colbert": "Dense -> ColBERT",
    }

    def fmt_match(audit: dict) -> str:
        m = audit.get("agentic_match")
        if m is True:
            return "ALLOWED"
        if m is False:
            return "NOT ALLOWED"
        return "n/a"

    for doc in documents:
        sid = doc["sample_id"]
        label = doc["predicted_label"]
        fn = doc["file_name"]
        lines.append(f"## Sample {sid} — **{label}** — `{fn}`\n")
        for mode_key in MODES:
            m = doc["modes"][mode_key]
            lines.append(f"### {mode_title.get(mode_key, mode_key)}\n")
            lines.append(
                f"- **Mitigation coverage**: {_fmt_pct(m['mitigation_coverage_pct'])} "
                f"({m['mitigation_coverage_display']})"
            )
            lines.append(f"  - **matched**: {', '.join(m.get('matched_actions') or []) or '—'}")
            lines.append(
                f"  - **missing**: {', '.join(m.get('missing_allowed_actions') or []) or '—'}"
            )
            lines.append(
                f"- **All known mitigations**: {'YES' if m.get('all_known_mitigations_mentioned') else 'NO'}"
            )
            lines.append(f"- **Action scenario (unique/total)**: {_fmt_action_scenario(m)}")
            lines.append(
                f"- **Agentic tier–action alignment**: {_fmt_agentic_alignment(m)}"
            )

            audits = m.get("multi_layer_row_audits") or []
            ps = [
                a
                for a in audits
                if a.get("source") in ("primary_actions", "supporting_actions")
                and a.get("action")
            ]
            if ps:
                lines.append("  - **row checks (primary/supporting)**:")
                for a in ps:
                    act = a.get("action") or ""
                    tier_raw = a.get("network_tier_raw")
                    tier_norm = a.get("network_tier_normalized")
                    holders = a.get("action_capabilities_agents") or []
                    lines.append(
                        f"    - `{act}` @ `{tier_raw}` (norm `{tier_norm}`) → "
                        f"**{fmt_match(a)}**; capability_agents={holders}"
                    )
            else:
                lines.append("  - **row checks (primary/supporting)**: —")

            layers = m.get("agentic_layers_involved") or []
            lines.append(
                f"- **Multi-layer strategies**: "
                f"{'Yes' if m.get('multi_layer') else 'No'} "
                f"({(' + '.join(layers)) if layers else 'no matched agentic tiers'})"
            )
            lines.append("")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    attacks = json.loads(ATTACK_OPTIONS.read_text(encoding="utf-8"))["attacks"]
    agentic_raw = json.loads(AGENTIC_FEATURES.read_text(encoding="utf-8"))
    agents_block = agentic_raw.get("agents") or {}
    documents: list[dict] = []
    by_mode: dict[str, list[dict]] = {m: [] for m in MODES}

    for path in sorted(SAMPLE_DIR.glob("rerank_comparison_sample_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = data.get("prediction", {}).get("predicted_label")
        if not label or label not in attacks:
            continue
        allowed = attacks[label]
        modes_out: dict[str, dict] = {}
        for m in MODES:
            block = (data.get("modes") or {}).get(m) or {}
            plan = block.get("llm_action_plan") or {}
            met = analyze_mode(plan, allowed, agents_block)
            met["rerank_mode"] = m
            met["rerank_label"] = block.get("label")
            modes_out[m] = met
            by_mode[m].append(met)

        documents.append(
            {
                "file_name": path.name,
                "sample_id": data.get("sample_id"),
                "predicted_label": label,
                "allowed_actions": allowed,
                "modes": modes_out,
            }
        )

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    aggregate = {}
    for m in MODES:
        rows = by_mode[m]
        aggregate[m] = {
            "n_documents": len(rows),
            "mean_mitigation_coverage_pct": round(
                mean(r["mitigation_coverage_pct"] for r in rows), 2
            ),
            "mean_actions_in_plan_count": round(
                mean(float(r["actions_in_plan_count"]) for r in rows), 2
            ),
            "mean_unique_valid_action_types": round(
                mean(float(r["unique_valid_action_types"]) for r in rows), 2
            ),
            "multi_layer_pct": round(
                100.0 * mean(1.0 if r["multi_layer"] else 0.0 for r in rows), 1
            ),
            "mean_agentic_tier_action_alignment_pct": (
                round(mean(align_vals), 2)
                if (
                    align_vals := [
                        r["agentic_tier_action_alignment_pct"]
                        for r in rows
                        if r.get("agentic_tier_action_alignment_pct") is not None
                    ]
                )
                else None
            ),
        }

    out = {
        "schema_version": "1.0",
        "attack_options_path": str(ATTACK_OPTIONS.as_posix()),
        "agentic_features_path": str(AGENTIC_FEATURES.as_posix()),
        "sample_dir": str(SAMPLE_DIR.as_posix()),
        "metric_definitions": {
            "mitigation_coverage_pct": "100 * (distinct allowed actions appearing in all_actions OR primary_actions[].action OR supporting_actions[].action) / len(allowed_actions for predicted_label)",
            "actions_in_plan_count": "len(all_actions) + count(primary with action) + count(supporting with action)",
            "unique_valid_action_types": "same as distinct_allowed_covered (distinct allowed hits from the union of the three sources)",
            "all_known_mitigations_mentioned": "True iff every action string emitted by the LLM (all_actions + primary + supporting) is in the allowed list for that attack label (i.e., invalid_actions is empty)",
            "action_scenario_pct": "100 * (#unique distinct action strings emitted) / (total action slots emitted); null if total is 0",
            "multi_layer": "True if >=2 distinct agents have at least one primary/supporting row where normalized network_tier equals that agent and the action is in that agent's action_capabilities per agentic_features.json",
            "agentic_layers_involved": "Agents with at least one strict tier+capability match on primary/supporting rows",
            "multi_layer_row_audits": "Per action: LLM tier vs agentic action_capabilities; agentic_match true only when tier agent lists that action",
            "agentic_tier_action_alignment_pct": "100 * (primary/supporting rows with agentic_match) / (primary/supporting rows with non-empty action); null if no such rows",
        },
        "documents": documents,
        "aggregate_by_mode": aggregate,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dest = REPORTS_DIR / "action_metrics_vs_attack_options.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {dest} ({len(documents)} documents)")

    md_path = REPORTS_DIR / "action_metrics_per_sample_tables.md"
    md_path.write_text(build_per_sample_markdown(documents), encoding="utf-8")
    print(f"Wrote {md_path}")

    exp_path = REPORTS_DIR / "action_metrics_explanations.md"
    exp_path.write_text(build_explanations_markdown(documents), encoding="utf-8")
    print(f"Wrote {exp_path}")


if __name__ == "__main__":
    main()
