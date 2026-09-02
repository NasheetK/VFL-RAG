#!/usr/bin/env python3
"""Add CySecBERT / CTI-BERT / SecureBERT metrics to Score.ipynb downstream cells."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Score.ipynb"

CELL_DF = r'''# Create DataFrame for easier analysis
from utils.score_display import (
    HEADLINE_METRIC_COLS,
    cyber_metric_columns,
    metric_display_rename,
    present_metric_columns,
)

df_results = pd.DataFrame(scoring_results)
METRIC_DISPLAY_RENAME = metric_display_rename()
HEADLINE_COLS = [c for c in HEADLINE_METRIC_COLS if c in df_results.columns]
CYBER_COLS = [c for c in cyber_metric_columns() if c in df_results.columns]
ALL_METRIC_COLS = present_metric_columns(list(df_results.columns))


def _short_file_name(name: str) -> str:
    prefix = "rerank_comparison_sample_"
    return name[len(prefix):] if isinstance(name, str) and name.startswith(prefix) else name


def _display_metrics(subset_cols: list[str], title: str) -> None:
    if not subset_cols or df_results.empty:
        return
    view = df_results[["file_name", "sample_id", "method", "predicted_label"] + subset_cols].copy()
    view["file_name"] = view["file_name"].map(_short_file_name)
    view = view.rename(columns=METRIC_DISPLAY_RENAME)
    print(title)
    print(view.to_string(index=False))


if not df_results.empty:
    print("Scoring Results — MiniLM headline metrics:")
    _display_metrics(HEADLINE_COLS, "")
    if CYBER_COLS:
        print("\nScoring Results — cyber BERT encoders (CySecBERT / CTI-BERT / SecureBERT):")
        _display_metrics(CYBER_COLS, "")
else:
    print("No results to display!")
'''

CELL_AVG = r'''# Calculate average scores by rerank mode
from utils.score_display import metric_display_rename, present_metric_columns

if not df_results.empty:
    metric_cols = present_metric_columns(list(df_results.columns))
    rename = metric_display_rename()

    avg_scores = (
        df_results.groupby("method")[metric_cols]
        .mean()
        .round(4)
        .rename(columns=rename)
    )

    print("\nAverage Scores by Rerank Mode (MiniLM + cyber BERT):")
    print("=" * 100)
    print(avg_scores.to_string())

    overall_avg = df_results[metric_cols].mean().round(4).rename(rename)

    print("\n\nOverall Average Scores (across all modes):")
    print("=" * 100)
    for metric, value in overall_avg.items():
        print(f"{metric:18s}: {value:.4f}")
else:
    print("No results to calculate averages!")
'''

CELL_ATTACK = r'''# Average scores grouped by attack class (predicted_label)
from utils.score_display import method_order_for_df, metric_display_rename, present_metric_columns

if not df_results.empty:
    metric_cols = present_metric_columns(list(df_results.columns))
    rename = metric_display_rename()

    observed_labels = set(df_results["predicted_label"].unique())
    preferred_order = [lbl for lbl in TARGET_CLASSES if lbl in observed_labels]
    tail = sorted(observed_labels - set(preferred_order))
    attack_order = preferred_order + tail

    avg_scores_by_attack = (
        df_results.groupby("predicted_label")[metric_cols]
        .mean()
        .round(4)
        .reindex(attack_order)
        .rename(columns=rename)
    )
    n_by_attack = (
        df_results.groupby("predicted_label").size().reindex(attack_order).astype(int)
    )
    avg_scores_by_attack.insert(0, "n", n_by_attack.values)
    avg_scores_by_attack.index.name = "attack_class"

    print("\nAverage Scores by Attack Class (MiniLM + cyber BERT):")
    print("=" * 100)
    print(avg_scores_by_attack.to_string())

    method_order = method_order_for_df(list(df_results["method"].unique()))

    for attack_label in attack_order:
        attack_df = df_results[df_results["predicted_label"] == attack_label]
        if attack_df.empty:
            continue

        per_method_avg = (
            attack_df.groupby("method")[metric_cols]
            .mean()
            .round(4)
            .reindex([m for m in method_order if m in attack_df["method"].unique()])
            .rename(columns=rename)
        )
        n_samples = int(attack_df["sample_id"].nunique())

        print(
            f"\n\nAverage Scores by Rerank Mode — {attack_label} "
            f"(n_samples={n_samples}, n_rows={len(attack_df)}):"
        )
        print("=" * 100)
        print(per_method_avg.to_string())
else:
    print("No results to calculate attack-class averages!")
'''

CELL_BARS = r'''# Bar charts: headline MiniLM metrics + cyber BERT reasoning cosine
from utils.cyber_encoders import CYBER_ENCODER_KEYS
from utils.score_display import (
    ENCODER_SHORT,
    HEADLINE_METRIC_COLS,
    METHOD_DISPLAY,
    method_order_for_df,
    metric_display_rename,
)

if not df_results.empty:
    rename = metric_display_rename()
    methods = method_order_for_df(list(df_results["method"].unique()))
    method_labels = [METHOD_DISPLAY.get(m, m) for m in methods]
    n_methods = max(len(methods), 1)
    bar_width = 0.8 / n_methods
    x_base = np.arange(len(methods))

    headline_plot = [
        c for c in HEADLINE_METRIC_COLS
        if c in df_results.columns and c != "bertscore_f1"
    ]
    if headline_plot:
        fig, axes = plt.subplots(
            len(headline_plot), 1,
            figsize=(12, 2.8 * len(headline_plot)),
            squeeze=False,
        )
        for ax_row, col in zip(axes.flat, headline_plot):
            means = [
                float(df_results.loc[df_results["method"] == m, col].mean())
                for m in methods
            ]
            ax_row.bar(x_base, means, width=0.7, color=sns.color_palette("tab10", n_methods))
            ax_row.set_xticks(x_base)
            ax_row.set_xticklabels(method_labels, rotation=35, ha="right")
            ax_row.set_title(rename.get(col, col))
            ax_row.set_ylim(0, 1)
        plt.tight_layout()
        plt.show()

    cyber_reason_cols = [
        f"reasoning_{enc}_cosine"
        for enc in CYBER_ENCODER_KEYS
        if f"reasoning_{enc}_cosine" in df_results.columns
    ]
    if cyber_reason_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        group_x = np.arange(len(methods))
        width = 0.8 / max(len(cyber_reason_cols), 1)
        for i, col in enumerate(cyber_reason_cols):
            enc = col.replace("reasoning_", "").replace("_cosine", "")
            means = [
                float(df_results.loc[df_results["method"] == m, col].mean())
                for m in methods
            ]
            offset = (i - (len(cyber_reason_cols) - 1) / 2) * width
            ax.bar(group_x + offset, means, width=width, label=ENCODER_SHORT.get(enc, enc))
        ax.set_xticks(group_x)
        ax.set_xticklabels(method_labels, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("LLM reasoning cosine — cyber BERT encoders")
        ax.legend()
        plt.tight_layout()
        plt.show()

    cyber_recall_cols = [
        f"recall_at_10_{enc}"
        for enc in CYBER_ENCODER_KEYS
        if f"recall_at_10_{enc}" in df_results.columns
    ]
    if cyber_recall_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, col in enumerate(cyber_recall_cols):
            enc = col.replace("recall_at_10_", "")
            means = [
                float(df_results.loc[df_results["method"] == m, col].mean())
                for m in methods
            ]
            offset = (i - (len(cyber_recall_cols) - 1) / 2) * width
            ax.bar(group_x + offset, means, width=width, label=ENCODER_SHORT.get(enc, enc))
        ax.set_xticks(group_x)
        ax.set_xticklabels(method_labels, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Recall@10 — cyber BERT encoders")
        ax.legend()
        plt.tight_layout()
        plt.show()
else:
    print("No results to plot!")
'''

CELL_INDIV = r'''# Display individual file scores
from utils.score_display import (
    HEADLINE_METRIC_COLS,
    cyber_metric_columns,
    metric_display_rename,
)

if not df_results.empty:
    rename = metric_display_rename()
    headline_cols = [c for c in HEADLINE_METRIC_COLS if c in df_results.columns]
    cyber_cols = [c for c in cyber_metric_columns() if c in df_results.columns]
    print("\nIndividual File Scores:")
    print("=" * 100)

    for file_name in df_results["file_name"].unique():
        file_data = df_results[df_results["file_name"] == file_name]
        print(f"\nFile: {file_name}")
        print("-" * 100)

        for _, row in file_data.iterrows():
            print(f"\nMode: {str(row['method']).upper()}")
            for col in headline_cols + cyber_cols:
                label = rename.get(col, col)
                print(f"  {label:18s}: {float(row[col]):.4f}")
            reasoning_text = str(row.get("text", "") or "")
            print(
                f"  Agent guidance: {reasoning_text[:120]}..."
                if len(reasoning_text) > 120
                else f"  Agent guidance: {reasoning_text}"
            )
else:
    print("No results to display!")
'''

CELL_JSON = r'''# Save detailed results to JSON
if scoring_results:
    from utils.cyber_encoders import scoring_metric_column_names
    from utils.pipeline_paths import save_scores_by_class, SCORES_ROOT
    from utils.score_display import metric_display_rename, present_metric_columns

    output_data = {
        "metric_notes": {
            "retrieval": (
                "Recall@10 / nDCG@10 over candidate_pool using graded relevance "
                "(0.5*ROUGE-1 + 0.5*encoder cosine). MiniLM (SBERT) plus CySecBERT, "
                "CTI-BERT, and SecureBERT bi-encoder variants (_cysecbert, _ctibert, "
                "_securebert suffixes)."
            ),
            "ndcg": "nDCG@10 uses the same per-encoder graded relevance over top-10.",
            "context": (
                "SBERT / cyber-encoder cosine and CRAG-style proxy between retrieved "
                "top-10 context and reference text."
            ),
            "utilization": (
                "SBERT / cyber-encoder cosine and CRAG proxy between retrieved context "
                "and agent_guidance (faithfulness)."
            ),
            "reasoning": (
                "SBERT / cyber-encoder cosine + CRAG proxy between agent_guidance and "
                "reference text."
            ),
            "bertscore": "BERTScore-F1 (generative) between reference and agent_guidance.",
            "cyber_encoders": (
                "CySecBERT, CTI-BERT, SecureBERT bi-encoder metrics at retrieval, "
                "context/utilization, and reasoning layers."
            ),
        },
        "adaptive_thresholds_used": adaptive,
        "top_k": TOP_K,
        "total_files": len(comparison_data),
        "timestamp": pd.Timestamp.now().isoformat(),
        "individual_scores": scoring_results,
        "average_scores": {},
    }

    if not df_results.empty:
        metric_cols = present_metric_columns(list(df_results.columns))
        rename = metric_display_rename()
        for method in sorted(df_results["method"].unique()):
            method_data = df_results[df_results["method"] == method]
            output_data["average_scores"][method] = {
                rename.get(m, m): float(method_data[m].mean())
                for m in metric_cols
                if m in method_data.columns
            }

    _class_paths = save_scores_by_class(
        scoring_results,
        adaptive_thresholds=adaptive,
        metric_notes=output_data.get("metric_notes"),
    )
    output_file = SCORES_ROOT / "scoring_results_combined.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Combined summary: {output_file}")
    print("Per-class score files:")
    for _cls, _p in sorted(_class_paths.items()):
        print(f"  {_cls}: {_p}")
else:
    print("No results to save!")
'''

CELL_CSV = r'''# Average metrics CSV (MiniLM headline + cyber BERT columns)
from utils.cyber_encoders import CYBER_ENCODER_KEYS
from utils.score_display import (
    ENCODER_SHORT,
    HEADLINE_METRIC_COLS,
    METHOD_DISPLAY,
    RERANK_METHOD_ORDER,
    metric_display_rename,
    present_metric_columns,
)

if not df_results.empty:
    rename = metric_display_rename()
    row_order = [m for m in RERANK_METHOD_ORDER if m in df_results["method"].unique()]
    row_order += sorted(set(df_results["method"].unique()) - set(row_order))
    metric_cols = present_metric_columns(list(df_results.columns))

    _agg = df_results.groupby("method")[metric_cols].mean().reset_index()

    rows_out = []
    for key in row_order:
        sub = _agg[_agg["method"] == key]
        if sub.empty:
            continue
        r = sub.iloc[0]
        row_out = {"Method": METHOD_DISPLAY.get(key, key)}
        for col in metric_cols:
            row_out[rename.get(col, col)] = round(float(r[col]), 6)
        rows_out.append(row_out)

    summary_df = pd.DataFrame(rows_out)
    combined_csv = SCORES_ROOT / "rerank_method_averages_all.csv"
    summary_df.to_csv(combined_csv, index=False)
    print(f"\nCombined average-by-method CSV: {combined_csv}")

    cyber_csv = SCORES_ROOT / "rerank_method_averages_cyber_all.csv"
    cyber_cols = [
        c for c in summary_df.columns
        if c != "Method" and any(c.endswith(f"_{ENCODER_SHORT[e]}") for e in CYBER_ENCODER_KEYS)
    ]
    if cyber_cols:
        summary_df[["Method"] + cyber_cols].to_csv(cyber_csv, index=False)
        print(f"Cyber-encoder CSV: {cyber_csv}")

    print("(Per-class CSVs are under output/scores/<CLASS>/)")
    print(summary_df.to_string(index=False))
else:
    print("No results — skipping rerank_method_averages CSV")
'''


def _set_cell_source(cell: dict, text: str) -> None:
    cell["source"] = [line + "\n" for line in text.splitlines()]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    replacements = {
        "# Create DataFrame for easier analysis": CELL_DF,
        "# Calculate average scores by rerank mode": CELL_AVG,
        "# Average scores grouped by attack class (predicted_label)": CELL_ATTACK,
        "# Display individual file scores": CELL_INDIV,
        "# Save detailed results to JSON": CELL_JSON,
        "# Average metrics CSV (Dense MMR / BM25 / docT5 / Dense+CE / BM25+CE / Dense+ColBERT)": CELL_CSV,
    }
    changed = 0
    has_bars = False

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for marker, new_src in replacements.items():
            if src.strip().startswith(marker):
                _set_cell_source(cell, new_src)
                changed += 1
                break
        if "Bar charts: headline MiniLM" in src:
            has_bars = True

    if not has_bars:
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "markdown":
                src = "".join(cell.get("source", []))
                if "## Visualizations: Bar Charts" in src:
                    nb["cells"].insert(
                        i + 1,
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [],
                        },
                    )
                    _set_cell_source(nb["cells"][i + 1], CELL_BARS)
                    changed += 1
                    break

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Patched Score.ipynb: {changed} cell(s)")


if __name__ == "__main__":
    main()
