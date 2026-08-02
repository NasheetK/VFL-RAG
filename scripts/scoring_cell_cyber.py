# Template for Score.ipynb scoring loop (imported by patch_score_pipeline.py)
SCORING_CELL = r'''# Score rerank modes with retrieval + reasoning metrics (MiniLM + cyber BERT encoders).
#
# Two-pass pipeline: Pass 1 grades chunks and collects context/utilization scores;
# Pass 2 emits Recall@10 / nDCG@10 per encoder family and CRAG / BERTScore.

from utils.cyber_encoders import (
    CYBER_ENCODER_KEYS,
    GRADING_ENCODER_KEYS,
    grades_for_texts,
    pairwise_cosine,
)
from utils.rag_retrieval_scores import recall_at_k, ndcg_at_k

RERANK_MODES = [
    "none",
    "bm25",
    "doct5query",
    "crossencoder",
    "bm25_crossencoder",
    "colbert",
    "none_securebert",
    "crossencoder_securebert",
]

pass1_rows: List[Dict[str, Any]] = []
all_pool_grades: List[float] = []
all_context_sberts: List[float] = []
all_context_rouges: List[float] = []

for comp_data in comparison_data:
    file_name = comp_data["file_name"]
    sample_id = comp_data.get("sample_id", "unknown")
    predicted_label = comp_data.get("prediction", {}).get("predicted_label", "OTHERS")
    reference_text = build_reference_text_for_label(predicted_label)

    modes = comp_data.get("modes", {}) or {}
    if not isinstance(modes, dict):
        continue

    for mode_key in RERANK_MODES:
        mode_block = modes.get(mode_key, {}) if isinstance(modes, dict) else {}
        if not isinstance(mode_block, dict):
            continue

        plan = mode_block.get("llm_action_plan", {})
        overall_reasoning = extract_overall_reasoning(plan)

        top_results = _get_mode_top_results(comp_data, mode_key)
        ranked_chunk_texts = [_chunk_string(r) for r in top_results]
        retrieved_context = "\n\n".join(ranked_chunk_texts[:TOP_K]).strip()

        candidate_pool = _get_mode_candidate_pool(comp_data, mode_key)
        pool_chunk_texts = [_chunk_string(r) for r in candidate_pool]

        encoder_grades: Dict[str, Dict[str, List[float]]] = {}
        for enc in GRADING_ENCODER_KEYS:
            rg, pg = grades_for_texts(
                reference_text, ranked_chunk_texts, pool_chunk_texts, enc
            )
            encoder_grades[enc] = {"ranked": rg, "pool": pg}
        if encoder_grades:
            all_pool_grades.extend(encoder_grades["minilm"]["pool"])

        context_sbert = (
            calculate_sbert_cosine(reference_text, retrieved_context)
            if retrieved_context else 0.0
        )
        context_rouge1 = (
            calculate_rouge1_score(reference_text, retrieved_context)
            if retrieved_context else 0.0
        )
        context_cyber = {
            enc: (
                pairwise_cosine(reference_text, retrieved_context, enc)
                if retrieved_context
                else 0.0
            )
            for enc in CYBER_ENCODER_KEYS
        }

        utilization_sbert = (
            calculate_sbert_cosine(retrieved_context, overall_reasoning)
            if retrieved_context and overall_reasoning else 0.0
        )
        utilization_rouge1 = (
            calculate_rouge1_score(retrieved_context, overall_reasoning)
            if retrieved_context and overall_reasoning else 0.0
        )
        utilization_cyber = {
            enc: (
                pairwise_cosine(retrieved_context, overall_reasoning, enc)
                if retrieved_context and overall_reasoning
                else 0.0
            )
            for enc in CYBER_ENCODER_KEYS
        }

        if retrieved_context:
            all_context_sberts.append(context_sbert)
            all_context_rouges.append(context_rouge1)

        pass1_rows.append({
            "file_name": file_name,
            "sample_id": sample_id,
            "predicted_label": predicted_label,
            "method": mode_key,
            "reference_text": reference_text,
            "overall_reasoning": overall_reasoning,
            "retrieved_context": retrieved_context,
            "encoder_grades": encoder_grades,
            "pool_size": len(encoder_grades.get("minilm", {}).get("pool", [])),
            "context_sbert": context_sbert,
            "context_rouge1": context_rouge1,
            "context_cyber": context_cyber,
            "utilization_sbert": utilization_sbert,
            "utilization_rouge1": utilization_rouge1,
            "utilization_cyber": utilization_cyber,
        })

adaptive = derive_adaptive_thresholds(
    per_chunk_grades=all_pool_grades,
    context_sbert_values=all_context_sberts,
    context_rouge_values=all_context_rouges,
)

print("Adaptive thresholds (batch-derived, clamped to floors):")
for k, v in adaptive.items():
    print(f"  {k:>22s} = {v:.4f}")

scoring_results: List[Dict[str, Any]] = []
current_file = None

for row in pass1_rows:
    file_name = row["file_name"]
    sample_id = row["sample_id"]
    predicted_label = row["predicted_label"]
    mode_key = row["method"]
    reference_text = row["reference_text"]
    overall_reasoning = row["overall_reasoning"]
    retrieved_context = row["retrieved_context"]
    encoder_grades = row["encoder_grades"]
    context_sbert = row["context_sbert"]
    context_rouge1 = row["context_rouge1"]
    context_cyber = row["context_cyber"]
    utilization_sbert = row["utilization_sbert"]
    utilization_rouge1 = row["utilization_rouge1"]
    utilization_cyber = row["utilization_cyber"]

    if file_name != current_file:
        current_file = file_name
        print(f"\n{'='*60}")
        print(f"File: {file_name} (Sample ID: {sample_id}, Label: {predicted_label})")
        print(f"{'='*60}")

    recall_ndcg: Dict[str, float] = {}
    for enc in GRADING_ENCODER_KEYS:
        eg = encoder_grades.get(enc, {})
        ranked_g = eg.get("ranked", [])
        pool_g = eg.get("pool", [])
        suffix = "" if enc == "minilm" else f"_{enc}"
        recall_ndcg[f"recall_at_10{suffix}"] = recall_at_k(
            ranked_grades=ranked_g,
            pool_grades=pool_g,
            k=TOP_K,
            relevance_threshold=adaptive["relevance_threshold"],
        )
        recall_ndcg[f"ndcg_at_10{suffix}"] = ndcg_at_k(ranked_grades=ranked_g, k=TOP_K)

    reasoning_sbert = (
        calculate_sbert_cosine(reference_text, overall_reasoning)
        if overall_reasoning else 0.0
    )
    reasoning_rouge1 = (
        calculate_rouge1_score(reference_text, overall_reasoning)
        if overall_reasoning else 0.0
    )
    reasoning_cyber = {
        enc: (
            pairwise_cosine(reference_text, overall_reasoning, enc)
            if overall_reasoning
            else 0.0
        )
        for enc in CYBER_ENCODER_KEYS
    }

    bertscore_reasoning = (
        calculate_bertscore_scores([reference_text], [overall_reasoning])
        if overall_reasoning
        else {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    )

    crag_context_proxy = calculate_crag_style_truthfulness(
        context_sbert=context_sbert,
        context_rouge1=context_rouge1,
        has_answer=bool(retrieved_context),
        sbert_good=adaptive["crag_sbert_good"],
        rouge_good=adaptive["crag_rouge_good"],
        sbert_weak=adaptive["crag_sbert_weak"],
        rouge_weak=adaptive["crag_rouge_weak"],
    )
    crag_reasoning_proxy = calculate_crag_style_truthfulness(
        context_sbert=reasoning_sbert,
        context_rouge1=reasoning_rouge1,
        has_answer=bool(overall_reasoning),
        sbert_good=adaptive["crag_sbert_good"],
        rouge_good=adaptive["crag_rouge_good"],
        sbert_weak=adaptive["crag_sbert_weak"],
        rouge_weak=adaptive["crag_rouge_weak"],
    )
    crag_utilization_proxy = calculate_crag_style_truthfulness(
        context_sbert=utilization_sbert,
        context_rouge1=utilization_rouge1,
        has_answer=bool(retrieved_context and overall_reasoning),
        sbert_good=adaptive["crag_sbert_good"],
        rouge_good=adaptive["crag_rouge_good"],
        sbert_weak=adaptive["crag_sbert_weak"],
        rouge_weak=adaptive["crag_rouge_weak"],
    )

    crag_context_cyber = {}
    crag_util_cyber = {}
    crag_reason_cyber = {}
    for enc in CYBER_ENCODER_KEYS:
        cs = context_cyber[enc]
        us = utilization_cyber[enc]
        rs = reasoning_cyber[enc]
        crag_context_cyber[enc] = calculate_crag_style_truthfulness(
            context_sbert=cs,
            context_rouge1=context_rouge1,
            has_answer=bool(retrieved_context),
            sbert_good=adaptive["crag_sbert_good"],
            rouge_good=adaptive["crag_rouge_good"],
            sbert_weak=adaptive["crag_sbert_weak"],
            rouge_weak=adaptive["crag_rouge_weak"],
        )
        crag_util_cyber[enc] = calculate_crag_style_truthfulness(
            context_sbert=us,
            context_rouge1=utilization_rouge1,
            has_answer=bool(retrieved_context and overall_reasoning),
            sbert_good=adaptive["crag_sbert_good"],
            rouge_good=adaptive["crag_rouge_good"],
            sbert_weak=adaptive["crag_sbert_weak"],
            rouge_weak=adaptive["crag_rouge_weak"],
        )
        crag_reason_cyber[enc] = calculate_crag_style_truthfulness(
            context_sbert=rs,
            context_rouge1=reasoning_rouge1,
            has_answer=bool(overall_reasoning),
            sbert_good=adaptive["crag_sbert_good"],
            rouge_good=adaptive["crag_rouge_good"],
            sbert_weak=adaptive["crag_sbert_weak"],
            rouge_weak=adaptive["crag_rouge_weak"],
        )

    print(
        f"Mode={mode_key:22s} | Recall@10={recall_ndcg['recall_at_10']:.4f} | "
        f"nDCG@10={recall_ndcg['ndcg_at_10']:.4f} | "
        f"Ctx-SBERT={context_sbert:.4f} | Reason-SBERT={reasoning_sbert:.4f} | "
        f"BERT-F1={bertscore_reasoning['bertscore_f1']:.4f}"
    )

    result_row: Dict[str, Any] = {
        "file_name": file_name,
        "sample_id": sample_id,
        "predicted_label": predicted_label,
        "method": mode_key,
        "text": overall_reasoning,
        "reference_text": reference_text,
        "pool_size": row["pool_size"],
        "recall_at_10": recall_ndcg["recall_at_10"],
        "ndcg_at_10": recall_ndcg["ndcg_at_10"],
        "context_sbert_cosine": context_sbert,
        "context_rouge1": context_rouge1,
        "context_crag_proxy": crag_context_proxy,
        "utilization_sbert_cosine": utilization_sbert,
        "utilization_rouge1": utilization_rouge1,
        "utilization_crag_proxy": crag_utilization_proxy,
        "reasoning_sbert_cosine": reasoning_sbert,
        "reasoning_rouge1": reasoning_rouge1,
        "reasoning_crag_proxy": crag_reasoning_proxy,
        "bertscore_precision": bertscore_reasoning["bertscore_precision"],
        "bertscore_recall": bertscore_reasoning["bertscore_recall"],
        "bertscore_f1": bertscore_reasoning["bertscore_f1"],
        "relevance_threshold_used": adaptive["relevance_threshold"],
    }
    for enc in CYBER_ENCODER_KEYS:
        result_row[f"recall_at_10_{enc}"] = recall_ndcg[f"recall_at_10_{enc}"]
        result_row[f"ndcg_at_10_{enc}"] = recall_ndcg[f"ndcg_at_10_{enc}"]
        result_row[f"context_{enc}_cosine"] = context_cyber[enc]
        result_row[f"context_{enc}_crag_proxy"] = crag_context_cyber[enc]
        result_row[f"utilization_{enc}_cosine"] = utilization_cyber[enc]
        result_row[f"utilization_{enc}_crag_proxy"] = crag_util_cyber[enc]
        result_row[f"reasoning_{enc}_cosine"] = reasoning_cyber[enc]
        result_row[f"reasoning_{enc}_crag_proxy"] = crag_reason_cyber[enc]

    scoring_results.append(result_row)

print(f"\n\nTotal scoring rows (files x modes): {len(scoring_results)}")
'''
