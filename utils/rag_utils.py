"""RAG pipeline helpers: predictions/config loaders, FAISS save/load, SHAP party/tier + JSON export.

Used by Index.ipynb and Plan.ipynb.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# LangChain still pulls pydantic.v1; that emits a noisy UserWarning on Python 3.14+.
if sys.version_info >= (3, 14):
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
        category=UserWarning,
    )

import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings

from utils.compute_device import sentence_transformer_model_kwargs
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# Predictions + RAG_docs JSON configs
# ---------------------------------------------------------------------------


def load_predictions(
    predictions_dir: Path,
    verbose: bool = True,
    *,
    recursive: bool = False,
) -> List[Dict[str, Any]]:
    """Load prediction JSON files; each sample gets _source_file.

    Set ``recursive=True`` to read ``output/inference/<CLASS>/*.json`` layout.
    """
    predictions_dir = Path(predictions_dir)
    if recursive:
        prediction_files = sorted(
            p
            for p in predictions_dir.rglob("*.json")
            if p.name != "manifest.json" and not p.name.startswith("manifest_")
        )
    else:
        prediction_files = sorted(predictions_dir.glob("*.json"))
    if not prediction_files:
        raise FileNotFoundError(f"No JSON prediction files found under {predictions_dir}")
    if verbose:
        print(f"Found {len(prediction_files)} prediction file(s) in {predictions_dir}")

    predictions_data: List[Dict[str, Any]] = []
    for prediction_file in prediction_files:
        if verbose:
            print(f"  Loading: {prediction_file.name}")
        with open(prediction_file, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        if isinstance(file_data, dict):
            file_data = [file_data]
        elif not isinstance(file_data, list):
            if verbose:
                print(f"    Skipping unexpected type in {prediction_file.name}")
            continue
        for sample in file_data:
            if isinstance(sample, dict):
                sample["_source_file"] = prediction_file.name
        predictions_data.extend(file_data)
        if verbose:
            print(f"    Loaded {len(file_data)} prediction(s) from {prediction_file.name}")
    if verbose:
        print(f"Total: {len(predictions_data)} prediction(s)")
    return predictions_data


def load_predictions_from_class_folders(
    inference_root: Path,
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Load prediction JSON from ``output/inference/<CLASS>/`` only (not root combined files).

    For each target class, uses the newest ``predictions_*.json`` in that class folder
    when multiple runs exist.
    """
    from utils.label_config import TARGET_CLASSES
    from utils.pipeline_paths import inference_dir_for

    inference_root = Path(inference_root)
    predictions_data: List[Dict[str, Any]] = []

    if verbose:
        print(f"Loading predictions from class folders under {inference_root}")

    for cls in TARGET_CLASSES:
        cls_dir = inference_dir_for(cls)
        if not cls_dir.is_dir():
            if verbose:
                print(f"  [{cls}] skip — missing folder {cls_dir}")
            continue

        json_files = sorted(
            p
            for p in cls_dir.glob("*.json")
            if not p.name.startswith("manifest")
        )
        if not json_files:
            if verbose:
                print(f"  [{cls}] skip — no prediction JSON in {cls_dir}")
            continue

        pred_file = max(json_files, key=lambda p: p.stat().st_mtime)
        if verbose:
            print(f"  [{cls}] {pred_file.name}")

        with open(pred_file, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        if isinstance(file_data, dict):
            file_data = [file_data]
        elif not isinstance(file_data, list):
            if verbose:
                print(f"    Skipping unexpected type in {pred_file.name}")
            continue

        for sample in file_data:
            if isinstance(sample, dict):
                sample["_source_file"] = pred_file.name
                sample["_source_class"] = cls
        predictions_data.extend(file_data)
        if verbose:
            print(f"    Loaded {len(file_data)} prediction(s)")

    if not predictions_data:
        raise FileNotFoundError(
            f"No prediction JSON found under {inference_root}/<CLASS>/ "
            f"for classes {list(TARGET_CLASSES)}. Run Predict.ipynb first."
        )

    if verbose:
        print(f"Total: {len(predictions_data)} prediction(s) from {len(TARGET_CLASSES)} class folders")
    return predictions_data


def load_attack_and_agentic(
    attack_path: Path = Path("RAG_docs/attack_options.json"),
    agentic_path: Path = Path("RAG_docs/agentic_features.json"),
    verbose: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load attack_options and agentic_features if files exist."""
    attack_actions_data: Optional[Dict[str, Any]] = None
    if attack_path.exists():
        with open(attack_path, "r", encoding="utf-8") as f:
            attack_actions_data = json.load(f)
        if verbose:
            print(f"Loaded attack actions from {attack_path.name}")
    elif verbose:
        print(f"Attack actions file not found: {attack_path}")

    agentic_features_data: Optional[Dict[str, Any]] = None
    if agentic_path.exists():
        with open(agentic_path, "r", encoding="utf-8") as f:
            raw_agentic = json.load(f)
        agents = raw_agentic.get("agents") if isinstance(raw_agentic, dict) else None
        agentic_features_data = agents if isinstance(agents, dict) else raw_agentic
        if verbose:
            print(f"Loaded agentic features from {agentic_path.name}")
    elif verbose:
        print(f"Agentic features file not found: {agentic_path}")

    return attack_actions_data, agentic_features_data


# ---------------------------------------------------------------------------
# Vector store (FAISS persist / load; build stays in Part 1 notebook)
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
# Defaults align with Index.ipynb (smaller chunks for retrieval precision)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128
MANIFEST_NAME = "rag_manifest.json"
# Cap how many retrieved sections are embedded in saved JSON (align with Plan.ipynb context size).
DEFAULT_EXPORT_RAG_TOP_N = 10


def _faiss_build_batch_size() -> int:
    raw = os.getenv("RAG_FAISS_BUILD_BATCH")
    if raw is None or not str(raw).strip():
        return 64
    try:
        return max(1, int(str(raw).strip()))
    except ValueError:
        return 64


def faiss_from_documents_batched(
    documents: List[Document],
    embedding: Embeddings,
    batch_size: Optional[int] = None,
) -> FAISS:
    """
    Build a FAISS store without embedding all ``page_content`` strings in one call
    (``FAISS.from_documents`` does that and can OOM on low-RAM machines).

    ``batch_size`` defaults to env ``RAG_FAISS_BUILD_BATCH`` or 64.
    """
    if not documents:
        raise ValueError("documents must be non-empty")
    bs = batch_size if batch_size is not None else _faiss_build_batch_size()
    first = documents[:bs]
    store = FAISS.from_documents(first, embedding)
    for i in range(bs, len(documents), bs):
        chunk = documents[i : i + bs]
        store.add_documents(chunk)
    return store


def save_vector_store(
    vector_store: FAISS,
    vector_store_dir: Path,
    knowledge_base: List[Dict[str, Any]],
    embed_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    n_chunks: Optional[int] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist FAISS index and a small JSON manifest next to it."""
    vector_store_dir = Path(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(vector_store_dir))
    manifest: Dict[str, Any] = {
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "n_source_docs": len(knowledge_base),
        "n_chunks": n_chunks,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path = vector_store_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def read_manifest(vector_store_dir: Path) -> Dict[str, Any]:
    p = Path(vector_store_dir) / MANIFEST_NAME
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name)
    return v is not None and str(v).strip().lower() in ("1", "true", "yes")


def _normalize_sentence_transformers_model_id(model: str) -> str:
    """Use HF repo ids for short names (manifest uses ``all-MiniLM-L6-v2``); keep local dirs as-is."""
    m = (model or "").strip() or DEFAULT_EMBED_MODEL
    if os.path.isdir(m):
        return m
    p = Path(m)
    try:
        if p.is_dir():
            return os.fspath(p.resolve())
    except OSError:
        pass
    if "/" not in m:
        return f"sentence-transformers/{m}"
    return m


def _sentence_transformer_embeddings_for_faiss_load(model: str) -> SentenceTransformerEmbeddings:
    """
    Load the same embedding model used when the FAISS index was built.

    Tries ``local_files_only=True`` first so an already-cached model does not hit the Hub
    (avoids DNS/offline failures and flaky httpx retry shutdown). If that fails and Hub
    offline env vars are set, raises a clear error. Otherwise retries with downloads enabled.

    Env:
    - ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``: only use local cache (no fallback download).
    - ``RAG_EMBED_ONLINE_FIRST``: skip local-only attempt (always allow Hub).
    """
    model_id = _normalize_sentence_transformers_model_id(model)
    hub_offline = _env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE")
    online_first = _env_truthy("RAG_EMBED_ONLINE_FIRST")

    def _with_local_only() -> SentenceTransformerEmbeddings:
        return SentenceTransformerEmbeddings(
            model_name=model_id,
            model_kwargs=sentence_transformer_model_kwargs(local_files_only=True),
        )

    def _with_hub() -> SentenceTransformerEmbeddings:
        return SentenceTransformerEmbeddings(
            model_name=model_id,
            model_kwargs=sentence_transformer_model_kwargs(),
        )

    if online_first:
        return _with_hub()
    if hub_offline:
        return _with_local_only()
    try:
        return _with_local_only()
    except Exception as e_local:
        try:
            return _with_hub()
        except Exception as e_hub:
            raise RuntimeError(
                f"Failed to load embedding model {model_id!r}: "
                f"cache/local load failed ({e_local!r}); "
                f"hub download failed ({e_hub!r}). "
                "Use a working network once to populate the Hugging Face cache, or copy "
                "the cache from another machine. For air-gapped runs with a filled cache, "
                "set HF_HUB_OFFLINE=1."
            ) from e_hub


def load_parent_store(vector_store_dir: Path) -> Dict[str, Any]:
    """Load rag_parents.json written by Part 1 (parent-child indexing). Empty dict if missing."""
    from utils.rag_index_build import load_parent_store as _load

    return _load(vector_store_dir)


def _doc_dedupe_key(doc: Any) -> Tuple[Any, ...]:
    """Stable key for a retrieved chunk (parent+child or legacy title)."""
    m = getattr(doc, "metadata", None) or {}
    pid, cid = m.get("parent_id"), m.get("child_index")
    if pid is not None and cid is not None:
        return ("pc", pid, cid)
    title = m.get("retrieval_title") or m.get("title", "")
    return ("legacy", title, m.get("source_file", ""))


def balance_vector_hits_by_source_file(
    hits: List[Tuple[Any, float]],
    top_k: int,
    *,
    max_per_source: Optional[int] = None,
) -> List[Tuple[Any, float]]:
    """
    Cap how many hits may come from one ``metadata['source_file']`` so retrieval is not
    dominated by a single KB file (e.g. a large JSON vs. sparse PDFs). Still respects
    global similarity order within those caps; remaining slots are filled from the
    sorted list without a per-source cap.

    ``max_per_source`` defaults to ``ceil(top_k / n_distinct_sources)`` among hits.
    """
    if not hits or top_k <= 0:
        return []

    best: Dict[Tuple[Any, ...], Tuple[Any, float]] = {}
    for doc, dist in hits:
        k = _doc_dedupe_key(doc)
        d = float(dist)
        if k not in best or d < float(best[k][1]):
            best[k] = (doc, d)

    unique_sorted = sorted(best.values(), key=lambda x: float(x[1]))
    sources = list(
        {
            (getattr(d, "metadata", None) or {}).get("source_file", "?")
            for d, _ in unique_sorted
        }
    )
    n_sources = len(sources) or 1
    if max_per_source is None:
        max_per_source = max(1, (top_k + n_sources - 1) // n_sources)

    picked: List[Tuple[Any, float]] = []
    seen: Set[Tuple[Any, ...]] = set()
    counts: Dict[str, int] = {}

    for doc, dist in unique_sorted:
        if len(picked) >= top_k:
            break
        dk = _doc_dedupe_key(doc)
        if dk in seen:
            continue
        src = (getattr(doc, "metadata", None) or {}).get("source_file", "?")
        if counts.get(src, 0) >= max_per_source:
            continue
        picked.append((doc, dist))
        seen.add(dk)
        counts[src] = counts.get(src, 0) + 1

    for doc, dist in unique_sorted:
        if len(picked) >= top_k:
            break
        dk = _doc_dedupe_key(doc)
        if dk in seen:
            continue
        picked.append((doc, dist))
        seen.add(dk)

    return picked[:top_k]


def load_vector_store_for_backbone(backbone: str = "minilm") -> FAISS:
    """Load FAISS index for ``minilm`` (default) or ``securebert`` bi-encoder store."""
    from utils.pipeline_paths import SECUREBERT_VECTOR_STORE_DIR, VECTOR_STORE_DIR

    key = (backbone or "minilm").strip().lower()
    if key in ("securebert", "sb", "securebert2"):
        return load_vector_store(SECUREBERT_VECTOR_STORE_DIR)
    return load_vector_store(VECTOR_STORE_DIR)


def load_vector_store(
    vector_store_dir: Path,
    embed_model: Optional[str] = None,
) -> FAISS:
    """Load FAISS from disk; uses manifest embed_model if embed_model is None."""
    vector_store_dir = Path(vector_store_dir)
    if not vector_store_dir.is_dir() or not any(vector_store_dir.iterdir()):
        raise FileNotFoundError(
            f"Vector store directory missing or empty: {vector_store_dir}. "
            "Run Index.ipynb first."
        )
    manifest = read_manifest(vector_store_dir)
    model = embed_model or manifest.get("embed_model") or DEFAULT_EMBED_MODEL
    embeddings = _sentence_transformer_embeddings_for_faiss_load(model)
    return FAISS.load_local(
        str(vector_store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def embeddings_from_vector_store(vector_store: Any) -> Embeddings:
    """Return the ``Embeddings`` instance backing a loaded LangChain FAISS store."""
    for attr in ("embedding_function", "embeddings"):
        e = getattr(vector_store, attr, None)
        if e is not None and hasattr(e, "embed_query") and hasattr(e, "embed_documents"):
            return e
    raise ValueError("Vector store has no LangChain-compatible Embeddings instance")


# ---------------------------------------------------------------------------
# SHAP party/tier + action-plan JSON export (Part 2)
# ---------------------------------------------------------------------------


def get_party_to_tier_mapping(sample: Dict[str, Any]) -> Dict[str, str]:
    """Map SHAP party keys to NIDS device ids (legacy RAN/Edge/Core still recognized)."""
    from utils.action_config import LEGACY_DEVICE_MAP, normalize_device_name

    shap_expl = sample.get("shap_explanation", {}) or {}
    party_contributions = shap_expl.get("party_contributions", {}) or {}
    party_to_tier: Dict[str, str] = {}
    for party_name in party_contributions.keys():
        if party_name in LEGACY_DEVICE_MAP:
            party_to_tier[party_name] = LEGACY_DEVICE_MAP[party_name]
            continue
        party_lower = party_name.lower()
        if party_lower.startswith("flowprobe") or party_lower.startswith("ran"):
            party_to_tier[party_name] = "FlowProbe"
        elif party_lower.startswith("firewall") or party_lower.startswith("edge"):
            party_to_tier[party_name] = "Firewall"
        elif party_lower.startswith("corerouter") or party_lower.startswith("core"):
            party_to_tier[party_name] = "CoreRouter"
        else:
            party_to_tier[party_name] = normalize_device_name(party_name)
    return party_to_tier


def get_dominant_party_info(sample: Dict[str, Any]) -> Tuple[str, str, float]:
    shap_expl = sample.get("shap_explanation", {}) or {}
    dominant_agent = shap_expl.get("dominant_agent", "")
    dominant_pct = shap_expl.get("dominant_contribution_pct", 0.0) or shap_expl.get(
        "party_contributions_pct", {}
    ).get(dominant_agent, 0.0)
    if isinstance(dominant_pct, dict):
        dominant_pct = dominant_pct.get(dominant_agent, 0.0)
    party_to_tier = get_party_to_tier_mapping(sample)
    network_tier = party_to_tier.get(dominant_agent, "") if dominant_agent else ""
    return (dominant_agent, network_tier, float(dominant_pct) * 100.0)


def _rag_result_score_fields(r: Dict[str, Any]) -> Dict[str, Any]:
    """Split ranking (CrossEncoder logit, can be negative) vs vector similarity (typically [0, 1])."""
    rank = float(r.get("score", 0.0) or 0.0)
    vs = r.get("vector_similarity")
    out: Dict[str, Any] = {"ranking_score": rank}
    if vs is not None:
        fv = float(vs)
        out["vector_similarity"] = fv
        out["similarity_score"] = fv
    else:
        out["similarity_score"] = rank
    ce = r.get("crossencoder_score")
    if ce is not None:
        out["crossencoder_score"] = float(ce)
    cb = r.get("colbert_score")
    if cb is not None:
        out["colbert_score"] = float(cb)
    bm = r.get("bm25_score")
    if bm is not None:
        out["bm25_score"] = float(bm)
    return out


def _rag_result_chunk_key_fields(r: Dict[str, Any]) -> Dict[str, Any]:
    """Stable chunk / section identity for offline joins (parent expansion may omit child_index)."""
    out: Dict[str, Any] = {}
    pid = r.get("parent_id")
    if pid is not None and str(pid).strip():
        out["parent_id"] = str(pid)
    cix = r.get("child_index")
    if cix is not None:
        try:
            out["child_index"] = int(cix)
        except (TypeError, ValueError):
            out["child_index"] = cix
    sf = r.get("source_file")
    if sf is not None and str(sf).strip():
        out["source_file"] = str(sf)
    return out


def _rag_result_export_fields(r: Dict[str, Any]) -> Dict[str, Any]:
    """Scores plus chunk keys for JSON exports (action plans, rerank comparison reports)."""
    merged = {**_rag_result_score_fields(r), **_rag_result_chunk_key_fields(r)}
    return merged


def convert_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


def _export_guidance_plan_fields(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    from utils.plan_prompts import agent_guidance_to_scoring_text

    if not plan or not isinstance(plan, dict):
        return {
            "guidance_text": "",
            "scoring_text": "",
        }
    scoring = agent_guidance_to_scoring_text(plan)
    return {
        "guidance_text": plan.get("guidance_text") or scoring,
        "format": plan.get("format", "plain_text"),
        "scoring_text": scoring,
        "parse_error": plan.get("parse_error"),
        # Legacy JSON field preserved when present
        "agent_guidance": plan.get("agent_guidance") if isinstance(plan.get("agent_guidance"), dict) else {},
    }


def _guidance_plan_differences(
    with_rag: Optional[Dict[str, Any]],
    without_rag: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    from utils.plan_prompts import agent_guidance_to_scoring_text, guidance_signature

    sw = guidance_signature(with_rag)
    sn = guidance_signature(without_rag)
    tw = agent_guidance_to_scoring_text(with_rag)
    tn = agent_guidance_to_scoring_text(without_rag)

    def _word_jaccard(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        wa = set(a.get("word_set") or ())
        wb = set(b.get("word_set") or ())
        if not wa and not wb:
            return 1.0
        u = wa | wb
        return len(wa & wb) / len(u) if u else 1.0

    return {
        "scoring_text_different": tw != tn,
        "guidance_word_jaccard": _word_jaccard(sw, sn) if not (sw.get("missing") or sn.get("missing")) else None,
        "with_rag_char_len": sw.get("char_len"),
        "without_rag_char_len": sn.get("char_len"),
    }


def save_comparison_file(
    sample: Dict[str, Any],
    query: str,
    rag_results: List[Dict[str, Any]],
    llm_response_with_rag: Optional[Dict[str, Any]],
    llm_response_without_rag: Optional[Dict[str, Any]],
    results_action_dir: Path,
    with_rag_file: Optional[Path],
    without_rag_file: Optional[Path],
) -> Path:
    comparison = {
        "sample_id": sample.get("sample_id"),
        "prediction": {
            "predicted_label": sample.get("predicted_label"),
            "confidence": float(sample.get("confidence", 0.0)),
        },
        "comparison_metadata": {
            "compared_at": datetime.now().isoformat(),
            "with_rag_file": with_rag_file.name if with_rag_file else None,
            "without_rag_file": without_rag_file.name if without_rag_file else None,
            "rag_query": query,
            "rag_documents_found": len(rag_results),
        },
        "rag_context": {
            "documents_used": len(rag_results),
            "top_documents": [
                {
                    **_rag_result_export_fields(r),
                    "title": r["title"],
                    "text_preview": (
                        r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
                    ),
                }
                for r in rag_results[:DEFAULT_EXPORT_RAG_TOP_N]
            ],
        },
        "action_plans": {
            "with_rag": _export_guidance_plan_fields(llm_response_with_rag),
            "without_rag": _export_guidance_plan_fields(llm_response_without_rag),
        },
        "differences": _guidance_plan_differences(
            llm_response_with_rag, llm_response_without_rag
        ),
    }
    comparison = convert_to_json_serializable(comparison)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = (
        results_action_dir / f"comparison_sample_{sample.get('sample_id')}_{timestamp}.json"
    )
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    return comparison_file


def save_action_plan(
    sample: Dict[str, Any],
    query: str,
    rag_results: List[Dict[str, Any]],
    llm_response: Optional[Dict[str, Any]],
    results_action_dir: Path,
    *,
    variant: Optional[str] = None,
    timestamp: Optional[str] = None,
    prompt_uses_rag: Optional[bool] = None,
) -> Path:
    result = {
        "sample_id": sample.get("sample_id"),
        "prediction": {
            "predicted_label": sample.get("predicted_label"),
            "confidence": float(sample.get("confidence", 0.0)),
        },
        "rag_info": {
            "documents_used": len(rag_results),
            "search_method": "vector_similarity",
            "search_queries": [query] if query else [],
            "top_results": [
                {
                    **_rag_result_export_fields(r),
                    "title": r["title"],
                    "text": r["text"],
                }
                for r in rag_results[:DEFAULT_EXPORT_RAG_TOP_N]
            ],
        },
        "llm_action_plan": llm_response,
        "processed_at": datetime.now().isoformat(),
    }
    if prompt_uses_rag is not None:
        result["prompt_uses_rag"] = prompt_uses_rag
    result = convert_to_json_serializable(result)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sid = sample.get("sample_id")
    if variant:
        output_file = results_action_dir / f"action_plan_sample_{sid}_{variant}_{ts}.json"
    else:
        output_file = results_action_dir / f"action_plan_sample_{sid}_{ts}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if not output_file.exists():
        raise FileNotFoundError(f"File was not created: {output_file}")
    return output_file


def save_rerank_comparison_report(
    sample: Dict[str, Any],
    query: str,
    modes_payload: Dict[str, Dict[str, Any]],
    results_action_dir: Path,
    *,
    retrieval_query_entries: Optional[List[Dict[str, str]]] = None,
    index_embed_model: Optional[str] = None,
) -> Path:
    """
    Save a single JSON comparing multiple rerank modes (each with RAG context + LLM output).

    ``modes_payload`` maps mode key -> payload dict with:
      - ``rag_results`` — LLM-visible top-K sections (typically 10), serialized under
        ``modes.<mode>.rag_info.top_results``.
      - ``candidate_pool`` (optional) — wider candidate pool (100 sections) used by
        ``Score.ipynb`` as the denominator for Recall@k / nDCG@k. Persisted under
        ``modes.<mode>.rag_info.candidate_pool``. Falls back to ``rag_results`` when omitted,
        so older callers remain compatible. Pool semantics per mode:

          * ``none``                — dense FAISS top-100 after MMR diversification.
          * ``bm25``                — BM25 top-100 over the full child-chunk corpus
                                       (single-stage; superset of the 10 shown to the LLM).
          * ``doct5query``          — BM25 top-100 over the T5-expanded corpus
                                       (single-stage; superset of the 10 shown to the LLM).
          * ``crossencoder``        — raw FAISS top-100 fed to the CrossEncoder
                                       (stage-1 shortlist, pre-rerank ordering).
          * ``bm25_crossencoder``   — BM25 top-100 fed to the CrossEncoder
                                       (stage-1 shortlist, pre-rerank ordering).
          * ``colbert``             — raw FAISS top-100 fed to the ColBERT reranker
                                       (stage-1 shortlist, pre-rerank ordering).

      - ``llm_response`` — the parsed LLM action-plan dict (or ``None``).
      - ``label`` / ``search_method`` — optional display strings.

    ``retrieval_query_entries`` — optional list of ``{"strategy_key": ..., "query_text": ...}``
    (multi-query retrieval). When set, the first entry's ``query_text`` is the rerank anchor.

    ``index_embed_model`` — embedding model id from the FAISS manifest (for analysis tooling).
    """
    results_action_dir = Path(results_action_dir)
    results_action_dir.mkdir(parents=True, exist_ok=True)

    search_method_labels = {
        "none": "dense_faiss_mmr_single_stage",
        "bm25": "bm25_full_corpus_single_stage",
        "doct5query": "bm25_doct5query_expanded_corpus_single_stage",
        "crossencoder": "dense_faiss_top100_then_crossencoder",
        "bm25_crossencoder": "bm25_top100_then_crossencoder",
        "colbert": "dense_faiss_top100_then_colbert",
        "crossencoder_colbert": "crossencoder_plus_colbert",
        "none_securebert": "securebert_faiss_mmr_single_stage",
        "crossencoder_securebert": "securebert_faiss_top100_then_crossencoder",
    }

    modes_out: Dict[str, Any] = {}
    for mode_key, payload in modes_payload.items():
        rag_results = payload.get("rag_results") or []
        # Optional wider pre-slice pool for retrieval-quality metrics (Recall@k, nDCG@k).
        # If not provided, falls back to ``rag_results`` so callers stay backward-compatible.
        candidate_pool = payload.get("candidate_pool") or rag_results
        llm_response = payload.get("llm_response")
        sm = search_method_labels.get(
            mode_key, str(payload.get("search_method") or mode_key)
        )
        mode_embed = payload.get("index_embed_model") or index_embed_model
        mode_backbone = payload.get("embedding_backbone")
        rag_info: Dict[str, Any] = {
            "documents_used": len(rag_results),
            "candidates_considered": len(candidate_pool),
            "search_queries": [query] if query else [],
            "top_results": [
                {
                    **_rag_result_export_fields(r),
                    "title": r["title"],
                    "text": r["text"],
                }
                for r in rag_results[:DEFAULT_EXPORT_RAG_TOP_N]
            ],
            "candidate_pool": [
                {
                    **_rag_result_export_fields(r),
                    "title": r["title"],
                    "text": r["text"],
                }
                for r in candidate_pool
            ],
        }
        if mode_embed:
            rag_info["index_embed_model"] = mode_embed
        if mode_backbone:
            rag_info["embedding_backbone"] = mode_backbone
        modes_out[mode_key] = {
            "label": payload.get("label") or mode_key,
            "search_method": sm,
            "rag_info": rag_info,
            "llm_action_plan": llm_response,
        }

    entries = retrieval_query_entries
    if not entries and (query or "").strip():
        entries = [{"strategy_key": "combined", "query_text": str(query)}]
    anchor_q = ""
    if entries:
        anchor_q = str(entries[0].get("query_text") or "")
    elif query:
        anchor_q = str(query)

    report = {
        "report_type": "rerank_ablation_comparison",
        "sample_id": sample.get("sample_id"),
        "prediction": {
            "predicted_label": sample.get("predicted_label"),
            "confidence": float(sample.get("confidence", 0.0)),
        },
        "rag_query": query,
        "retrieval_queries": entries or [],
        "mmr_rerank_anchor_query": anchor_q,
        "index_embed_model": index_embed_model,
        "modes": modes_out,
        "processed_at": datetime.now().isoformat(),
    }
    report = convert_to_json_serializable(report)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sid = sample.get("sample_id")
    out_path = results_action_dir / f"rerank_comparison_sample_{sid}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path
