"""Helpers for the LLM runtime benchmark notebook.

The benchmark preserves the exact model names discussed in chat, but only
attempts generation for entries that plausibly map to a runnable text generator.
Encoder-only or framework-like entries return a structured ``unsupported``
result instead of fabricating output.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.label_config import TARGET_CLASSES
from utils.pipeline_paths import AGENTIC_FEATURES_PATH, INFERENCE_ROOT, ensure_pipeline_dirs
from utils.plan_prompts import (
    LLM_SYSTEM_MESSAGE,
    create_prompt_without_RAG,
    parse_llm_json_response,
)


@dataclass(frozen=True)
class BenchmarkModelSpec:
    model_name: str
    backend: str
    note: str
    model_id_env: Optional[str] = None
    api_key_env: Optional[str] = None
    api_model_env: Optional[str] = None
    api_base_env: Optional[str] = None
    trust_remote_code: bool = False


# Encoder-only models are scored in Score.ipynb / Plan.ipynb (not runtime-benchmarked here).
ENCODER_MODEL_NOTES: Dict[str, str] = {
    "SecureBERT": "cisco-ai/SecureBERT2.0-biencoder — parallel FAISS index + metrics",
    "CySecBERT": "markusbayer/CySecBERT — retrieval / evaluation cosine metrics",
    "CTI-BERT": "ibm-research/CTI-BERT — retrieval / evaluation cosine metrics",
}

MODEL_SPECS: Dict[str, BenchmarkModelSpec] = {
    "Foundation-Sec-8B": BenchmarkModelSpec(
        model_name="Foundation-Sec-8B",
        backend="hf_causal",
        model_id_env="FOUNDATION_SEC_8B_MODEL_ID",
        note="Attempts local Hugging Face causal generation when the repo id/path is configured.",
    ),
    "Foundation-Sec-8B-Reasoning": BenchmarkModelSpec(
        model_name="Foundation-Sec-8B-Reasoning",
        backend="hf_causal",
        model_id_env="FOUNDATION_SEC_8B_REASONING_MODEL_ID",
        note="Attempts local Hugging Face causal generation when the repo id/path is configured.",
    ),
    "SecGPT": BenchmarkModelSpec(
        model_name="SecGPT",
        backend="mixed_api_or_hf",
        model_id_env="SECGPT_MODEL_ID",
        api_key_env="SECGPT_API_KEY",
        api_model_env="SECGPT_API_MODEL",
        api_base_env="SECGPT_API_BASE_URL",
        note=(
            "Uses an OpenAI-compatible API when SECGPT_API_* vars are set, "
            "otherwise falls back to a local Hugging Face causal model when "
            "SECGPT_MODEL_ID is configured."
        ),
    ),
    "CyberInstruct": BenchmarkModelSpec(
        model_name="CyberInstruct",
        backend="mixed_api_or_hf",
        model_id_env="CYBERINSTRUCT_MODEL_ID",
        api_key_env="CYBERINSTRUCT_API_KEY",
        api_model_env="CYBERINSTRUCT_API_MODEL",
        api_base_env="CYBERINSTRUCT_API_BASE_URL",
        note=(
            "Uses an OpenAI-compatible API when CYBERINSTRUCT_API_* vars are set, "
            "otherwise falls back to a local Hugging Face causal model when "
            "CYBERINSTRUCT_MODEL_ID is configured."
        ),
    ),
}


DEFAULT_GENERATION_KWARGS: Dict[str, Any] = {
    "max_new_tokens": 320,
    "temperature": 0.2,
    "top_p": 0.9,
    "do_sample": False,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _latest_prediction_files(root: Path = INFERENCE_ROOT) -> Dict[str, Path]:
    ensure_pipeline_dirs()
    root = Path(root)
    by_class: Dict[str, Path] = {}
    for label in TARGET_CLASSES:
        files = sorted(
            (root / label).glob("predictions_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            by_class[label] = files[0]
    return by_class


def load_latest_inference_samples(
    root: Path = INFERENCE_ROOT,
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Load the newest per-class inference JSON from ``output/inference/<CLASS>/``."""
    latest = _latest_prediction_files(root)
    if not latest:
        raise FileNotFoundError(
            f"No per-class prediction JSON files found under {Path(root).resolve()}"
        )

    rows: List[Dict[str, Any]] = []
    if verbose:
        print("Latest inference files:")
    for label in TARGET_CLASSES:
        fp = latest.get(label)
        if fp is None:
            continue
        if verbose:
            print(f"  [{label}] {fp.name}")
        payload = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list of predictions in {fp}")
        for row in payload:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["_source_file"] = fp.name
            item["_source_label_dir"] = label
            rows.append(item)

    rows.sort(
        key=lambda item: (
            str(item.get("predicted_label") or item.get("true_label") or ""),
            int(item.get("sample_id", -1)),
        )
    )
    if verbose:
        print(f"Loaded {len(rows)} sample(s) from {len(latest)} class file(s).")
    return rows


def select_benchmark_sample(
    samples: List[Dict[str, Any]],
    *,
    predicted_label: Optional[str] = None,
    sample_id: Optional[int] = None,
    index: int = 0,
) -> Dict[str, Any]:
    """Pick one inference sample for all notebook model cells."""
    if not samples:
        raise ValueError("samples must be non-empty")

    selected = samples
    if predicted_label:
        selected = [
            s
            for s in selected
            if str(s.get("predicted_label", "")).strip().upper() == predicted_label.strip().upper()
        ]
    if sample_id is not None:
        selected = [s for s in selected if int(s.get("sample_id", -1)) == int(sample_id)]
    if not selected:
        raise ValueError("No samples matched the requested selection.")
    return selected[index]


def load_agentic_features() -> Dict[str, Any]:
    if not AGENTIC_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing agentic features file: {AGENTIC_FEATURES_PATH}")
    payload = json.loads(AGENTIC_FEATURES_PATH.read_text(encoding="utf-8"))
    agents = payload.get("agents") if isinstance(payload, dict) else None
    return agents if isinstance(agents, dict) else payload


def build_prompt_bundle(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Build the prompt sent to each model from the inference sample only."""
    agentic_features = load_agentic_features()
    user_prompt = create_prompt_without_RAG(sample, agentic_features)
    combined_prompt = (
        f"System:\n{LLM_SYSTEM_MESSAGE}\n\n"
        f"User:\n{user_prompt}"
    )
    return {
        "system_prompt": LLM_SYSTEM_MESSAGE,
        "user_prompt": user_prompt,
        "combined_prompt": combined_prompt,
        "sample_id": sample.get("sample_id"),
        "predicted_label": sample.get("predicted_label"),
    }


def sample_overview(sample: Dict[str, Any]) -> Dict[str, Any]:
    shap = sample.get("shap_explanation", {}) or {}
    return {
        "sample_id": sample.get("sample_id"),
        "predicted_label": sample.get("predicted_label"),
        "confidence": sample.get("confidence"),
        "dominant_agent": shap.get("dominant_agent"),
        "dominant_contribution_pct": shap.get("dominant_contribution_pct"),
        "source_file": sample.get("_source_file"),
        "source_label_dir": sample.get("_source_label_dir"),
    }


def model_support_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, spec in MODEL_SPECS.items():
        rows.append(
            {
                "model_name": name,
                "backend": spec.backend,
                "note": spec.note,
                "model_id_env": spec.model_id_env,
                "api_model_env": spec.api_model_env,
            }
        )
    return rows


def _base_result(
    spec: BenchmarkModelSpec,
    prompt_bundle: Dict[str, Any],
    *,
    status: str,
    runtime_seconds: Optional[float],
    output_text: str = "",
    error: str = "",
    backend_used: Optional[str] = None,
) -> Dict[str, Any]:
    parsed_output = parse_llm_json_response(output_text) if output_text else None
    return {
        "model_name": spec.model_name,
        "status": status,
        "backend": spec.backend,
        "backend_used": backend_used or spec.backend,
        "runtime_seconds": runtime_seconds,
        "prompt_preview": prompt_bundle["user_prompt"][:500],
        "output_text": output_text,
        "parsed_output": parsed_output,
        "error": error,
        "note": spec.note,
        "sample_id": prompt_bundle.get("sample_id"),
        "predicted_label": prompt_bundle.get("predicted_label"),
    }


def _render_generated_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict):
            if "generated_text" in first:
                val = first["generated_text"]
                if isinstance(val, str):
                    return val.strip()
                if isinstance(val, list):
                    content = []
                    for item in val:
                        if isinstance(item, dict):
                            maybe = item.get("content")
                            if isinstance(maybe, str):
                                content.append(maybe)
                    return "\n".join(content).strip()
            if "text" in first and isinstance(first["text"], str):
                return first["text"].strip()
    return str(raw).strip()


def _needs_configuration(
    spec: BenchmarkModelSpec,
    prompt_bundle: Dict[str, Any],
    message: str,
) -> Dict[str, Any]:
    return _base_result(
        spec,
        prompt_bundle,
        status="needs_configuration",
        runtime_seconds=None,
        error=message,
    )


def _run_hf_causal(
    spec: BenchmarkModelSpec,
    prompt_bundle: Dict[str, Any],
    *,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_target = os.getenv(spec.model_id_env or "", "").strip() if spec.model_id_env else ""
    if not model_target:
        return _needs_configuration(
            spec,
            prompt_bundle,
            (
                f"Set {spec.model_id_env} to a local model path or Hugging Face repo id "
                f"for {spec.model_name}."
            ),
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception as exc:
        return _base_result(
            spec,
            prompt_bundle,
            status="error",
            runtime_seconds=None,
            error=f"transformers import failed: {exc}",
            backend_used="hf_causal",
        )

    kwargs = dict(DEFAULT_GENERATION_KWARGS)
    if generation_kwargs:
        kwargs.update(generation_kwargs)
    hf_token = os.getenv("HF_TOKEN") or None

    start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_target,
            trust_remote_code=spec.trust_remote_code,
            token=hf_token,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_target,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=spec.trust_remote_code,
            token=hf_token,
        )

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompt_bundle["system_prompt"]},
                    {"role": "user", "content": prompt_bundle["user_prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = prompt_bundle["combined_prompt"]

        text_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        generation_config = {
            "max_new_tokens": int(kwargs["max_new_tokens"]),
            "do_sample": bool(kwargs["do_sample"]),
            "return_full_text": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if generation_config["do_sample"]:
            generation_config["temperature"] = float(kwargs["temperature"])
            generation_config["top_p"] = float(kwargs["top_p"])

        generated = text_pipe(
            prompt_text,
            **generation_config,
        )
        runtime = time.perf_counter() - start
        return _base_result(
            spec,
            prompt_bundle,
            status="completed",
            runtime_seconds=runtime,
            output_text=_render_generated_text(generated),
            backend_used="hf_causal",
        )
    except Exception:
        runtime = time.perf_counter() - start
        return _base_result(
            spec,
            prompt_bundle,
            status="error",
            runtime_seconds=runtime,
            error=traceback.format_exc(limit=6),
            backend_used="hf_causal",
        )


def _run_openai_compatible(
    spec: BenchmarkModelSpec,
    prompt_bundle: Dict[str, Any],
    *,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    api_key = os.getenv(spec.api_key_env or "", "").strip() if spec.api_key_env else ""
    model_name = os.getenv(spec.api_model_env or "", "").strip() if spec.api_model_env else ""
    base_url = os.getenv(spec.api_base_env or "", "").strip() if spec.api_base_env else ""

    if not api_key or not model_name:
        missing = []
        if spec.api_key_env and not api_key:
            missing.append(spec.api_key_env)
        if spec.api_model_env and not model_name:
            missing.append(spec.api_model_env)
        return _needs_configuration(
            spec,
            prompt_bundle,
            f"Set {', '.join(missing)} for {spec.model_name} to use the API path.",
        )

    try:
        from openai import OpenAI
    except Exception as exc:
        return _base_result(
            spec,
            prompt_bundle,
            status="error",
            runtime_seconds=None,
            error=f"openai import failed: {exc}",
            backend_used="openai_compatible",
        )

    kwargs = dict(DEFAULT_GENERATION_KWARGS)
    if generation_kwargs:
        kwargs.update(generation_kwargs)

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt_bundle["system_prompt"]},
                {"role": "user", "content": prompt_bundle["user_prompt"]},
            ],
            temperature=float(kwargs["temperature"]),
            top_p=float(kwargs["top_p"]),
            max_tokens=int(kwargs["max_new_tokens"]),
        )
        runtime = time.perf_counter() - start
        text = ""
        if response.choices:
            message = response.choices[0].message
            text = message.content or ""
        return _base_result(
            spec,
            prompt_bundle,
            status="completed",
            runtime_seconds=runtime,
            output_text=text,
            backend_used="openai_compatible",
        )
    except Exception:
        runtime = time.perf_counter() - start
        return _base_result(
            spec,
            prompt_bundle,
            status="error",
            runtime_seconds=runtime,
            error=traceback.format_exc(limit=6),
            backend_used="openai_compatible",
        )


def run_model_benchmark(
    model_name: str,
    sample: Dict[str, Any],
    *,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one benchmarked model against one inference sample."""
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model name: {model_name}")
    spec = MODEL_SPECS[model_name]
    prompt_bundle = build_prompt_bundle(sample)

    if spec.backend == "unsupported":
        return _base_result(
            spec,
            prompt_bundle,
            status="unsupported",
            runtime_seconds=0.0,
            backend_used="unsupported",
        )
    if spec.backend == "hf_causal":
        return _run_hf_causal(spec, prompt_bundle, generation_kwargs=generation_kwargs)
    if spec.backend == "mixed_api_or_hf":
        api_ready = bool(
            spec.api_key_env
            and spec.api_model_env
            and os.getenv(spec.api_key_env, "").strip()
            and os.getenv(spec.api_model_env, "").strip()
        )
        if api_ready:
            return _run_openai_compatible(
                spec,
                prompt_bundle,
                generation_kwargs=generation_kwargs,
            )
        return _run_hf_causal(spec, prompt_bundle, generation_kwargs=generation_kwargs)
    raise ValueError(f"Unhandled backend {spec.backend!r} for {model_name}")


def results_to_frame(results: Iterable[Dict[str, Any]]):
    import pandas as pd

    rows = []
    for result in results:
        rows.append(
            {
                "model_name": result.get("model_name"),
                "status": result.get("status"),
                "backend_used": result.get("backend_used"),
                "runtime_seconds": result.get("runtime_seconds"),
                "predicted_label": result.get("predicted_label"),
                "sample_id": result.get("sample_id"),
                "error": result.get("error"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and "runtime_seconds" in df.columns:
        df = df.sort_values(
            by=["status", "runtime_seconds"],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    return df


def completed_runtime_frame(results: Iterable[Dict[str, Any]]):
    df = results_to_frame(results)
    if df.empty:
        return df
    return df[df["status"] == "completed"].sort_values("runtime_seconds").reset_index(drop=True)


def display_result(result: Dict[str, Any]) -> None:
    """Notebook-friendly pretty print for one model result."""
    print(f"Model: {result.get('model_name')}")
    print(f"Status: {result.get('status')}")
    print(f"Backend used: {result.get('backend_used')}")
    print(f"Runtime (s): {result.get('runtime_seconds')}")
    if result.get("error"):
        print("Error / note:")
        print(result["error"])
    if result.get("output_text"):
        print("\nOutput preview:")
        preview = result["output_text"][:1200]
        print(preview + ("..." if len(result["output_text"]) > 1200 else ""))
    elif result.get("parsed_output"):
        print("\nParsed output:")
        print(json.dumps(result["parsed_output"], indent=2, ensure_ascii=False, default=_json_default))

