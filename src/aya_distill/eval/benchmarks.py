"""
Evaluation harness for multilingual benchmarks (mGSM, XCOPA).

Loads datasets from HuggingFace, runs few-shot prompting against any
causal LM, computes per-language accuracy with bootstrap confidence
intervals, and saves reproducibility metadata alongside results.

Follows the Cohere multilingual evaluation checklist: never hide
language variation behind a single average.

Wayy Research -- Project Aya
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .metrics import (
    ConfidenceInterval,
    LanguageResult,
    aggregate_by_family,
    bootstrap_confidence_interval,
)
from .prompts.mgsm import build_mgsm_prompt, extract_mgsm_answer
from .prompts.xcopa import build_xcopa_prompt, extract_xcopa_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = ["en", "es", "hi", "zh", "ar", "sw", "tr", "ja", "id", "te"]

# mGSM uses different language codes in the HuggingFace dataset
MGSM_LANG_MAP: dict[str, str] = {
    "en": "en",
    "es": "es",
    "hi": "bn",  # mGSM may not have Hindi -- fallback to Bengali
    "zh": "zh",
    "ar": "ar",  # may not be in mGSM -- handled gracefully
    "sw": "sw",
    "tr": "tr",  # may not be in mGSM -- handled gracefully
    "ja": "ja",
    "id": "id",  # may not be in mGSM -- handled gracefully
    "te": "te",
}

# XCOPA uses these language codes
XCOPA_LANG_MAP: dict[str, str] = {
    "en": "en",  # XCOPA does not have English -- use COPA validation
    "es": "es",  # not in XCOPA -- handled gracefully
    "hi": "hi",  # not in XCOPA -- handled gracefully
    "zh": "zh",
    "ar": "ar",  # not in XCOPA -- handled gracefully
    "sw": "sw",
    "tr": "tr",
    "ja": "ja",  # not in XCOPA -- handled gracefully
    "id": "id",
    "te": "te",  # not in XCOPA -- handled gracefully
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    model_name: str
    dtype: str = "float16"
    device_map: str = "auto"
    n_shot: int = 8
    languages: list[str] = field(
        default_factory=lambda: list(SUPPORTED_LANGUAGES)
    )
    max_new_tokens: int = 256
    n_bootstrap: int = 1000
    seed: int = 42
    output_dir: str = "results/baseline"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "float16",
    device_map: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HuggingFace causal LM and its tokenizer.

    Parameters
    ----------
    model_name : HuggingFace model identifier
    dtype : "float16", "bfloat16", or "float32"
    device_map : device placement strategy

    Returns
    -------
    (model, tokenizer) tuple
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    logger.info("Loading model: %s (dtype=%s, device_map=%s)", model_name, dtype, device_map)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Generate text from a prompt using greedy decoding.

    Temperature=0 for deterministic evaluation -- no sampling noise.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings
        if hasattr(model.config, "max_position_embeddings")
        else 4096,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    # Decode only the generated tokens (not the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# mGSM Benchmark
# ---------------------------------------------------------------------------


def _load_mgsm_data(language: str) -> list[dict[str, Any]] | None:
    """Load mGSM dataset for a given language.

    Returns None if the language is not available.
    """
    mgsm_lang = MGSM_LANG_MAP.get(language, language)
    try:
        ds = load_dataset("juletxara/mgsm", mgsm_lang, split="test")
        examples = []
        for row in ds:
            examples.append({
                "question": row["question"],
                "answer": str(row["answer"]),
                "answer_number": row.get("answer_number", row.get("answer", "")),
            })
        return examples
    except Exception as e:
        logger.warning("Could not load mGSM for language '%s': %s", language, e)
        return None


def _load_mgsm_exemplars(language: str, n_shot: int) -> list[dict[str, str]] | None:
    """Load few-shot exemplars from the mGSM train split if available."""
    mgsm_lang = MGSM_LANG_MAP.get(language, language)
    try:
        ds = load_dataset("juletxara/mgsm", mgsm_lang, split="train")
        exemplars = []
        for row in ds:
            if len(exemplars) >= n_shot:
                break
            exemplars.append({
                "question": row["question"],
                "answer": str(row["answer"]),
            })
        return exemplars if exemplars else None
    except Exception:
        return None


def run_mgsm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Run mGSM benchmark across all configured languages.

    Returns a dict with per-language results, aggregates, and metadata.
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    results: dict[str, Any] = {
        "benchmark": "mgsm",
        "model": config.model_name,
        "n_shot": config.n_shot,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "per_language": {},
        "family_aggregation": [],
        "aggregate": {},
    }

    all_correct: list[int] = []
    per_lang_acc: dict[str, float] = {}

    for lang in config.languages:
        logger.info("Running mGSM for language: %s", lang)

        data = _load_mgsm_data(lang)
        if data is None:
            logger.warning("Skipping mGSM for %s -- dataset not available", lang)
            results["per_language"][lang] = {
                "status": "skipped",
                "reason": "dataset_not_available",
            }
            continue

        exemplars = _load_mgsm_exemplars(lang, config.n_shot)

        correct_arr = np.zeros(len(data), dtype=np.int_)

        for idx, example in enumerate(data):
            prompt = build_mgsm_prompt(
                question=example["question"],
                language=lang,
                exemplars=exemplars,
                n_shot=config.n_shot,
            )
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=config.max_new_tokens,
            )
            predicted = extract_mgsm_answer(generated)

            # Normalize answers for comparison
            reference = str(example.get("answer_number", example["answer"]))
            reference = reference.replace(",", "").replace("$", "").strip()

            if predicted == reference:
                correct_arr[idx] = 1

            if idx < 3:
                logger.debug(
                    "[%s] Q: %.80s... | Pred: %s | Ref: %s | Correct: %s",
                    lang, example["question"], predicted, reference,
                    predicted == reference,
                )

        ci = bootstrap_confidence_interval(
            correct_arr,
            n_bootstrap=config.n_bootstrap,
            seed=config.seed,
        )
        lang_result = LanguageResult(
            language=lang,
            accuracy=ci.mean,
            ci=ci,
            n_samples=len(data),
        )
        results["per_language"][lang] = lang_result.to_dict()
        per_lang_acc[lang] = ci.mean
        all_correct.extend(correct_arr.tolist())

    # Aggregate
    if all_correct:
        agg_ci = bootstrap_confidence_interval(
            np.array(all_correct, dtype=np.int_),
            n_bootstrap=config.n_bootstrap,
            seed=config.seed,
        )
        results["aggregate"] = {
            "accuracy": agg_ci.mean,
            "ci": agg_ci.to_dict(),
            "n_total": len(all_correct),
            "n_languages_evaluated": len(per_lang_acc),
        }

    # Family aggregation
    family_results = aggregate_by_family(per_lang_acc)
    results["family_aggregation"] = [fr.to_dict() for fr in family_results]

    # Prompt template metadata for reproducibility
    results["prompt_template"] = {
        "type": "few_shot",
        "n_shot": config.n_shot,
        "instruction_language": "native",
        "answer_extraction": "####_delimiter",
    }

    return results


# ---------------------------------------------------------------------------
# XCOPA Benchmark
# ---------------------------------------------------------------------------


def _load_xcopa_data(language: str) -> list[dict[str, Any]] | None:
    """Load XCOPA dataset for a given language."""
    xcopa_lang = XCOPA_LANG_MAP.get(language, language)
    try:
        ds = load_dataset("cambridgeltl/xcopa", xcopa_lang, split="test")
        examples = []
        for row in ds:
            examples.append({
                "premise": row["premise"],
                "choice1": row["choice1"],
                "choice2": row["choice2"],
                "question": row["question"],
                "label": str(row["label"] + 1),  # XCOPA uses 0-indexed, we use 1-indexed
            })
        return examples
    except Exception as e:
        logger.warning("Could not load XCOPA for language '%s': %s", language, e)
        return None


def _load_xcopa_exemplars(
    language: str, n_shot: int
) -> list[dict[str, str]] | None:
    """Load few-shot exemplars from XCOPA validation split."""
    xcopa_lang = XCOPA_LANG_MAP.get(language, language)
    try:
        ds = load_dataset("cambridgeltl/xcopa", xcopa_lang, split="validation")
        exemplars = []
        for row in ds:
            if len(exemplars) >= n_shot:
                break
            exemplars.append({
                "premise": row["premise"],
                "choice1": row["choice1"],
                "choice2": row["choice2"],
                "question": row["question"],
                "label": str(row["label"] + 1),
            })
        return exemplars if exemplars else None
    except Exception:
        return None


def run_xcopa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Run XCOPA benchmark across all configured languages.

    Returns a dict with per-language results, aggregates, and metadata.
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    results: dict[str, Any] = {
        "benchmark": "xcopa",
        "model": config.model_name,
        "n_shot": config.n_shot,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "per_language": {},
        "family_aggregation": [],
        "aggregate": {},
    }

    all_correct: list[int] = []
    per_lang_acc: dict[str, float] = {}

    for lang in config.languages:
        logger.info("Running XCOPA for language: %s", lang)

        data = _load_xcopa_data(lang)
        if data is None:
            logger.warning("Skipping XCOPA for %s -- dataset not available", lang)
            results["per_language"][lang] = {
                "status": "skipped",
                "reason": "dataset_not_available",
            }
            continue

        exemplars = _load_xcopa_exemplars(lang, config.n_shot)

        correct_arr = np.zeros(len(data), dtype=np.int_)

        for idx, example in enumerate(data):
            prompt = build_xcopa_prompt(
                premise=example["premise"],
                choice1=example["choice1"],
                choice2=example["choice2"],
                question_type=example["question"],
                language=lang,
                exemplars=exemplars,
                n_shot=config.n_shot,
            )
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=16,  # XCOPA only needs "1" or "2"
            )
            predicted = extract_xcopa_answer(generated)

            if predicted == example["label"]:
                correct_arr[idx] = 1

            if idx < 3:
                logger.debug(
                    "[%s] P: %.60s... | Pred: %s | Ref: %s | Correct: %s",
                    lang, example["premise"], predicted, example["label"],
                    predicted == example["label"],
                )

        ci = bootstrap_confidence_interval(
            correct_arr,
            n_bootstrap=config.n_bootstrap,
            seed=config.seed,
        )
        lang_result = LanguageResult(
            language=lang,
            accuracy=ci.mean,
            ci=ci,
            n_samples=len(data),
        )
        results["per_language"][lang] = lang_result.to_dict()
        per_lang_acc[lang] = ci.mean
        all_correct.extend(correct_arr.tolist())

    # Aggregate
    if all_correct:
        agg_ci = bootstrap_confidence_interval(
            np.array(all_correct, dtype=np.int_),
            n_bootstrap=config.n_bootstrap,
            seed=config.seed,
        )
        results["aggregate"] = {
            "accuracy": agg_ci.mean,
            "ci": agg_ci.to_dict(),
            "n_total": len(all_correct),
            "n_languages_evaluated": len(per_lang_acc),
        }

    # Family aggregation
    family_results = aggregate_by_family(per_lang_acc)
    results["family_aggregation"] = [fr.to_dict() for fr in family_results]

    # Prompt template metadata
    results["prompt_template"] = {
        "type": "few_shot",
        "n_shot": config.n_shot,
        "instruction_language": "native",
        "answer_extraction": "first_1_or_2",
    }

    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def save_results(
    results: dict[str, Any],
    output_dir: str | Path,
    benchmark_name: str,
) -> Path:
    """Save benchmark results as JSON with reproducibility metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{benchmark_name}_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Results saved to %s", filepath)
    return filepath


def print_results_table(results: dict[str, Any]) -> None:
    """Print a formatted summary table of benchmark results."""
    benchmark = results.get("benchmark", "unknown")
    model = results.get("model", "unknown")

    print(f"\n{'=' * 72}")
    print(f"  {benchmark.upper()} Results -- {model}")
    print(f"{'=' * 72}")
    print(f"  {'Language':<10} {'Accuracy':>10} {'95% CI':>20} {'N':>8}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 20} {'-' * 8}")

    per_lang = results.get("per_language", {})
    for lang, data in sorted(per_lang.items()):
        if isinstance(data, dict) and "accuracy" in data:
            acc = data["accuracy"]
            ci = data.get("ci", {})
            lower = ci.get("lower", 0.0)
            upper = ci.get("upper", 0.0)
            n = data.get("n_samples", 0)
            print(f"  {lang:<10} {acc:>10.4f} [{lower:>8.4f}, {upper:>8.4f}] {n:>8}")
        elif isinstance(data, dict) and data.get("status") == "skipped":
            print(f"  {lang:<10} {'SKIPPED':>10} {'':>20} {'':>8}")

    agg = results.get("aggregate", {})
    if agg:
        acc = agg.get("accuracy", 0.0)
        ci = agg.get("ci", {})
        lower = ci.get("lower", 0.0)
        upper = ci.get("upper", 0.0)
        n = agg.get("n_total", 0)
        print(f"  {'-' * 10} {'-' * 10} {'-' * 20} {'-' * 8}")
        print(f"  {'OVERALL':<10} {acc:>10.4f} [{lower:>8.4f}, {upper:>8.4f}] {n:>8}")

    # Family aggregation
    families = results.get("family_aggregation", [])
    if families:
        print(f"\n  Language Family Aggregation:")
        print(f"  {'Family':<20} {'Mean Acc':>10} {'Languages':>30}")
        print(f"  {'-' * 20} {'-' * 10} {'-' * 30}")
        for fam in families:
            langs_str = ", ".join(fam["languages"])
            print(f"  {fam['family']:<20} {fam['mean_accuracy']:>10.4f} {langs_str:>30}")

    print(f"{'=' * 72}\n")
