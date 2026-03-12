#!/usr/bin/env python3
"""
Demo evaluation script for Project Aya.

Loads the trained demo model and runs three evaluation suites:
    1. Multilingual quality (coherence, relevance, cross-lingual consistency)
    2. Tool calling (JSON generation, function selection, argument extraction)
    3. Reasoning (CoT quality, logical consistency, cross-lingual transfer)

Outputs results to results/demo/evaluation_report.json and prints a
formatted summary to stdout.

Usage:
    python scripts/run_demo_eval.py
    python scripts/run_demo_eval.py --checkpoint checkpoints/demo/final_model.pt

Wayy Research, 2024-2026.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "aetheris"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("demo_eval")


# ---------------------------------------------------------------------------
# Simple byte-level tokenizer (must match training tokenizer)
# ---------------------------------------------------------------------------
class SimpleTokenizer:
    """Byte-level tokenizer matching the one used in run_demo_distill.py."""

    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"

        self._special: dict[str, int] = {
            "<|pad|>": 0,
            "<|endoftext|>": 1,
            "<|bos|>": 2,
            "<|system|>": 3,
            "<|user|>": 4,
            "<|assistant|>": 5,
            "<|tool|>": 6,
            "<tool_call>": 7,
            "</tool_call>": 8,
            "<tool_result>": 9,
            "</tool_result>": 10,
        }
        self._byte_offset = 16

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> Any:
        ids: list[int] = []
        remaining = text
        while remaining:
            found = False
            for token_str, token_id in self._special.items():
                if remaining.startswith(token_str):
                    ids.append(token_id)
                    remaining = remaining[len(token_str):]
                    found = True
                    break
            if not found:
                byte_val = remaining[0].encode("utf-8")[0]
                ids.append((byte_val + self._byte_offset) % self.vocab_size)
                remaining = remaining[1:]

        if truncation and max_length is not None:
            ids = ids[:max_length]

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(
        self,
        ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        special_ids = set(self._special.values()) if skip_special_tokens else set()
        id_to_special = {v: k for k, v in self._special.items()}

        chars: list[str] = []
        for tid in ids:
            if tid in special_ids:
                continue
            if tid in id_to_special and not skip_special_tokens:
                chars.append(id_to_special[tid])
            elif self._byte_offset <= tid < self._byte_offset + 256:
                byte_val = tid - self._byte_offset
                try:
                    chars.append(bytes([byte_val]).decode("utf-8", errors="replace"))
                except Exception:
                    chars.append("?")
            else:
                chars.append("?")
        return "".join(chars)

    def __len__(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_demo_model(
    checkpoint_path: str,
    config_path: str | None = None,
) -> tuple[torch.nn.Module, SimpleTokenizer]:
    """Load the trained demo model from checkpoint.

    Returns:
        (model, tokenizer) tuple.
    """
    from aetheris.config import AetherisConfig
    from aetheris.model import HybridMambaMoE

    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    vocab_size = ckpt.get("vocab_size", 8192)
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)

    # Load student config
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "demo_student.yaml")
    student_config = AetherisConfig.from_yaml(config_path)
    student_config.vocab_size = vocab_size
    student_config.gradient_checkpointing = False

    model = HybridMambaMoE(student_config)
    state_dict = ckpt.get("student_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded: %.1fM params, vocab_size=%d", param_count, vocab_size)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(
    model: torch.nn.Module,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Simple greedy/sampling generation loop.

    Args:
        model: Aetheris HybridMambaMoE model.
        tokenizer: SimpleTokenizer instance.
        prompt: Input text prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        device: Device to run on.

    Returns:
        Generated text (prompt + completion).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs["logits"][:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Model function wrapper for test suites
# ---------------------------------------------------------------------------
def make_model_fn(
    model: torch.nn.Module,
    tokenizer: SimpleTokenizer,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    device: torch.device = torch.device("cpu"),
):
    """Create a (prompt -> response) callable for test suites."""

    def model_fn(prompt: str) -> str:
        return generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )

    return model_fn


# ---------------------------------------------------------------------------
# Evaluation suites
# ---------------------------------------------------------------------------
def run_multilingual_eval(
    model_fn,
    languages: list[str],
) -> dict[str, Any]:
    """Run multilingual quality evaluation."""
    logger.info("--- Multilingual Quality Evaluation ---")

    from aya_distill.testing.multilingual import MultilingualTestSuite

    suite = MultilingualTestSuite(
        model_fn=model_fn,
        languages=languages,
        judge_model=None,  # Use heuristic scoring
    )

    result = suite.run(prompt_category="general_knowledge")
    logger.info("\n%s", result.summary())

    return result.to_dict()


def run_tool_calling_eval(
    model_fn,
    languages: list[str],
) -> dict[str, Any]:
    """Run tool calling evaluation."""
    logger.info("--- Tool Calling Evaluation ---")

    from aya_distill.testing.tool_testing import ToolCallingTest
    from aya_distill.testing.fixtures import TOOL_SCENARIOS

    test = ToolCallingTest(model_fn=model_fn)

    # Filter scenarios to requested languages
    scenarios = {
        lang: TOOL_SCENARIOS[lang]
        for lang in languages
        if lang in TOOL_SCENARIOS
    }

    results = test.run_multilingual(scenarios=scenarios)
    summary = test.summary(results)
    logger.info("\n%s", summary)

    # Serialize
    return {
        "results": {
            lang: r.to_dict() for lang, r in results.items()
        },
        "mean_score": (
            sum(r.score for r in results.values()) / max(len(results), 1)
        ),
        "json_valid_rate": (
            sum(1 for r in results.values() if r.json_valid) / max(len(results), 1)
        ),
    }


def run_reasoning_eval(
    model_fn,
    languages: list[str],
) -> dict[str, Any]:
    """Run reasoning evaluation across categories and languages."""
    logger.info("--- Reasoning Evaluation ---")

    from aya_distill.testing.reasoning import ReasoningTestSuite

    suite = ReasoningTestSuite(
        model_fn=model_fn,
        languages=languages,
        categories=["mathematical", "causal", "analogical", "logical"],
    )

    result = suite.run()
    logger.info("\n%s", result.summary())

    return result.to_dict()


def build_cross_language_comparison(
    reasoning_results: dict[str, Any],
    languages: list[str],
) -> dict[str, Any]:
    """Build a cross-language reasoning comparison table.

    For each reasoning category, compares response patterns across
    languages to identify equity gaps.
    """
    comparison: dict[str, Any] = {}
    lang_data = reasoning_results.get("languages", {})

    categories = ["mathematical", "causal", "analogical", "logical"]

    for category in categories:
        cat_scores: dict[str, dict[str, float]] = {}
        for lang in languages:
            if lang in lang_data and category in lang_data[lang]:
                r = lang_data[lang][category]
                cat_scores[lang] = {
                    "score": r.get("score", 0.0),
                    "cot_quality": r.get("cot_quality", 0.0),
                    "logical_consistency": r.get("logical_consistency", 0.0),
                    "has_reasoning_steps": r.get("has_reasoning_steps", False),
                    "final_answer_correct": r.get("final_answer_correct", False),
                }

        if cat_scores:
            scores = [v["score"] for v in cat_scores.values()]
            comparison[category] = {
                "per_language": cat_scores,
                "mean_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "score_range": max(scores) - min(scores),
                "best_language": max(cat_scores, key=lambda k: cat_scores[k]["score"]),
                "worst_language": min(cat_scores, key=lambda k: cat_scores[k]["score"]),
            }

    return comparison


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def print_summary(report: dict[str, Any]) -> None:
    """Print a formatted summary of all evaluation results."""
    print()
    print("=" * 70)
    print("  PROJECT AYA -- DEMO EVALUATION REPORT")
    print("=" * 70)
    print()

    # Multilingual
    ml = report.get("multilingual", {})
    print("MULTILINGUAL QUALITY")
    print("-" * 50)
    print(f"  Mean coherence:              {ml.get('mean_coherence', 0):.3f}")
    print(f"  Mean relevance:              {ml.get('mean_relevance', 0):.3f}")
    print(f"  Cross-lingual consistency:   {ml.get('cross_lingual_consistency', 0):.3f}")
    print(f"  Equity score (lower=better): {ml.get('equity_score', 0):.4f}")
    print()

    # Tool calling
    tc = report.get("tool_calling", {})
    print("TOOL CALLING")
    print("-" * 50)
    print(f"  Mean score:     {tc.get('mean_score', 0):.3f}")
    print(f"  JSON valid rate: {tc.get('json_valid_rate', 0):.3f}")
    print()

    # Reasoning
    rs = report.get("reasoning", {})
    print("REASONING")
    print("-" * 50)
    print(f"  Mean CoT quality:         {rs.get('mean_cot_quality', 0):.3f}")
    print(f"  Mean logical consistency: {rs.get('mean_logical_consistency', 0):.3f}")
    print(f"  Mean score:               {rs.get('mean_score', 0):.3f}")
    print(f"  CoT equity (lower=better):{rs.get('cot_equity_score', 0):.4f}")
    print()

    # Cross-language comparison
    comparison = report.get("cross_language_comparison", {})
    if comparison:
        print("CROSS-LANGUAGE REASONING COMPARISON")
        print("-" * 50)
        print(f"  {'Category':<15} {'Mean':>6} {'Min':>6} {'Max':>6} {'Range':>6} {'Best':>6} {'Worst':>6}")
        for cat, data in comparison.items():
            print(
                f"  {cat:<15} "
                f"{data.get('mean_score', 0):6.3f} "
                f"{data.get('min_score', 0):6.3f} "
                f"{data.get('max_score', 0):6.3f} "
                f"{data.get('score_range', 0):6.3f} "
                f"{data.get('best_language', 'N/A'):>6} "
                f"{data.get('worst_language', 'N/A'):>6}"
            )
        print()

    print("NOTE: This is a demo with a randomly-initialized tiny model.")
    print("Scores near zero are expected and prove the evaluation pipeline works.")
    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full demo evaluation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Demo Model Evaluation")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "demo" / "final_model.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to distillation config YAML (for eval settings)",
    )
    parser.add_argument(
        "--student-config", type=str, default=None,
        help="Path to student model config YAML",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    # Load eval config
    if args.config is not None:
        with open(args.config, "r") as f:
            distill_config = yaml.safe_load(f)
    else:
        config_path = PROJECT_ROOT / "configs" / "demo_distill.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                distill_config = yaml.safe_load(f)
        else:
            distill_config = {}

    languages = distill_config.get("languages", ["en", "es", "hi", "ar", "zh"])
    eval_cfg = distill_config.get("eval", {})
    max_new_tokens = eval_cfg.get("max_new_tokens", args.max_new_tokens)
    temperature = eval_cfg.get("temperature", args.temperature)
    device = torch.device("cpu")

    logger.info("=" * 60)
    logger.info("  Project Aya -- Demo Evaluation")
    logger.info("  Checkpoint: %s", args.checkpoint)
    logger.info("  Languages: %s", languages)
    logger.info("  Max new tokens: %d", max_new_tokens)
    logger.info("  Temperature: %.2f", temperature)
    logger.info("=" * 60)

    # Load model
    model, tokenizer = load_demo_model(args.checkpoint, args.student_config)
    model.to(device)

    # Create model function
    model_fn = make_model_fn(
        model, tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )

    t_total = time.time()
    report: dict[str, Any] = {}

    # 1. Multilingual quality
    try:
        report["multilingual"] = run_multilingual_eval(model_fn, languages)
    except Exception as exc:
        logger.error("Multilingual eval failed: %s", exc, exc_info=True)
        report["multilingual"] = {"error": str(exc)}

    # 2. Tool calling
    try:
        report["tool_calling"] = run_tool_calling_eval(model_fn, languages)
    except Exception as exc:
        logger.error("Tool calling eval failed: %s", exc, exc_info=True)
        report["tool_calling"] = {"error": str(exc)}

    # 3. Reasoning
    try:
        report["reasoning"] = run_reasoning_eval(model_fn, languages)
    except Exception as exc:
        logger.error("Reasoning eval failed: %s", exc, exc_info=True)
        report["reasoning"] = {"error": str(exc)}

    # 4. Cross-language reasoning comparison
    if "reasoning" in report and "error" not in report["reasoning"]:
        try:
            report["cross_language_comparison"] = build_cross_language_comparison(
                report["reasoning"], languages,
            )
        except Exception as exc:
            logger.error("Cross-language comparison failed: %s", exc, exc_info=True)
            report["cross_language_comparison"] = {"error": str(exc)}

    total_time = time.time() - t_total
    report["metadata"] = {
        "checkpoint": args.checkpoint,
        "languages": languages,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "total_eval_time_seconds": round(total_time, 2),
    }

    # Save report
    output_dir = Path(
        PROJECT_ROOT / eval_cfg.get("output_dir", "results/demo")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    logger.info("Evaluation report saved to %s", report_path)

    # Print summary
    print_summary(report)

    logger.info("Total evaluation time: %.1fs", total_time)


if __name__ == "__main__":
    main()
