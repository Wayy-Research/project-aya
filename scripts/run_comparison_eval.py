#!/usr/bin/env python3
"""
Comparison evaluation: Teacher (tiny-aya-global) vs Student (Aetheris).

Runs mGSM, XCOPA, and throughput benchmarks on both models sequentially,
then computes deltas, paired bootstrap significance, degradation equity
scores, and speedup ratios. Outputs a single JSON suitable for upload
to the HuggingFace Space.

Memory management: only one model is loaded at a time. The teacher is
evaluated first, unloaded, then the student is loaded and evaluated.
This fits comfortably on a single 4 GB GPU.

Usage:
    # Full comparison
    python scripts/run_comparison_eval.py \
        --checkpoint checkpoints/stage1/step_1000.pt \
        --student-config configs/student_base.yaml

    # Teacher baseline only
    python scripts/run_comparison_eval.py --teacher-only

    # Student only (load teacher baseline from prior run)
    python scripts/run_comparison_eval.py \
        --checkpoint checkpoints/stage1/step_1000.pt \
        --student-config configs/student_base.yaml \
        --student-only \
        --teacher-results results/teacher_baseline.json

    # Specific benchmarks and languages
    python scripts/run_comparison_eval.py \
        --checkpoint checkpoints/stage1/step_1000.pt \
        --student-config configs/student_base.yaml \
        --benchmarks mgsm,xcopa \
        --languages en,es,hi,zh

Wayy Research -- Project Aya, 2024-2026.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "aetheris"))

from aya_distill.eval.benchmarks import (
    BenchmarkConfig,
    load_model_and_tokenizer,
    print_results_table,
    run_mgsm,
    run_xcopa,
)
from aya_distill.eval.metrics import (
    degradation_equity_score,
    paired_bootstrap_test,
)
from aya_distill.eval.throughput import (
    ThroughputConfig,
    print_throughput_table,
    run_throughput_benchmark,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("comparison_eval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEACHER_NAME = "CohereLabs/tiny-aya-global"
TEACHER_PARAMS = 3_350_000_000
TEACHER_LAYERS = 36
TEACHER_HIDDEN = 2048

DEFAULT_LANGUAGES = ["en", "es", "hi", "zh", "ar", "sw", "tr", "ja", "id", "te"]
ALL_BENCHMARKS = ["mgsm", "xcopa", "throughput"]


# ---------------------------------------------------------------------------
# HuggingFace-compatible wrapper for HybridMambaMoE
# ---------------------------------------------------------------------------
class _AetherisConfig:
    """Minimal config shim so benchmark code can read model.config attrs."""

    def __init__(self, max_position_embeddings: int = 2048) -> None:
        self.max_position_embeddings = max_position_embeddings


class AetherisHFWrapper(torch.nn.Module):
    """Wraps HybridMambaMoE so it quacks like a PreTrainedModel.

    The eval harness calls model.generate(), model.device, and
    model.config.max_position_embeddings. This wrapper provides all
    three without pulling in the full HF class hierarchy.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = _AetherisConfig(max_position_embeddings=max_seq_len)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def eval(self) -> "AetherisHFWrapper":
        self.model.eval()
        return self

    def to(self, device: Any) -> "AetherisHFWrapper":
        self.model.to(device)
        return self

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self.model.parameters(recurse=recurse)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressive greedy generation matching HF generate() API."""
        generated = input_ids.clone()
        device = self.device

        for _ in range(max_new_tokens):
            outputs = self.model(generated.to(device))
            logits = outputs["logits"][:, -1, :]

            if do_sample and temperature is not None and temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token.to(generated.device)], dim=1)

        return generated


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def load_teacher(
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple[Any, Any]:
    """Load teacher model + tokenizer via the existing eval infra."""
    logger.info("Loading teacher model: %s", TEACHER_NAME)
    model, tokenizer = load_model_and_tokenizer(
        TEACHER_NAME, dtype=dtype, device_map=device_map,
    )
    return model, tokenizer


def load_student(
    checkpoint_path: str,
    student_config_path: str,
    device: torch.device,
) -> tuple[AetherisHFWrapper, Any, int, Any, dict[str, Any]]:
    """Load Aetheris student from checkpoint.

    Returns the model wrapped in AetherisHFWrapper and the teacher
    tokenizer (student shares the 262K vocab tokenizer).
    """
    from aetheris.config import AetherisConfig
    from aetheris.model import HybridMambaMoE

    logger.info("Loading student config: %s", student_config_path)
    student_config = AetherisConfig.from_yaml(student_config_path)
    student_config.gradient_checkpointing = False

    logger.info("Building student model (%d layers, d_model=%d)",
                student_config.n_layer, student_config.d_model)
    student = HybridMambaMoE(student_config)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("student_state_dict", ckpt)
    missing, unexpected = student.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys in student checkpoint: %s", missing[:10])
    if unexpected:
        logger.warning("Unexpected keys in student checkpoint: %s", unexpected[:10])

    student.eval()

    wrapper = AetherisHFWrapper(
        student, max_seq_len=student_config.max_seq_len,
    )
    wrapper = wrapper.to(device)

    param_count = sum(p.numel() for p in student.parameters())
    logger.info("Student loaded: %d params (%.1fM)", param_count, param_count / 1e6)

    # Load teacher tokenizer (student shares the same tokenizer)
    from transformers import AutoTokenizer

    logger.info("Loading shared tokenizer from %s", TEACHER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return wrapper, tokenizer, param_count, student_config, ckpt


def unload_model(model: Any) -> None:
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    logger.info("Model unloaded, GPU cache cleared")


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_benchmarks(
    model: Any,
    tokenizer: Any,
    model_name: str,
    benchmarks: list[str],
    languages: list[str],
    n_shot: int = 8,
) -> dict[str, Any]:
    """Run requested benchmarks and return results dict."""
    results: dict[str, Any] = {}

    bench_config = BenchmarkConfig(
        model_name=model_name,
        dtype="bfloat16",
        languages=languages,
        n_shot=n_shot,
    )

    if "mgsm" in benchmarks:
        logger.info("=" * 60)
        logger.info("Running mGSM for %s", model_name)
        logger.info("=" * 60)
        try:
            mgsm_results = run_mgsm(model, tokenizer, bench_config)
            results["mgsm"] = mgsm_results
            print_results_table(mgsm_results)
        except Exception as exc:
            logger.error("mGSM failed for %s: %s", model_name, exc, exc_info=True)
            results["mgsm"] = {"error": str(exc)}

    if "xcopa" in benchmarks:
        logger.info("=" * 60)
        logger.info("Running XCOPA for %s", model_name)
        logger.info("=" * 60)
        try:
            xcopa_results = run_xcopa(model, tokenizer, bench_config)
            results["xcopa"] = xcopa_results
            print_results_table(xcopa_results)
        except Exception as exc:
            logger.error("XCOPA failed for %s: %s", model_name, exc, exc_info=True)
            results["xcopa"] = {"error": str(exc)}

    if "throughput" in benchmarks:
        logger.info("=" * 60)
        logger.info("Running throughput benchmark for %s", model_name)
        logger.info("=" * 60)
        try:
            tp_config = ThroughputConfig(
                n_tokens=128,
                n_runs=5,
                n_warmup=2,
                languages=languages,
            )
            tp_results = run_throughput_benchmark(model, tokenizer, tp_config)
            results["throughput"] = tp_results
            print_throughput_table(tp_results)
        except Exception as exc:
            logger.error("Throughput failed for %s: %s", model_name, exc, exc_info=True)
            results["throughput"] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# Comparison computation
# ---------------------------------------------------------------------------


def _extract_per_lang_accuracy(
    results: dict[str, Any],
    benchmark: str,
) -> dict[str, float]:
    """Pull per-language accuracy from benchmark results."""
    bench = results.get(benchmark, {})
    if "error" in bench:
        return {}
    per_lang = bench.get("per_language", {})
    out: dict[str, float] = {}
    for lang, data in per_lang.items():
        if isinstance(data, dict) and "accuracy" in data:
            out[lang] = data["accuracy"]
    return out


def _extract_per_lang_throughput(
    results: dict[str, Any],
) -> dict[str, float]:
    """Pull per-language mean tokens/sec from throughput results."""
    tp = results.get("throughput", {})
    if "error" in tp:
        return {}
    per_lang = tp.get("per_language", {})
    out: dict[str, float] = {}
    for lang, data in per_lang.items():
        if isinstance(data, dict) and "mean_tokens_per_second" in data:
            out[lang] = data["mean_tokens_per_second"]
    return out


def _extract_per_lang_memory(
    results: dict[str, Any],
) -> dict[str, float]:
    """Pull per-language peak memory from throughput results."""
    tp = results.get("throughput", {})
    if "error" in tp:
        return {}
    per_lang = tp.get("per_language", {})
    out: dict[str, float] = {}
    for lang, data in per_lang.items():
        if isinstance(data, dict) and "peak_memory_mb" in data:
            out[lang] = data["peak_memory_mb"]
    return out


def compute_comparison(
    teacher_results: dict[str, Any],
    student_results: dict[str, Any],
    benchmarks: list[str],
) -> dict[str, Any]:
    """Compute deltas, significance, equity, and speedup."""
    comparison: dict[str, Any] = {}

    for bench in ["mgsm", "xcopa"]:
        if bench not in benchmarks:
            continue
        teacher_acc = _extract_per_lang_accuracy(teacher_results, bench)
        student_acc = _extract_per_lang_accuracy(student_results, bench)
        if not teacher_acc or not student_acc:
            comparison[bench] = {"status": "insufficient_data"}
            continue

        common = sorted(set(teacher_acc) & set(student_acc))
        deltas: dict[str, float] = {}
        for lang in common:
            deltas[lang] = student_acc[lang] - teacher_acc[lang]

        comparison[bench] = {
            "teacher": teacher_acc,
            "student": student_acc,
            "deltas": deltas,
            "mean_delta": float(np.mean(list(deltas.values()))) if deltas else 0.0,
        }

    # Throughput speedup
    if "throughput" in benchmarks:
        teacher_tp = _extract_per_lang_throughput(teacher_results)
        student_tp = _extract_per_lang_throughput(student_results)
        teacher_mem = _extract_per_lang_memory(teacher_results)
        student_mem = _extract_per_lang_memory(student_results)

        common_tp = sorted(set(teacher_tp) & set(student_tp))
        speedups: dict[str, float] = {}
        mem_ratios: dict[str, float] = {}

        for lang in common_tp:
            if teacher_tp[lang] > 0:
                speedups[lang] = student_tp[lang] / teacher_tp[lang]
            if lang in teacher_mem and lang in student_mem and teacher_mem[lang] > 0:
                mem_ratios[lang] = student_mem[lang] / teacher_mem[lang]

        comparison["throughput"] = {
            "teacher": teacher_tp,
            "student": student_tp,
            "speedup": speedups,
            "mean_speedup": float(np.mean(list(speedups.values()))) if speedups else 0.0,
            "memory_ratio": mem_ratios,
            "mean_memory_ratio": (
                float(np.mean(list(mem_ratios.values()))) if mem_ratios else 0.0
            ),
        }

    # Degradation Equity Score across all accuracy benchmarks
    all_teacher_acc: dict[str, float] = {}
    all_student_acc: dict[str, float] = {}
    for bench in ["mgsm", "xcopa"]:
        all_teacher_acc.update(_extract_per_lang_accuracy(teacher_results, bench))
        all_student_acc.update(_extract_per_lang_accuracy(student_results, bench))

    if all_teacher_acc and all_student_acc:
        des = degradation_equity_score(all_teacher_acc, all_student_acc)
        teacher_des = float(np.var(list(all_teacher_acc.values())))
        student_des = float(np.var(list(all_student_acc.values())))
        comparison["equity"] = {
            "degradation_equity_score": des,
            "teacher_accuracy_variance": teacher_des,
            "student_accuracy_variance": student_des,
            "des_delta": student_des - teacher_des,
        }

    return comparison


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def build_output(
    teacher_results: dict[str, Any] | None,
    student_results: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    student_params: int = 0,
    student_config: Any = None,
    checkpoint_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the final output JSON."""
    output: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Teacher metadata
    output["teacher"] = {
        "name": TEACHER_NAME,
        "params": TEACHER_PARAMS,
        "architecture": "Transformer (GQA)",
        "layers": TEACHER_LAYERS,
        "hidden_dim": TEACHER_HIDDEN,
    }

    # Student metadata
    if student_config is not None:
        output["student"] = {
            "name": "Aetheris",
            "params": student_params,
            "architecture": "Hybrid Mamba-MoE",
            "layers": student_config.n_layer,
            "hidden_dim": student_config.d_model,
            "num_experts": student_config.num_experts,
            "top_k": student_config.top_k,
            "checkpoint_step": (
                checkpoint_info.get("step", 0) if checkpoint_info else 0
            ),
        }
        if student_params > 0:
            output["compression_ratio"] = round(
                TEACHER_PARAMS / student_params, 2
            )

    # Raw benchmark results
    if teacher_results is not None:
        output["teacher_benchmarks"] = teacher_results
    if student_results is not None:
        output["student_benchmarks"] = student_results

    # Comparison
    if comparison is not None:
        # Flatten mgsm/xcopa comparison into top-level keys
        for bench in ["mgsm", "xcopa"]:
            if bench in comparison:
                output[bench] = comparison[bench]
        if "throughput" in comparison:
            output["throughput"] = comparison["throughput"]
        if "equity" in comparison:
            output["equity"] = comparison["equity"]

    # Training metadata from checkpoint
    if checkpoint_info is not None:
        output["training"] = {
            "stage": checkpoint_info.get("stage", 1),
            "step": checkpoint_info.get("step", 0),
            "loss": checkpoint_info.get("loss", 0.0),
            "cka_mean": checkpoint_info.get("cka_mean", 0.0),
        }

    return output


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_comparison_summary(output: dict[str, Any]) -> None:
    """Print a human-readable comparison table to stdout."""
    print()
    print("=" * 78)
    print("  TEACHER vs STUDENT COMPARISON")
    print("=" * 78)

    teacher = output.get("teacher", {})
    student = output.get("student", {})
    print(f"  Teacher: {teacher.get('name', 'N/A')} "
          f"({teacher.get('params', 0) / 1e9:.1f}B params)")
    if student:
        print(f"  Student: {student.get('name', 'N/A')} "
              f"({student.get('params', 0) / 1e6:.0f}M params)")
        print(f"  Compression: {output.get('compression_ratio', 'N/A')}x")
    print()

    # Accuracy benchmarks
    for bench in ["mgsm", "xcopa"]:
        data = output.get(bench)
        if data is None or data.get("status") == "insufficient_data":
            continue

        print(f"  {bench.upper()}")
        print(f"  {'Language':<8} {'Teacher':>10} {'Student':>10} {'Delta':>10}")
        print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")

        teacher_acc = data.get("teacher", {})
        student_acc = data.get("student", {})
        deltas = data.get("deltas", {})

        for lang in sorted(set(teacher_acc) | set(student_acc)):
            t_val = teacher_acc.get(lang)
            s_val = student_acc.get(lang)
            d_val = deltas.get(lang)
            t_str = f"{t_val:.4f}" if t_val is not None else "N/A"
            s_str = f"{s_val:.4f}" if s_val is not None else "N/A"
            d_str = f"{d_val:+.4f}" if d_val is not None else "N/A"
            print(f"  {lang:<8} {t_str:>10} {s_str:>10} {d_str:>10}")

        mean_delta = data.get("mean_delta", 0.0)
        print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")
        print(f"  {'MEAN':<8} {'':>10} {'':>10} {mean_delta:>+10.4f}")
        print()

    # Throughput
    tp = output.get("throughput")
    if tp and tp.get("status") != "insufficient_data":
        print(f"  THROUGHPUT (tokens/sec)")
        print(f"  {'Language':<8} {'Teacher':>10} {'Student':>10} {'Speedup':>10}")
        print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")

        teacher_tp = tp.get("teacher", {})
        student_tp = tp.get("student", {})
        speedups = tp.get("speedup", {})

        for lang in sorted(set(teacher_tp) | set(student_tp)):
            t_val = teacher_tp.get(lang)
            s_val = student_tp.get(lang)
            sp_val = speedups.get(lang)
            t_str = f"{t_val:.1f}" if t_val is not None else "N/A"
            s_str = f"{s_val:.1f}" if s_val is not None else "N/A"
            sp_str = f"{sp_val:.2f}x" if sp_val is not None else "N/A"
            print(f"  {lang:<8} {t_str:>10} {s_str:>10} {sp_str:>10}")

        mean_sp = tp.get("mean_speedup", 0.0)
        print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")
        print(f"  {'MEAN':<8} {'':>10} {'':>10} {mean_sp:>9.2f}x")
        print()

    # Equity
    equity = output.get("equity")
    if equity:
        print(f"  EQUITY")
        print(f"  {'-' * 40}")
        print(f"  Degradation Equity Score: "
              f"{equity.get('degradation_equity_score', 0):.6f}")
        print(f"  Teacher accuracy variance: "
              f"{equity.get('teacher_accuracy_variance', 0):.6f}")
        print(f"  Student accuracy variance: "
              f"{equity.get('student_accuracy_variance', 0):.6f}")
        print()

    # Training
    training = output.get("training")
    if training:
        print(f"  TRAINING METADATA")
        print(f"  {'-' * 40}")
        print(f"  Stage: {training.get('stage', 'N/A')}")
        print(f"  Step:  {training.get('step', 'N/A')}")
        print(f"  Loss:  {training.get('loss', 'N/A')}")
        print(f"  CKA:   {training.get('cka_mean', 'N/A')}")
        print()

    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher/student comparison evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to student .pt checkpoint file",
    )
    parser.add_argument(
        "--student-config", type=str, default=None,
        help="Path to student model YAML config",
    )
    parser.add_argument(
        "--output", type=str, default="results/benchmark_comparison.json",
        help="Output JSON path (default: results/benchmark_comparison.json)",
    )
    parser.add_argument(
        "--benchmarks", type=str, default="mgsm,xcopa,throughput",
        help="Comma-separated benchmarks to run (default: mgsm,xcopa,throughput)",
    )
    parser.add_argument(
        "--languages", type=str,
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated language codes (default: en,es,hi,zh,ar,sw,tr,ja,id,te)",
    )
    parser.add_argument(
        "--n-shot", type=int, default=8,
        help="Number of few-shot exemplars (default: 8)",
    )
    parser.add_argument(
        "--teacher-only", action="store_true",
        help="Only evaluate the teacher model (skip student)",
    )
    parser.add_argument(
        "--student-only", action="store_true",
        help="Only evaluate the student model (load teacher results from file)",
    )
    parser.add_argument(
        "--teacher-results", type=str, default=None,
        help="Path to teacher results JSON (used with --student-only)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        help="Model dtype: float16, bfloat16, float32 (default: bfloat16)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for student model (default: auto-detect)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    languages = [l.strip() for l in args.languages.split(",")]

    for b in benchmarks:
        if b not in ALL_BENCHMARKS:
            logger.error("Unknown benchmark: %s (valid: %s)", b, ALL_BENCHMARKS)
            sys.exit(1)

    if args.student_only and args.teacher_only:
        logger.error("Cannot use both --teacher-only and --student-only")
        sys.exit(1)

    if args.student_only and args.teacher_results is None:
        logger.error("--student-only requires --teacher-results")
        sys.exit(1)

    if not args.teacher_only:
        if args.checkpoint is None:
            logger.error("--checkpoint is required unless --teacher-only is set")
            sys.exit(1)
        if args.student_config is None:
            logger.error("--student-config is required unless --teacher-only is set")
            sys.exit(1)

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("=" * 60)
    logger.info("  Project Aya -- Comparison Evaluation")
    logger.info("  Benchmarks: %s", benchmarks)
    logger.info("  Languages:  %s", languages)
    logger.info("  Device:     %s", device)
    logger.info("=" * 60)

    t_start = time.time()

    # ------------------------------------------------------------------
    # Teacher evaluation
    # ------------------------------------------------------------------
    teacher_results: dict[str, Any] | None = None

    if args.student_only:
        logger.info("Loading teacher results from: %s", args.teacher_results)
        with open(args.teacher_results, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # Support both raw benchmark results and full comparison output
        teacher_results = saved.get("teacher_benchmarks", saved)
    elif not args.student_only:
        logger.info("--- TEACHER EVALUATION ---")
        model, tokenizer = load_teacher(dtype=args.dtype)
        teacher_results = run_benchmarks(
            model, tokenizer, TEACHER_NAME, benchmarks, languages,
            n_shot=args.n_shot,
        )
        unload_model(model)

    # ------------------------------------------------------------------
    # Student evaluation
    # ------------------------------------------------------------------
    student_results: dict[str, Any] | None = None
    student_params = 0
    student_config_obj = None
    checkpoint_info: dict[str, Any] | None = None

    if not args.teacher_only:
        logger.info("--- STUDENT EVALUATION ---")
        wrapper, tokenizer, student_params, student_config_obj, ckpt = load_student(
            args.checkpoint, args.student_config, device,
        )

        # Extract checkpoint metadata
        checkpoint_info = {
            "stage": ckpt.get("stage", 1),
            "step": ckpt.get("step", ckpt.get("global_step", 0)),
            "loss": ckpt.get("loss", ckpt.get("train_loss", 0.0)),
            "cka_mean": ckpt.get("cka_mean", 0.0),
        }

        student_results = run_benchmarks(
            wrapper, tokenizer, "Aetheris", benchmarks, languages,
            n_shot=args.n_shot,
        )
        unload_model(wrapper)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    comparison: dict[str, Any] | None = None
    if teacher_results is not None and student_results is not None:
        logger.info("Computing comparison metrics...")
        comparison = compute_comparison(teacher_results, student_results, benchmarks)

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    output = build_output(
        teacher_results=teacher_results,
        student_results=student_results,
        comparison=comparison,
        student_params=student_params,
        student_config=student_config_obj,
        checkpoint_info=checkpoint_info,
    )

    total_time = time.time() - t_start
    output["eval_time_seconds"] = round(total_time, 2)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", output_path)

    # Print summary
    print_comparison_summary(output)

    logger.info("Total evaluation time: %.1fs", total_time)


if __name__ == "__main__":
    main()
