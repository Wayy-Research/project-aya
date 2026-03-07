#!/usr/bin/env python3
"""
Run baseline teacher evaluation for Project Aya.

Loads configuration, runs mGSM + XCOPA benchmarks and throughput
measurement against the teacher model, validates results against
the Cohere multilingual checklist, and saves everything to disk.

Usage::

    python scripts/run_baseline.py
    python scripts/run_baseline.py --config configs/eval_baseline.yaml
    python scripts/run_baseline.py --benchmarks mgsm xcopa
    python scripts/run_baseline.py --languages en es hi

Wayy Research -- Project Aya
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmarks import (
    BenchmarkConfig,
    load_model_and_tokenizer,
    print_results_table,
    run_mgsm,
    run_xcopa,
    save_results,
)
from eval.checklist import validate_results
from eval.throughput import (
    ThroughputConfig,
    print_throughput_table,
    run_throughput_benchmark,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load evaluation config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def config_to_benchmark(
    raw: dict[str, Any],
    benchmark: str,
    overrides: dict[str, Any] | None = None,
) -> BenchmarkConfig:
    """Create a BenchmarkConfig from raw YAML config + overrides."""
    model_cfg = raw.get("model", {})
    bench_cfg = raw.get("benchmarks", {}).get(benchmark, {})
    eval_cfg = raw.get("evaluation", {})

    kwargs: dict[str, Any] = {
        "model_name": model_cfg.get("name", "CohereForAI/aya-expanse-8b"),
        "dtype": model_cfg.get("dtype", "float16"),
        "device_map": model_cfg.get("device_map", "auto"),
        "n_shot": bench_cfg.get("n_shot", 8 if benchmark == "mgsm" else 4),
        "languages": bench_cfg.get("languages", []),
        "max_new_tokens": bench_cfg.get("max_new_tokens", 256),
        "n_bootstrap": eval_cfg.get("n_bootstrap", 1000),
        "seed": eval_cfg.get("seed", 42),
        "output_dir": raw.get("output_dir", "results/baseline"),
    }

    if overrides:
        if "languages" in overrides and overrides["languages"]:
            kwargs["languages"] = overrides["languages"]
        if "seed" in overrides:
            kwargs["seed"] = overrides["seed"]

    return BenchmarkConfig(**kwargs)


def config_to_throughput(
    raw: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> ThroughputConfig:
    """Create a ThroughputConfig from raw YAML config."""
    bench_cfg = raw.get("benchmarks", {}).get("throughput", {})
    eval_cfg = raw.get("evaluation", {})

    kwargs: dict[str, Any] = {
        "n_tokens": bench_cfg.get("n_tokens", 512),
        "n_runs": bench_cfg.get("n_runs", 10),
        "n_warmup": bench_cfg.get("n_warmup", 2),
        "languages": bench_cfg.get("languages", []),
        "seed": eval_cfg.get("seed", 42),
    }

    if overrides:
        if "languages" in overrides and overrides["languages"]:
            kwargs["languages"] = overrides["languages"]

    return ThroughputConfig(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline teacher evaluation for Project Aya"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "eval_baseline.yaml"),
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mgsm", "xcopa", "throughput"],
        choices=["mgsm", "xcopa", "throughput"],
        help="Which benchmarks to run",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Override languages to evaluate (ISO 639-1 codes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    logger.info("Loading config from: %s", args.config)
    raw_config = load_config(args.config)

    if args.output_dir:
        raw_config["output_dir"] = args.output_dir
    if args.seed is not None:
        raw_config.setdefault("evaluation", {})["seed"] = args.seed

    output_dir = Path(raw_config.get("output_dir", "results/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = {
        "languages": args.languages,
        "seed": args.seed,
    }

    # Set global seeds
    seed = raw_config.get("evaluation", {}).get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model (shared across benchmarks)
    model_name = raw_config.get("model", {}).get("name", "CohereForAI/aya-expanse-8b")
    dtype = raw_config.get("model", {}).get("dtype", "float16")
    device_map = raw_config.get("model", {}).get("device_map", "auto")

    logger.info("Loading teacher model: %s", model_name)
    start_load = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(model_name, dtype, device_map)
    load_time = time.perf_counter() - start_load
    logger.info("Model loaded in %.2f seconds", load_time)

    all_results: dict[str, Any] = {
        "run_metadata": {
            "config_path": str(args.config),
            "model_name": model_name,
            "dtype": dtype,
            "device_map": device_map,
            "model_load_time_sec": load_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "N/A"
            ),
            "benchmarks_requested": args.benchmarks,
        }
    }

    checklist_reports: list[dict[str, Any]] = []

    # --- Run mGSM ---
    if "mgsm" in args.benchmarks:
        logger.info("=" * 60)
        logger.info("Running mGSM benchmark")
        logger.info("=" * 60)

        mgsm_config = config_to_benchmark(raw_config, "mgsm", overrides)
        start = time.perf_counter()
        mgsm_results = run_mgsm(model, tokenizer, mgsm_config)
        elapsed = time.perf_counter() - start
        mgsm_results["elapsed_sec"] = elapsed

        print_results_table(mgsm_results)
        save_results(mgsm_results, output_dir, "mgsm")
        all_results["mgsm"] = mgsm_results

        # Checklist validation
        report = validate_results(mgsm_results)
        report.print_report()
        checklist_reports.append({"benchmark": "mgsm", **report.to_dict()})

    # --- Run XCOPA ---
    if "xcopa" in args.benchmarks:
        logger.info("=" * 60)
        logger.info("Running XCOPA benchmark")
        logger.info("=" * 60)

        xcopa_config = config_to_benchmark(raw_config, "xcopa", overrides)
        start = time.perf_counter()
        xcopa_results = run_xcopa(model, tokenizer, xcopa_config)
        elapsed = time.perf_counter() - start
        xcopa_results["elapsed_sec"] = elapsed

        print_results_table(xcopa_results)
        save_results(xcopa_results, output_dir, "xcopa")
        all_results["xcopa"] = xcopa_results

        # Checklist validation
        report = validate_results(xcopa_results)
        report.print_report()
        checklist_reports.append({"benchmark": "xcopa", **report.to_dict()})

    # --- Run Throughput ---
    if "throughput" in args.benchmarks:
        logger.info("=" * 60)
        logger.info("Running throughput benchmark")
        logger.info("=" * 60)

        tp_config = config_to_throughput(raw_config, overrides)
        start = time.perf_counter()
        tp_results = run_throughput_benchmark(model, tokenizer, tp_config)
        elapsed = time.perf_counter() - start
        tp_results["elapsed_sec"] = elapsed

        print_throughput_table(tp_results)
        save_results(tp_results, output_dir, "throughput")
        all_results["throughput"] = tp_results

        # Checklist validation (relaxed for throughput)
        report = validate_results(tp_results, include_throughput=True)
        report.print_report()
        checklist_reports.append({"benchmark": "throughput", **report.to_dict()})

    # --- Save combined results ---
    all_results["checklist_reports"] = checklist_reports

    combined_path = output_dir / "baseline_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Combined results saved to: %s", combined_path)

    # --- Summary ---
    print("\n" + "=" * 72)
    print("  BASELINE EVALUATION COMPLETE")
    print("=" * 72)
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")
    print(f"  Benchmarks: {', '.join(args.benchmarks)}")

    all_compliant = all(
        r.get("compliant", False) for r in checklist_reports
    )
    if all_compliant:
        print("  Checklist: ALL COMPLIANT")
    else:
        failed = [
            r["benchmark"] for r in checklist_reports
            if not r.get("compliant", False)
        ]
        print(f"  Checklist: NON-COMPLIANT ({', '.join(failed)})")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
