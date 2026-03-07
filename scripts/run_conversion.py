#!/usr/bin/env python3
"""
Convert Cohere Aya transformer to Aetheris Mamba-MoE.

Usage:
    # Direct weight mapping (fast, no data needed):
    python scripts/run_conversion.py \
      --strategy weight_map \
      --output checkpoints/converted/

    # Hybrid (weight map + refinement):
    python scripts/run_conversion.py \
      --strategy hybrid \
      --refinement-steps 1000 \
      --output checkpoints/converted/

    # With custom config:
    python scripts/run_conversion.py --config configs/conversion.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_conversion")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Aya → Aetheris")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--teacher", default="CohereForAI/aya-expanse-8b")
    p.add_argument("--student-config", default="configs/student.yaml")
    p.add_argument("--strategy", default="weight_map",
                    choices=["weight_map", "distillation", "hybrid"])
    p.add_argument("--ffn-to-moe", default="replicate",
                    choices=["replicate", "svd_split", "random"])
    p.add_argument("--a-init", default="exponential_decay",
                    choices=["random", "constant", "exponential_decay"])
    p.add_argument("--refinement-steps", type=int, default=1000)
    p.add_argument("--refinement-lr", type=float, default=1e-4)
    p.add_argument("--output", default="checkpoints/converted/")
    p.add_argument("--teacher-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from distill.converter import ConversionConfig
    from distill.block_surgery import convert_model

    config = ConversionConfig(
        strategy=args.strategy,
        a_init=args.a_init,
        ffn_to_moe=args.ffn_to_moe,
        refinement_steps=args.refinement_steps,
        refinement_lr=args.refinement_lr,
        seed=args.seed,
    )

    logger.info("Starting model conversion:")
    logger.info("  Teacher:  %s", args.teacher)
    logger.info("  Student:  %s", args.student_config)
    logger.info("  Strategy: %s", args.strategy)
    logger.info("  FFN→MoE:  %s", args.ffn_to_moe)
    logger.info("  A init:   %s", args.a_init)
    logger.info("  Output:   %s", args.output)

    student, report = convert_model(
        teacher_name=args.teacher,
        student_config_path=args.student_config,
        strategy=args.strategy,
        conversion_config=config,
        device="cuda",
        teacher_dtype=args.teacher_dtype,
        output_dir=args.output,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Conversion Complete")
    print(f"{'='*60}")
    print(f"  Teacher layers:  {report['teacher_layers']}")
    print(f"  Student layers:  {report['student_layers']}")
    print(f"  Student params:  {report['student_params']:,}")
    if "mean_cka" in report:
        print(f"  Mean CKA:        {report['mean_cka']:.4f}")
        print(f"  Min CKA:         {report['min_cka']:.4f}")
        if report.get("low_cka_layers"):
            print(f"  Low CKA layers:  {report['low_cka_layers']}")
    print(f"  Output:          {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
