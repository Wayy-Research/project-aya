#!/usr/bin/env python3
"""
Quick-start script for distilling on NVIDIA ClimbMix.

Usage:
    # Pretokenized mode (GPT-2 vocab, fastest):
    python scripts/run_climbmix_distill.py --mode pretokenized

    # Retokenized mode (Aya vocab, for multilingual student):
    python scripts/run_climbmix_distill.py --mode retokenize

    # With custom config:
    python scripts/run_climbmix_distill.py --config configs/distill_climbmix.yaml

    # Dry run (just load data and print shapes):
    python scripts/run_climbmix_distill.py --dry-run

Dataset: https://huggingface.co/datasets/nvidia/ClimbMix
Paper:   https://arxiv.org/abs/2504.13161
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aya_distill.distill.climbmix import ClimbMixConfig, create_climbmix_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill on NVIDIA ClimbMix dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: use CLI args)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pretokenized",
        choices=["pretokenized", "retokenize"],
        help="Token mode: pretokenized (GPT-2) or retokenize (Aya vocab)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="CohereForAI/aya-expanse-8b",
        help="Teacher model name",
    )
    parser.add_argument(
        "--student-config",
        type=str,
        default="configs/student.yaml",
        help="Student model config path",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=20000,
        help="Total training steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/climbmix_distill",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just load data and print shapes, don't train",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dry_run(config: ClimbMixConfig, tokenizer: object | None) -> None:
    """Load a few batches and print shapes for verification."""
    logger.info("=== DRY RUN: Loading ClimbMix data ===")
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Max seq len: {config.max_seq_len}")
    logger.info(f"Batch size: {config.batch_size}")

    loader = create_climbmix_dataloader(config, tokenizer=tokenizer)

    for i, batch in enumerate(loader):
        logger.info(
            f"Batch {i}: "
            f"input_ids={batch['input_ids'].shape}, "
            f"labels={batch['labels'].shape}, "
            f"dtype={batch['input_ids'].dtype}"
        )
        # Show token range to verify vocab
        ids = batch["input_ids"]
        logger.info(
            f"  Token range: [{ids.min().item()}, {ids.max().item()}]"
        )
        logger.info(
            f"  Non-pad tokens: {(ids != 0).sum().item()}/{ids.numel()}"
        )

        if i >= 4:
            break

    logger.info("=== DRY RUN COMPLETE ===")


def run_distillation(
    args: argparse.Namespace,
    data_config: ClimbMixConfig,
    tokenizer: object | None,
) -> None:
    """Run KL distillation with ClimbMix data."""
    from aya_distill.distill.kl_distillation import KLDistillationConfig, KLDistillationTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load teacher
    logger.info(f"Loading teacher: {args.teacher}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()

    # Load student
    logger.info(f"Loading student config: {args.student_config}")
    student_config_path = Path(args.student_config)
    if not student_config_path.exists():
        logger.error(f"Student config not found: {student_config_path}")
        sys.exit(1)

    # Import Aetheris model
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent.parent / "aetheris")
    )
    try:
        from aetheris import AetherisConfig, HybridMambaMoE

        with open(student_config_path) as f:
            student_yaml = yaml.safe_load(f)
        student_cfg = AetherisConfig(**student_yaml)
        student = HybridMambaMoE(student_cfg).to(device)
        logger.info(
            f"Student loaded: {sum(p.numel() for p in student.parameters()):,} params"
        )
    except ImportError:
        logger.error(
            "Could not import Aetheris. Make sure the aetheris package "
            "is available at ../aetheris/ or installed."
        )
        sys.exit(1)

    # Create data loader
    loader = create_climbmix_dataloader(data_config, tokenizer=tokenizer)

    # Set up KL distillation
    kl_config = KLDistillationConfig(
        temperature=2.0,
        alpha=0.7,
        lr=5e-5,
        warmup_steps=500,
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    trainer = KLDistillationTrainer(
        teacher=teacher,
        student=student,
        config=kl_config,
        device=device,
    )

    logger.info("Starting ClimbMix KL distillation...")
    results = trainer.train(loader)
    logger.info(f"Training complete. Final step: {results.get('final_step')}")


def main() -> None:
    args = parse_args()

    # Build ClimbMix config
    data_config = ClimbMixConfig(
        mode=args.mode,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Load tokenizer if retokenize mode
    tokenizer = None
    if args.mode == "retokenize":
        from aya_distill.distill.data import get_shared_tokenizer

        logger.info("Loading Aya tokenizer for retokenization...")
        tokenizer = get_shared_tokenizer()
        data_config.target_tokenizer_name = "CohereForAI/aya-expanse-8b"

    if args.dry_run:
        dry_run(data_config, tokenizer)
        return

    run_distillation(args, data_config, tokenizer)


if __name__ == "__main__":
    main()
