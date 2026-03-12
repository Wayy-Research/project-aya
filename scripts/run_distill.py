#!/usr/bin/env python3
"""
Main distillation entry point for Project Aya.

3-stage MambaInLlama pipeline:
    Stage 1: Layer Alignment (CKA-guided)
    Stage 2: KL Distillation (temperature-scaled)
    Stage 3: Supervised Fine-Tuning (multilingual recovery)

Usage:
    python scripts/run_distill.py --stage 1 --config configs/distill_stage1.yaml
    python scripts/run_distill.py --stage 2 --config configs/distill_stage2.yaml
    python scripts/run_distill.py --stage 3 --config configs/distill_stage3.yaml

Wayy Research, 2024-2026.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("distill")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {path}")
    return config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_teacher(
    teacher_cfg: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load the Aya teacher model from HuggingFace."""
    from transformers import AutoModelForCausalLM

    name = teacher_cfg["name"]
    dtype_str = teacher_cfg.get("dtype", "float16")
    device_map = teacher_cfg.get("device_map", "auto")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    logger.info(f"Loading teacher: {name} (dtype={dtype_str})")
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    logger.info(
        f"Teacher loaded: "
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params"
    )
    return model


def load_student(
    student_cfg: dict[str, Any],
    device: torch.device,
    vocab_size_override: int | None = None,
) -> torch.nn.Module:
    """
    Load or initialize the Aetheris student model.

    If a checkpoint path is provided, loads weights from that checkpoint.
    Otherwise initializes from the config YAML.
    """
    # Import Aetheris components
    # Add aetheris to path if needed
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aetheris"))
    from aetheris.config import AetherisConfig
    from aetheris.model import HybridMambaMoE

    config_path = student_cfg.get("config_path", "configs/student.yaml")
    student_config = AetherisConfig.from_yaml(config_path)

    # Override vocab size to match teacher tokenizer
    if vocab_size_override is not None:
        student_config.vocab_size = vocab_size_override

    logger.info(
        f"Initializing student: d_model={student_config.d_model}, "
        f"n_layer={student_config.n_layer}, "
        f"vocab_size={student_config.vocab_size}"
    )

    model = HybridMambaMoE(student_config)

    # Load checkpoint if specified
    checkpoint_path = student_cfg.get("checkpoint", None)
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading student checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("student_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Student checkpoint loaded.")

    # Set dtype
    dtype_str = student_cfg.get("dtype", "float16")
    if dtype_str == "float16":
        model = model.half()
    elif dtype_str == "bfloat16":
        model = model.to(torch.bfloat16)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Student: {param_count:.1f}M params")

    return model.to(device)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stage1(config: dict[str, Any], device: torch.device) -> None:
    """Stage 1: Layer Alignment with CKA monitoring."""
    from aya_distill.distill.alignment import AlignmentConfig, LayerAlignmentTrainer
    from aya_distill.distill.data import (
        DistillDataConfig,
        create_distill_dataloader,
        get_shared_tokenizer,
    )

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Load tokenizer
    teacher_name = config["teacher"]["name"]
    tokenizer = get_shared_tokenizer(teacher_name)

    # Load models
    teacher = load_teacher(config["teacher"], device)
    student = load_student(
        config["student"], device, vocab_size_override=len(tokenizer)
    )
    student.resize_token_embeddings(len(tokenizer))

    # Get layer counts
    from aya_distill.distill.hooks import _get_transformer_layers
    teacher_layers = _get_transformer_layers(teacher)
    n_teacher = len(teacher_layers) if teacher_layers else 32
    n_student = len(student.layers)

    # Teacher hidden size
    if hasattr(teacher.config, "hidden_size"):
        d_teacher = teacher.config.hidden_size
    else:
        d_teacher = 4096  # Aya-8B default

    d_student = student.config.d_model

    # Build configs
    train_cfg = config.get("training", {})
    align_cfg = config.get("alignment", {})

    alignment_config = AlignmentConfig(
        loss_type=align_cfg.get("loss_type", "mse+cosine"),
        cka_threshold=align_cfg.get("cka_threshold", 0.75),
        cka_check_every=align_cfg.get("cka_check_every", 500),
        layer_mapping_strategy=align_cfg.get("layer_mapping", "interleaved"),
        lr=train_cfg.get("lr", 1e-4),
        warmup_steps=train_cfg.get("warmup_steps", 500),
        total_steps=train_cfg.get("total_steps", 10000),
        batch_size=train_cfg.get("batch_size", 4),
        max_seq_len=train_cfg.get("max_seq_len", 512),
        gradient_accumulation=train_cfg.get("gradient_accumulation", 8),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        output_dir=config.get("output_dir", "checkpoints/stage1_alignment"),
        seed=seed,
    )

    # Create data loader
    data_config = DistillDataConfig(
        mode="alignment",
        languages=config.get("languages", []),
        max_seq_len=alignment_config.max_seq_len,
        batch_size=alignment_config.batch_size,
        seed=seed,
    )
    dataloader = create_distill_dataloader(data_config, tokenizer)

    # Create trainer
    trainer = LayerAlignmentTrainer(
        teacher=teacher,
        student=student,
        config=alignment_config,
        n_teacher_layers=n_teacher,
        n_student_layers=n_student,
        d_teacher=d_teacher,
        d_student=d_student,
        device=device,
    )

    # Train
    results = trainer.train(dataloader)
    logger.info(f"Stage 1 complete: {results}")


def run_stage2(config: dict[str, Any], device: torch.device) -> None:
    """Stage 2: KL Distillation with temperature scaling."""
    from aya_distill.distill.kl_distillation import KLDistillationConfig, KLDistillationTrainer
    from aya_distill.distill.data import (
        DistillDataConfig,
        create_distill_dataloader,
        get_shared_tokenizer,
    )

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Load tokenizer
    teacher_name = config["teacher"]["name"]
    tokenizer = get_shared_tokenizer(teacher_name)

    # Load models
    teacher = load_teacher(config["teacher"], device)
    student = load_student(
        config["student"], device, vocab_size_override=len(tokenizer)
    )
    student.resize_token_embeddings(len(tokenizer))

    # Build config
    train_cfg = config.get("training", {})
    kl_cfg = config.get("kl", {})

    kl_config = KLDistillationConfig(
        temperature=kl_cfg.get("temperature", 2.0),
        alpha=kl_cfg.get("alpha", 0.7),
        lr=train_cfg.get("lr", 5e-5),
        ssm_lr_multiplier=train_cfg.get("ssm_lr_multiplier", 1.0),
        warmup_steps=train_cfg.get("warmup_steps", 500),
        total_steps=train_cfg.get("total_steps", 20000),
        batch_size=train_cfg.get("batch_size", 4),
        max_seq_len=train_cfg.get("max_seq_len", 512),
        gradient_accumulation=train_cfg.get("gradient_accumulation", 8),
        teacher_vocab_size=len(tokenizer),
        student_vocab_size=len(tokenizer),
        output_dir=config.get("output_dir", "checkpoints/stage2_kl"),
        seed=seed,
    )

    # Create data loader
    data_config = DistillDataConfig(
        mode="distillation",
        languages=config.get("languages", []),
        max_seq_len=kl_config.max_seq_len,
        batch_size=kl_config.batch_size,
        seed=seed,
    )
    dataloader = create_distill_dataloader(data_config, tokenizer)

    # Create trainer
    trainer = KLDistillationTrainer(
        teacher=teacher,
        student=student,
        config=kl_config,
        device=device,
    )

    # Train
    results = trainer.train(dataloader)
    logger.info(f"Stage 2 complete: {results}")


def run_stage3(config: dict[str, Any], device: torch.device) -> None:
    """Stage 3: Supervised Fine-Tuning for capability recovery."""
    from aya_distill.distill.sft import SFTConfig, SFTTrainer
    from aya_distill.distill.data import (
        DistillDataConfig,
        create_distill_dataloader,
        get_shared_tokenizer,
    )

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Load tokenizer
    teacher_name = config.get("teacher", {}).get(
        "name", "CohereForAI/aya-expanse-8b"
    )
    tokenizer = get_shared_tokenizer(teacher_name)

    # Load student only (no teacher needed for SFT)
    student = load_student(
        config["student"], torch.device("cpu"),
        vocab_size_override=len(tokenizer),
    )
    student.resize_token_embeddings(len(tokenizer))

    # Build config
    train_cfg = config.get("training", {})

    sft_config = SFTConfig(
        lr=train_cfg.get("lr", 2e-5),
        warmup_steps=train_cfg.get("warmup_steps", 200),
        total_steps=train_cfg.get("total_steps", 5000),
        batch_size=train_cfg.get("batch_size", 4),
        max_seq_len=train_cfg.get("max_seq_len", 1024),
        gradient_accumulation=train_cfg.get("gradient_accumulation", 4),
        output_dir=config.get("output_dir", "checkpoints/stage3_sft"),
        seed=seed,
    )

    # Create data loader
    data_config = DistillDataConfig(
        mode="sft",
        languages=config.get("languages", []),
        max_seq_len=sft_config.max_seq_len,
        batch_size=sft_config.batch_size,
        seed=seed,
    )
    dataloader = create_distill_dataloader(data_config, tokenizer)

    # Create trainer
    trainer = SFTTrainer(
        student=student,
        config=sft_config,
        device=device,
    )

    # Train
    results = trainer.train(dataloader)
    logger.info(f"Stage 3 complete: {results}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project Aya Distillation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1  Layer Alignment (CKA-guided, MSE+cosine loss)
  2  KL Distillation (temperature-scaled soft targets)
  3  Supervised Fine-Tuning (multilingual recovery)

Example:
  python scripts/run_distill.py --stage 1 --config configs/distill_stage1.yaml
        """,
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Distillation stage to run (1, 2, or 3)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to stage config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cuda:0', 'cpu'",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=0,
        help="Step to resume training from",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            logger.warning("CUDA not available, falling back to CPU.")
            return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    device = resolve_device(args.device)

    logger.info(
        f"=== Project Aya Distillation Stage {args.stage} ==="
    )
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    stage_runners = {
        1: run_stage1,
        2: run_stage2,
        3: run_stage3,
    }

    runner = stage_runners[args.stage]
    runner(config, device)

    logger.info(f"=== Stage {args.stage} finished ===")


if __name__ == "__main__":
    main()
