#!/usr/bin/env python3
"""
Full production distillation pipeline for Project Aya.

Runs all 3 stages sequentially on GPU with real data:
    Stage 1: Layer Alignment (CKA-guided) with ClimbMix data
    Stage 2: KL Distillation (temperature-scaled, SSM 10x LR) with ClimbMix
    Stage 3: SFT (multilingual recovery) with Aya dataset

Designed for RunPod A6000 (48GB VRAM):
    - Teacher: aya-expanse-8b in bf16 (~16GB)
    - Student: Aetheris ~800M in bf16 (~1.5GB)
    - Remaining ~30GB for activations, optimizer, grad checkpointing

Usage:
    # Run all stages:
    python scripts/run_full_distill.py --config configs/runpod_distill.yaml

    # Run a single stage:
    python scripts/run_full_distill.py --config configs/runpod_distill.yaml --stage 2

    # Resume from checkpoint:
    python scripts/run_full_distill.py --config configs/runpod_distill.yaml --stage 2 \
        --student-checkpoint checkpoints/stage2_kl/step_5000.pt --resume-step 5000

Wayy Research, 2024-2026.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
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
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "training.log"),
    ],
)
logger = logging.getLogger("full_distill")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_teacher(
    teacher_cfg: dict[str, Any],
) -> torch.nn.Module:
    """Load teacher model with HuggingFace."""
    from transformers import AutoModelForCausalLM

    name = teacher_cfg["name"]
    dtype_str = teacher_cfg.get("dtype", "bfloat16")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    logger.info("Loading teacher: %s (dtype=%s)", name, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map=teacher_cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Teacher loaded: %.2fB params", n_params)
    return model


def load_student(
    student_cfg: dict[str, Any],
    vocab_size: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load or initialize the Aetheris student model."""
    from aetheris.config import AetherisConfig
    from aetheris.model import HybridMambaMoE

    config_path = student_cfg.get("config_path", "configs/student.yaml")
    if not Path(config_path).is_absolute():
        config_path = str(PROJECT_ROOT / config_path)

    student_config = AetherisConfig.from_yaml(config_path)
    student_config.vocab_size = vocab_size
    student_config.gradient_checkpointing = True

    model = HybridMambaMoE(student_config)

    # Load checkpoint if provided
    checkpoint_path = student_cfg.get("checkpoint")
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info("Loading student checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("student_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    # Set dtype
    dtype_str = student_cfg.get("dtype", "bfloat16")
    if dtype_str == "bfloat16":
        model = model.to(torch.bfloat16)
    elif dtype_str == "float16":
        model = model.half()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        "Student: d_model=%d, n_layer=%d, vocab=%d, %.1fM params",
        student_config.d_model, student_config.n_layer,
        student_config.vocab_size, n_params,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Stage 1: Layer Alignment
# ---------------------------------------------------------------------------
def run_stage1(
    config: dict[str, Any],
    device: torch.device,
) -> None:
    """Stage 1: CKA-guided layer alignment with ClimbMix data."""
    stage_cfg = config.get("stage1", {})
    if not stage_cfg.get("enabled", True):
        logger.info("Stage 1 disabled, skipping")
        return

    logger.info("=" * 60)
    logger.info("  STAGE 1: Layer Alignment")
    logger.info("=" * 60)

    from aya_distill.distill.alignment import AlignmentConfig, LayerAlignmentTrainer
    from aya_distill.distill.climbmix import ClimbMixConfig, create_climbmix_dataloader
    from aya_distill.distill.data import get_shared_tokenizer
    from aya_distill.distill.hooks import _get_transformer_layers

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Tokenizer
    teacher_name = config["teacher"]["name"]
    tokenizer = get_shared_tokenizer(teacher_name)

    # Models
    teacher = load_teacher(config["teacher"])
    student = load_student(config["student"], len(tokenizer), device)
    student.resize_token_embeddings(len(tokenizer))

    # Layer info
    teacher_layers = _get_transformer_layers(teacher)
    n_teacher = len(teacher_layers) if teacher_layers else 32
    n_student = len(student.layers)
    d_teacher = getattr(teacher.config, "hidden_size", 4096)
    d_student = student.config.d_model

    # Alignment config
    alignment_config = AlignmentConfig(
        loss_type=stage_cfg.get("loss_type", "mse+cosine"),
        cka_threshold=stage_cfg.get("cka_threshold", 0.75),
        cka_check_every=stage_cfg.get("cka_check_every", 500),
        layer_mapping_strategy="interleaved",
        lr=float(stage_cfg.get("lr", 1e-4)),
        warmup_steps=stage_cfg.get("warmup_steps", 500),
        total_steps=stage_cfg.get("total_steps", 10000),
        batch_size=stage_cfg.get("batch_size", 4),
        max_seq_len=stage_cfg.get("max_seq_len", 512),
        gradient_accumulation=stage_cfg.get("gradient_accumulation", 8),
        gradient_checkpointing=stage_cfg.get("gradient_checkpointing", True),
        output_dir=stage_cfg.get("output_dir", "checkpoints/stage1_alignment"),
        save_every=stage_cfg.get("save_every", 1000),
        log_every=stage_cfg.get("log_every", 50),
        seed=seed,
    )

    # Data: ClimbMix retokenized
    climbmix_cfg = config.get("data", {}).get("climbmix", {})
    data_config = ClimbMixConfig(
        mode=climbmix_cfg.get("mode", "retokenize"),
        max_seq_len=alignment_config.max_seq_len,
        batch_size=alignment_config.batch_size,
        min_tokens=climbmix_cfg.get("min_tokens", 32),
        buffer_size=climbmix_cfg.get("buffer_size", 500),
        seed=seed,
    )
    dataloader = create_climbmix_dataloader(data_config, tokenizer=tokenizer)

    # Trainer
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

    t_start = time.time()
    results = trainer.train(dataloader)
    elapsed = time.time() - t_start

    logger.info(
        "Stage 1 complete: %d steps in %.1f min, best CKA=%.4f",
        results["final_step"], elapsed / 60, results["best_cka_mean"],
    )

    # Cleanup teacher from GPU
    del teacher
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 2: KL Distillation
# ---------------------------------------------------------------------------
def run_stage2(
    config: dict[str, Any],
    device: torch.device,
    resume_step: int = 0,
) -> None:
    """Stage 2: KL distillation with SSM 10x LR boost."""
    stage_cfg = config.get("stage2", {})
    if not stage_cfg.get("enabled", True):
        logger.info("Stage 2 disabled, skipping")
        return

    logger.info("=" * 60)
    logger.info("  STAGE 2: KL Distillation")
    logger.info("=" * 60)

    from aya_distill.distill.kl_distillation import KLDistillationConfig, KLDistillationTrainer
    from aya_distill.distill.climbmix import ClimbMixConfig, create_climbmix_dataloader
    from aya_distill.distill.data import get_shared_tokenizer

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Tokenizer
    teacher_name = config["teacher"]["name"]
    tokenizer = get_shared_tokenizer(teacher_name)

    # Models
    teacher = load_teacher(config["teacher"])

    # Load student from Stage 1 checkpoint if no explicit checkpoint
    student_cfg = dict(config["student"])
    if not student_cfg.get("checkpoint"):
        stage1_ckpt = Path(
            config.get("stage1", {}).get(
                "output_dir", "checkpoints/stage1_alignment"
            )
        ) / "best.pt"
        if stage1_ckpt.exists():
            student_cfg["checkpoint"] = str(stage1_ckpt)
            logger.info("Loading Stage 1 best checkpoint: %s", stage1_ckpt)

    student = load_student(student_cfg, len(tokenizer), device)
    student.resize_token_embeddings(len(tokenizer))

    # KL config with SSM LR boost
    kl_config = KLDistillationConfig(
        temperature=float(stage_cfg.get("temperature", 2.0)),
        alpha=float(stage_cfg.get("alpha", 0.7)),
        lr=float(stage_cfg.get("lr", 5e-5)),
        ssm_lr_multiplier=float(stage_cfg.get("ssm_lr_multiplier", 10.0)),
        warmup_steps=stage_cfg.get("warmup_steps", 500),
        total_steps=stage_cfg.get("total_steps", 20000),
        batch_size=stage_cfg.get("batch_size", 4),
        max_seq_len=stage_cfg.get("max_seq_len", 512),
        gradient_accumulation=stage_cfg.get("gradient_accumulation", 8),
        gradient_checkpointing=stage_cfg.get("gradient_checkpointing", True),
        teacher_vocab_size=len(tokenizer),
        student_vocab_size=len(tokenizer),
        output_dir=stage_cfg.get("output_dir", "checkpoints/stage2_kl"),
        save_every=stage_cfg.get("save_every", 2000),
        log_every=stage_cfg.get("log_every", 50),
        seed=seed,
    )

    # Data: ClimbMix retokenized
    climbmix_cfg = config.get("data", {}).get("climbmix", {})
    data_config = ClimbMixConfig(
        mode=climbmix_cfg.get("mode", "retokenize"),
        max_seq_len=kl_config.max_seq_len,
        batch_size=kl_config.batch_size,
        min_tokens=climbmix_cfg.get("min_tokens", 32),
        buffer_size=climbmix_cfg.get("buffer_size", 500),
        seed=seed,
    )
    dataloader = create_climbmix_dataloader(data_config, tokenizer=tokenizer)

    # Trainer
    trainer = KLDistillationTrainer(
        teacher=teacher,
        student=student,
        config=kl_config,
        device=device,
    )

    t_start = time.time()
    results = trainer.train(dataloader, resume_step=resume_step)
    elapsed = time.time() - t_start

    logger.info(
        "Stage 2 complete: %d steps in %.1f min, best loss=%.4f",
        results["final_step"], elapsed / 60, results["best_loss"],
    )

    del teacher
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 3: Supervised Fine-Tuning
# ---------------------------------------------------------------------------
def run_stage3(
    config: dict[str, Any],
    device: torch.device,
    resume_step: int = 0,
) -> None:
    """Stage 3: SFT for multilingual capability recovery."""
    stage_cfg = config.get("stage3", {})
    if not stage_cfg.get("enabled", True):
        logger.info("Stage 3 disabled, skipping")
        return

    logger.info("=" * 60)
    logger.info("  STAGE 3: Supervised Fine-Tuning")
    logger.info("=" * 60)

    from aya_distill.distill.sft import SFTConfig, SFTTrainer
    from aya_distill.distill.data import (
        DistillDataConfig,
        create_distill_dataloader,
        get_shared_tokenizer,
    )

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Tokenizer
    teacher_name = config["teacher"]["name"]
    tokenizer = get_shared_tokenizer(teacher_name)

    # Load student from Stage 2 checkpoint
    student_cfg = dict(config["student"])
    if not student_cfg.get("checkpoint"):
        stage2_ckpt = Path(
            config.get("stage2", {}).get(
                "output_dir", "checkpoints/stage2_kl"
            )
        ) / "best.pt"
        if stage2_ckpt.exists():
            student_cfg["checkpoint"] = str(stage2_ckpt)
            logger.info("Loading Stage 2 best checkpoint: %s", stage2_ckpt)

    student = load_student(student_cfg, len(tokenizer), device)
    student.resize_token_embeddings(len(tokenizer))

    # SFT config
    sft_config = SFTConfig(
        lr=float(stage_cfg.get("lr", 2e-5)),
        warmup_steps=stage_cfg.get("warmup_steps", 200),
        total_steps=stage_cfg.get("total_steps", 5000),
        batch_size=stage_cfg.get("batch_size", 4),
        max_seq_len=stage_cfg.get("max_seq_len", 1024),
        gradient_accumulation=stage_cfg.get("gradient_accumulation", 4),
        gradient_checkpointing=stage_cfg.get("gradient_checkpointing", True),
        output_dir=stage_cfg.get("output_dir", "checkpoints/stage3_sft"),
        save_every=stage_cfg.get("save_every", 500),
        log_every=stage_cfg.get("log_every", 25),
        seed=seed,
    )

    # Data: Aya multilingual chat dataset
    sft_data_cfg = config.get("data", {}).get("sft", {})
    data_config = DistillDataConfig(
        mode="sft",
        languages=config.get("languages", []),
        max_seq_len=sft_config.max_seq_len,
        batch_size=sft_config.batch_size,
        dataset_name=sft_data_cfg.get("dataset_name", "CohereForAI/aya_dataset"),
        streaming=sft_data_cfg.get("streaming", True),
        seed=seed,
    )
    dataloader = create_distill_dataloader(data_config, tokenizer)

    # Trainer (no teacher needed for SFT)
    trainer = SFTTrainer(
        student=student,
        config=sft_config,
        device=device,
    )

    t_start = time.time()
    results = trainer.train(dataloader, resume_step=resume_step)
    elapsed = time.time() - t_start

    logger.info(
        "Stage 3 complete: %d steps in %.1f min, best eval loss=%.4f",
        results["final_step"], elapsed / 60, results["best_eval_loss"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project Aya Full Distillation Pipeline",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to distillation config YAML",
    )
    parser.add_argument(
        "--stage", type=int, default=None, choices=[1, 2, 3],
        help="Run a single stage (default: all stages sequentially)",
    )
    parser.add_argument(
        "--student-checkpoint", type=str, default=None,
        help="Override student checkpoint path",
    )
    parser.add_argument(
        "--resume-step", type=int, default=0,
        help="Step to resume training from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Override student checkpoint if provided
    if args.student_checkpoint:
        config["student"]["checkpoint"] = args.student_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — this will be very slow!")

    logger.info("=" * 60)
    logger.info("  Project Aya — Full Distillation Pipeline")
    logger.info("  Device: %s", device)
    if torch.cuda.is_available():
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "  VRAM: %.1f GB",
            torch.cuda.get_device_properties(0).total_mem / 1e9,
        )
    logger.info("  Languages: %s", config.get("languages", []))
    logger.info("=" * 60)

    t_total = time.time()

    if args.stage is None:
        # Run all stages
        run_stage1(config, device)
        run_stage2(config, device)
        run_stage3(config, device)
    elif args.stage == 1:
        run_stage1(config, device)
    elif args.stage == 2:
        run_stage2(config, device, resume_step=args.resume_step)
    elif args.stage == 3:
        run_stage3(config, device, resume_step=args.resume_step)

    total_time = time.time() - t_total
    logger.info("=" * 60)
    logger.info("  Pipeline complete!")
    logger.info("  Total time: %.1f min (%.1f hours)", total_time / 60, total_time / 3600)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
