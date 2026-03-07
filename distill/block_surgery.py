"""
Full model surgery: convert a complete transformer into Aetheris Mamba-MoE.

Takes a Cohere Aya model and produces a fully initialized Aetheris model
by converting all attention layers → SSM blocks and all FFN layers → MoE blocks.

Usage:
    from distill.block_surgery import convert_model

    aetheris, report = convert_model(
        teacher_name="CohereForAI/aya-expanse-8b",
        student_config_path="configs/student.yaml",
        strategy="hybrid",
        device="cuda",
    )

Wayy Research, 2024-2026.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml

from distill.converter import ConversionConfig, convert_block

logger = logging.getLogger(__name__)


def _get_teacher_layers(model: nn.Module) -> list[nn.Module]:
    """Extract transformer layers from a Cohere/Llama-style model."""
    # CohereForCausalLM: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # Direct model: model.layers
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError(
        f"Cannot find transformer layers in model of type {type(model)}. "
        f"Expected model.model.layers or model.layers."
    )


def _get_attn(layer: nn.Module) -> nn.Module:
    """Get attention sub-module from a transformer layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    raise ValueError(f"Cannot find attention in {type(layer)}")


def _get_mlp(layer: nn.Module) -> nn.Module:
    """Get MLP sub-module from a transformer layer."""
    if hasattr(layer, "mlp"):
        return layer.mlp
    raise ValueError(f"Cannot find MLP in {type(layer)}")


def convert_model(
    teacher_name: str,
    student_config_path: str,
    strategy: str = "hybrid",
    conversion_config: Optional[ConversionConfig] = None,
    calibration_data: Optional[list[torch.Tensor]] = None,
    device: str = "cuda",
    teacher_dtype: str = "bfloat16",
    output_dir: Optional[str] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Convert a full Cohere transformer into an Aetheris Mamba-MoE model.

    Args:
        teacher_name: HuggingFace model ID for the teacher.
        student_config_path: Path to student YAML config.
        strategy: "weight_map", "distillation", or "hybrid".
        conversion_config: Optional detailed conversion config.
        calibration_data: Optional calibration tensors for refinement.
        device: Target device.
        teacher_dtype: dtype for loading teacher.
        output_dir: If set, save converted model and report here.

    Returns:
        (student_model, conversion_report) tuple.
    """
    if conversion_config is None:
        conversion_config = ConversionConfig(strategy=strategy)

    torch.manual_seed(conversion_config.seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    report: dict[str, Any] = {
        "teacher": teacher_name,
        "student_config": student_config_path,
        "strategy": strategy,
        "device": str(dev),
        "layers": [],
    }

    # --- Load teacher ---
    logger.info("Loading teacher: %s", teacher_name)
    t0 = time.perf_counter()

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=dtype_map.get(teacher_dtype, torch.bfloat16),
        device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()

    teacher_layers = _get_teacher_layers(teacher)
    n_teacher = len(teacher_layers)
    teacher_hidden = teacher.config.hidden_size
    teacher_intermediate = teacher.config.intermediate_size

    report["teacher_load_time"] = round(time.perf_counter() - t0, 2)
    report["teacher_layers"] = n_teacher
    report["teacher_hidden"] = teacher_hidden
    report["teacher_intermediate"] = teacher_intermediate

    logger.info(
        "Teacher loaded: %d layers, hidden=%d, intermediate=%d (%.1fs)",
        n_teacher, teacher_hidden, teacher_intermediate,
        report["teacher_load_time"],
    )

    # --- Load student ---
    logger.info("Loading student config: %s", student_config_path)

    # Try to import Aetheris
    aetheris_path = Path(__file__).resolve().parent.parent.parent / "aetheris"
    if aetheris_path.exists():
        sys.path.insert(0, str(aetheris_path))

    try:
        from aetheris import AetherisConfig, HybridMambaMoE
    except ImportError:
        raise ImportError(
            "Cannot import Aetheris model. Ensure the aetheris package is "
            f"available at {aetheris_path} or installed."
        )

    with open(student_config_path) as f:
        student_yaml = yaml.safe_load(f)
    student_cfg = AetherisConfig(**student_yaml)
    student = HybridMambaMoE(student_cfg)

    n_student = student_cfg.n_layer
    student_d_model = student_cfg.d_model
    student_d_inner = student_cfg.d_inner or student_cfg.d_model * student_cfg.ssm_expand
    student_d_ff = student_cfg.d_ff or student_cfg.d_model * 3

    report["student_layers"] = n_student
    report["student_d_model"] = student_d_model
    report["student_d_inner"] = student_d_inner
    report["student_d_ff"] = student_d_ff
    report["student_params"] = sum(p.numel() for p in student.parameters())

    logger.info(
        "Student: %d layers, d_model=%d, d_inner=%d, d_ff=%d, params=%s",
        n_student, student_d_model, student_d_inner, student_d_ff,
        f"{report['student_params']:,}",
    )

    # --- Transfer embeddings ---
    logger.info("Transferring embeddings...")
    teacher_embed = teacher.model.embed_tokens.weight.data
    student_embed = student.embedding.weight.data

    t_vocab, t_dim = teacher_embed.shape
    s_vocab, s_dim = student_embed.shape

    # Copy what fits, SVD-project if dimensions differ
    copy_vocab = min(t_vocab, s_vocab)
    copy_dim = min(t_dim, s_dim)

    if t_dim == s_dim:
        student.embedding.weight.data[:copy_vocab] = teacher_embed[:copy_vocab]
    else:
        # SVD project embedding matrix
        from distill.converter import svd_project
        projected = svd_project(teacher_embed[:copy_vocab], copy_vocab, s_dim)
        student.embedding.weight.data[:copy_vocab] = projected

    report["embedding_transfer"] = {
        "teacher_vocab": t_vocab,
        "student_vocab": s_vocab,
        "teacher_dim": t_dim,
        "student_dim": s_dim,
    }

    # --- Convert layers ---
    # Aetheris alternates: even=SSM, odd=MoE
    # We map teacher layers proportionally to student layers
    teacher_per_student = n_teacher / n_student

    for s_idx in range(n_student):
        t_idx = int(s_idx * teacher_per_student)
        t_idx = min(t_idx, n_teacher - 1)
        teacher_layer = teacher_layers[t_idx]

        # Even layers = SSM (from attention)
        if s_idx % 2 == 0:
            logger.info(
                "Converting layer %d/%d: teacher[%d].attn → student[%d].SSM",
                s_idx + 1, n_student, t_idx, s_idx,
            )
            attn = _get_attn(teacher_layer)
            ssm_block = student.layers[s_idx]

            layer_metrics = convert_block(
                teacher_layer=attn,
                student_block=ssm_block,
                config=conversion_config,
                layer_idx=s_idx,
                block_type="attn",
                calibration_data=calibration_data,
                device=dev,
            )
        # Odd layers = MoE (from FFN)
        else:
            logger.info(
                "Converting layer %d/%d: teacher[%d].mlp → student[%d].MoE",
                s_idx + 1, n_student, t_idx, s_idx,
            )
            mlp = _get_mlp(teacher_layer)
            moe_block = student.layers[s_idx]

            layer_metrics = convert_block(
                teacher_layer=mlp,
                student_block=moe_block,
                config=conversion_config,
                layer_idx=s_idx,
                block_type="ffn",
                calibration_data=calibration_data,
                device=dev,
            )

        layer_metrics["teacher_layer_idx"] = t_idx
        report["layers"].append(layer_metrics)

    # --- Summary ---
    cka_scores = [
        m.get("refinement", {}).get("final_cka")
        for m in report["layers"]
        if m.get("refinement", {}).get("final_cka") is not None
    ]
    if cka_scores:
        report["mean_cka"] = sum(cka_scores) / len(cka_scores)
        report["min_cka"] = min(cka_scores)
        report["low_cka_layers"] = [
            m["layer_idx"] for m in report["layers"]
            if (m.get("refinement", {}).get("final_cka") or 1.0)
            < conversion_config.cka_threshold
        ]

    logger.info("Conversion complete.")
    if "mean_cka" in report:
        logger.info(
            "  Mean CKA: %.4f | Min CKA: %.4f | Low CKA layers: %s",
            report["mean_cka"], report["min_cka"], report.get("low_cka_layers", []),
        )

    # --- Save ---
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(student.state_dict(), out / "converted_model.pt")
        logger.info("Model saved to %s", out / "converted_model.pt")

        # Save report
        with open(out / "conversion_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved to %s", out / "conversion_report.json")

    # Free teacher memory
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return student, report
