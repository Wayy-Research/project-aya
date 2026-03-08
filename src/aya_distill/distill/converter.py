"""
Transformer → Mamba block converter.

Converts individual transformer attention blocks into Mamba SSM blocks
with minimal information loss. This is the core research contribution.

Three strategies:
    1. weight_map: Direct weight initialization from attention weights
    2. distillation: Train SSM block to match attention block activations
    3. hybrid: Weight map for init, then refine with distillation

The key insight: attention computes a weighted sum of values based on
query-key similarity. SSM computes a linear recurrence with selective
gating. We initialize the SSM's projections from the attention's
projections so it starts near the attention solution, then refine.

References:
    Wang et al., "The Mamba in the Llama" (2024)
    Gu & Dao, "Mamba: Linear-Time Sequence Modeling" (2023)

Wayy Research, 2024-2026.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cohere teacher architecture reference (aya-expanse-8b)
# ---------------------------------------------------------------------------
# model.model.layers[i].self_attn:
#   q_proj: (hidden_size, hidden_size)        -- 4096 → 4096
#   k_proj: (hidden_size, num_kv_heads * head_dim)  -- 4096 → 1024 (GQA)
#   v_proj: (hidden_size, num_kv_heads * head_dim)  -- 4096 → 1024 (GQA)
#   o_proj: (hidden_size, hidden_size)        -- 4096 → 4096
#
# model.model.layers[i].mlp:
#   gate_proj: (hidden_size, intermediate_size)  -- 4096 → 14336
#   up_proj:   (hidden_size, intermediate_size)  -- 4096 → 14336
#   down_proj: (intermediate_size, hidden_size)  -- 14336 → 4096
#
# Student SSMBlock (from aetheris/modules/ssm.py):
#   in_proj:    (d_model → 2*d_inner)  -- e.g. 1024 → 2048
#   conv_d:     depthwise conv1d(d_inner, k=3, pad=2)
#   gate_proj:  (d_model → d_inner)
#   B_proj:     (d_inner → d_state)
#   C_proj:     (d_inner → d_state)
#   delta_proj: (d_inner → d_inner)
#   out_proj:   (d_inner → d_model)
#   A_log:      (d_inner, d_state)  parameter
#   D:          (d_inner,) parameter
#
# Student Expert (from aetheris/modules/expert.py):
#   w1: (d_model → d_ff)
#   w2: (d_ff → d_model)


@dataclass
class ConversionConfig:
    """Configuration for block conversion."""

    strategy: str = "hybrid"  # "weight_map", "distillation", "hybrid"

    # Weight mapping options
    a_init: str = "exponential_decay"  # "random", "constant", "exponential_decay"
    delta_init: str = "uniform"  # "random", "uniform", "attention_stats"
    ffn_to_moe: str = "replicate"  # "replicate", "svd_split", "random"

    # Distillation refinement options
    refinement_steps: int = 1000
    refinement_lr: float = 1e-4
    refinement_batch_size: int = 4
    refinement_max_seq_len: int = 512
    cka_check_every: int = 100
    cka_threshold: float = 0.75

    seed: int = 42


# ---------------------------------------------------------------------------
# SVD helpers for dimension reduction
# ---------------------------------------------------------------------------

def svd_project(
    weight: torch.Tensor,
    target_rows: int,
    target_cols: int,
) -> torch.Tensor:
    """Project a weight matrix to target dimensions using truncated SVD.

    Preserves as much variance as possible in the lower-rank approximation.
    """
    w = weight.float()

    # If we need fewer rows, take top singular vectors on the left
    # If we need fewer cols, take top singular vectors on the right
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)

    k = min(target_rows, target_cols, len(S))
    U_k = U[:target_rows, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :target_cols]

    projected = U_k @ torch.diag(S_k) @ Vh_k
    return projected.to(weight.dtype)


def _extract_submatrix(
    weight: torch.Tensor,
    target_rows: int,
    target_cols: int,
) -> torch.Tensor:
    """Extract or project a weight matrix to target dimensions.

    If the target is smaller, uses SVD for optimal low-rank approximation.
    If the target is larger, pads with scaled random initialization.
    """
    src_rows, src_cols = weight.shape

    if src_rows == target_rows and src_cols == target_cols:
        return weight.clone()

    if src_rows >= target_rows and src_cols >= target_cols:
        return svd_project(weight, target_rows, target_cols)

    # Need to pad — start with SVD projection to available size
    out = torch.zeros(target_rows, target_cols, dtype=weight.dtype, device=weight.device)
    r = min(src_rows, target_rows)
    c = min(src_cols, target_cols)

    if r == src_rows and c == src_cols:
        out[:r, :c] = weight
    else:
        out[:r, :c] = svd_project(weight, r, c)

    # Initialize padding with small random values
    if r < target_rows:
        nn.init.xavier_uniform_(out[r:, :].unsqueeze(0), gain=0.1)
        out[r:, :] = out[r:, :].squeeze(0)
    if c < target_cols:
        nn.init.xavier_uniform_(out[:, c:].unsqueeze(0), gain=0.1)
        out[:, c:] = out[:, c:].squeeze(0)

    return out


# ---------------------------------------------------------------------------
# Attention → SSM block conversion
# ---------------------------------------------------------------------------

def convert_attention_to_ssm(
    attn_layer: nn.Module,
    ssm_block: nn.Module,
    config: ConversionConfig,
    layer_idx: int = 0,
) -> dict[str, Any]:
    """
    Initialize an SSMBlock from a transformer attention layer's weights.

    Maps attention projections to SSM projections:
        - V_proj → first half of in_proj (the "input" path)
        - Q_proj norms → gate_proj initialization
        - O_proj → out_proj
        - A_log initialized for exponential decay (recency bias)
        - D initialized from residual stream statistics

    Args:
        attn_layer: Transformer attention module with q_proj, k_proj, v_proj, o_proj.
        ssm_block: Target SSMBlock to initialize.
        config: Conversion configuration.
        layer_idx: Layer index for logging.

    Returns:
        Dict with conversion metrics.
    """
    metrics: dict[str, Any] = {"layer_idx": layer_idx, "type": "attn->ssm"}

    d_model = ssm_block.d_model
    d_inner = ssm_block.d_inner
    d_state = ssm_block.d_state

    with torch.no_grad():
        # --- in_proj: (d_model → 2*d_inner) ---
        # Split into x_in and z_gate paths
        # x_in path: initialized from V projection (value = what to remember)
        # z_gate path: initialized from Q projection (query = what to attend to)
        v_weight = attn_layer.v_proj.weight.data  # (kv_dim, d_model)
        q_weight = attn_layer.q_proj.weight.data  # (d_model, d_model)

        x_in_weight = _extract_submatrix(v_weight, d_inner, d_model)
        z_gate_weight = _extract_submatrix(q_weight, d_inner, d_model)

        ssm_block.in_proj.weight.data = torch.cat(
            [x_in_weight, z_gate_weight], dim=0
        )  # (2*d_inner, d_model)

        logger.info(
            "  [Layer %d] in_proj: V(%s)→x_in(%d,%d), Q(%s)→z_gate(%d,%d)",
            layer_idx, list(v_weight.shape), d_inner, d_model,
            list(q_weight.shape), d_inner, d_model,
        )

        # --- gate_proj: (d_model → d_inner) ---
        # Initialize from K projection (key = what determines gating)
        k_weight = attn_layer.k_proj.weight.data  # (kv_dim, d_model)
        ssm_block.gate_proj.weight.data = _extract_submatrix(
            k_weight, d_inner, d_model
        )

        # --- out_proj: (d_inner → d_model) ---
        # Direct mapping from attention output projection
        o_weight = attn_layer.o_proj.weight.data  # (d_model, d_model)
        ssm_block.out_proj.weight.data = _extract_submatrix(
            o_weight, d_model, d_inner
        )

        # --- A_log: (d_inner, d_state) ---
        # Initialize for exponential decay — recent tokens have stronger influence
        # A = -exp(A_log), so A_log ≈ -4 → A ≈ -0.018 (slow decay)
        # More negative A_log → faster decay (more local attention)
        if config.a_init == "exponential_decay":
            # Vary decay rate across dimensions — some local, some global
            decay_rates = torch.linspace(-5.0, -2.0, d_inner)
            A_log = decay_rates.unsqueeze(1).expand(d_inner, d_state)
            A_log = A_log + torch.randn_like(A_log) * 0.1  # small noise
            ssm_block.A_log.data = A_log
        elif config.a_init == "constant":
            ssm_block.A_log.data.fill_(-4.0)
        # else: keep random init

        # --- D: (d_inner,) skip connection ---
        # Initialize to small values — residual connection is already in the block
        ssm_block.D.data = torch.ones(d_inner) * 0.1

        # --- delta_proj: (d_inner → d_inner) ---
        if config.delta_init == "uniform":
            # Initialize to produce roughly uniform time steps
            nn.init.xavier_uniform_(ssm_block.delta_proj.weight, gain=0.1)
        # else: keep existing init

        # --- B_proj, C_proj: (d_inner → d_state) ---
        # These are the SSM input/output matrices — keep xavier init
        # They'll be refined during distillation

        # --- conv_d: depthwise conv1d ---
        # Initialize to approximate identity (pass-through with slight smoothing)
        conv_weight = ssm_block.conv_d.weight.data  # (d_inner, 1, 3)
        conv_weight.zero_()
        conv_weight[:, 0, 1] = 1.0  # center element = 1 (identity)
        conv_weight[:, 0, 0] = 0.1  # slight left context
        conv_weight[:, 0, 2] = 0.1  # slight right context

    # Compute weight transfer statistics
    metrics["v_proj_shape"] = list(v_weight.shape)
    metrics["q_proj_shape"] = list(q_weight.shape)
    metrics["k_proj_shape"] = list(k_weight.shape)
    metrics["o_proj_shape"] = list(o_weight.shape)
    metrics["ssm_d_model"] = d_model
    metrics["ssm_d_inner"] = d_inner
    metrics["ssm_d_state"] = d_state

    return metrics


# ---------------------------------------------------------------------------
# FFN → MoE block conversion
# ---------------------------------------------------------------------------

def convert_ffn_to_moe(
    mlp_layer: nn.Module,
    moe_block: nn.Module,
    config: ConversionConfig,
    layer_idx: int = 0,
) -> dict[str, Any]:
    """
    Initialize a SparseMoELayer from a transformer FFN (MLP) layer.

    Strategies:
        - replicate: Copy FFN weights to all experts (diversify during training)
        - svd_split: Decompose FFN with SVD and assign components to experts
        - random: Keep random initialization

    Args:
        mlp_layer: Transformer MLP with gate_proj, up_proj, down_proj.
        moe_block: Target SparseMoELayer to initialize.
        config: Conversion configuration.
        layer_idx: Layer index for logging.

    Returns:
        Dict with conversion metrics.
    """
    metrics: dict[str, Any] = {"layer_idx": layer_idx, "type": "ffn->moe"}

    num_experts = moe_block.num_experts
    d_model = moe_block.d_model

    with torch.no_grad():
        # Teacher FFN has: gate_proj (d→4d), up_proj (d→4d), down_proj (4d→d)
        # Gated FFN: output = down_proj(act(gate_proj(x)) * up_proj(x))
        #
        # Student experts have: w1 (d→d_ff), w2 (d_ff→d)
        # Expert: output = w2(act(w1(x)))

        # Get teacher weights
        gate_w = mlp_layer.gate_proj.weight.data  # (intermediate, hidden)
        up_w = mlp_layer.up_proj.weight.data      # (intermediate, hidden)
        down_w = mlp_layer.down_proj.weight.data   # (hidden, intermediate)

        teacher_intermediate = gate_w.shape[0]
        student_d_ff = moe_block.experts[0].w1.weight.shape[0]
        student_d_model = moe_block.experts[0].w1.weight.shape[1]

        logger.info(
            "  [Layer %d] FFN→MoE: teacher(%d,%d) → %d experts(%d,%d)",
            layer_idx, teacher_intermediate, gate_w.shape[1],
            num_experts, student_d_ff, student_d_model,
        )

        if config.ffn_to_moe == "replicate":
            # Initialize all experts from the same FFN weights
            # They'll diversify during training via router gradient
            for i, expert in enumerate(moe_block.experts):
                # Combine gate and up projections: w1 ≈ gate_proj
                expert.w1.weight.data = _extract_submatrix(
                    gate_w, student_d_ff, student_d_model
                )
                # w2 ≈ down_proj
                expert.w2.weight.data = _extract_submatrix(
                    down_w, student_d_model, student_d_ff
                )

                # Add small perturbation per expert to break symmetry
                expert.w1.weight.data += torch.randn_like(expert.w1.weight) * 0.01
                expert.w2.weight.data += torch.randn_like(expert.w2.weight) * 0.01

            metrics["init_strategy"] = "replicate_with_noise"

        elif config.ffn_to_moe == "svd_split":
            # Decompose the FFN into expert-sized components via SVD
            # Each expert gets a different "slice" of the FFN's capacity

            # For w1: split gate_proj across experts
            U, S, Vh = torch.linalg.svd(gate_w.float(), full_matrices=False)
            k_per_expert = min(student_d_ff, len(S) // num_experts)

            for i, expert in enumerate(moe_block.experts):
                start = i * k_per_expert
                end = start + k_per_expert

                U_i = U[:student_d_ff, start:end]
                S_i = S[start:end]
                Vh_i = Vh[start:end, :student_d_model]

                w1_init = (U_i @ torch.diag(S_i) @ Vh_i).to(gate_w.dtype)
                expert.w1.weight.data = w1_init

                # For w2: corresponding slice of down_proj
                U_d, S_d, Vh_d = torch.linalg.svd(
                    down_w.float(), full_matrices=False
                )
                U_d_i = U_d[:student_d_model, start:end]
                S_d_i = S_d[start:end]
                Vh_d_i = Vh_d[start:end, :student_d_ff]
                w2_init = (U_d_i @ torch.diag(S_d_i) @ Vh_d_i).to(down_w.dtype)
                expert.w2.weight.data = w2_init

            metrics["init_strategy"] = "svd_split"
            metrics["k_per_expert"] = k_per_expert

        else:
            # Random init — keep existing xavier initialization
            metrics["init_strategy"] = "random"

        # Initialize router to uniform routing
        nn.init.zeros_(moe_block.gate.weight)

        # Copy LayerNorm if dimensions match
        if hasattr(mlp_layer, "input_layernorm"):
            _transfer_layernorm(mlp_layer.input_layernorm, moe_block.norm)

    metrics["teacher_intermediate"] = teacher_intermediate
    metrics["student_d_ff"] = student_d_ff
    metrics["num_experts"] = num_experts

    return metrics


# ---------------------------------------------------------------------------
# LayerNorm transfer
# ---------------------------------------------------------------------------

def _transfer_layernorm(src: nn.Module, dst: nn.LayerNorm) -> None:
    """Transfer LayerNorm parameters, handling dimension mismatch."""
    if not hasattr(src, "weight"):
        return

    src_dim = src.weight.shape[0]
    dst_dim = dst.weight.shape[0]

    if src_dim == dst_dim:
        dst.weight.data.copy_(src.weight.data)
        if hasattr(src, "bias") and src.bias is not None and dst.bias is not None:
            dst.bias.data.copy_(src.bias.data)
    else:
        # Dimension mismatch — interpolate
        ratio = src_dim / dst_dim
        indices = (torch.arange(dst_dim).float() * ratio).long().clamp(max=src_dim - 1)
        dst.weight.data = src.weight.data[indices]
        if hasattr(src, "bias") and src.bias is not None and dst.bias is not None:
            dst.bias.data = src.bias.data[indices]


# ---------------------------------------------------------------------------
# Single-block distillation refinement
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_block_cka(
    teacher_block: nn.Module,
    student_block: nn.Module,
    sample_input: torch.Tensor,
    is_attn: bool = True,
) -> float:
    """Compute CKA between a teacher block and student block on sample data."""
    from .cka import linear_cka

    # Get teacher output
    teacher_block.eval()
    if is_attn:
        # For attention: output is the hidden state after the layer
        # We need to run the full layer including residual
        t_out = teacher_block(sample_input)[0] if isinstance(
            teacher_block(sample_input), tuple
        ) else teacher_block(sample_input)
    else:
        t_out = teacher_block(sample_input)
        if isinstance(t_out, tuple):
            t_out = t_out[0]

    # Get student output
    student_block.eval()
    s_out = student_block(sample_input)
    if isinstance(s_out, tuple):
        s_out = s_out[0]

    # Flatten to (n_tokens, d) for CKA
    t_flat = t_out.reshape(-1, t_out.shape[-1]).cpu().float()
    s_flat = s_out.reshape(-1, s_out.shape[-1]).cpu().float()

    # Handle dimension mismatch
    min_d = min(t_flat.shape[1], s_flat.shape[1])
    t_flat = t_flat[:, :min_d]
    s_flat = s_flat[:, :min_d]

    return linear_cka(t_flat, s_flat).item()


def refine_block(
    teacher_block: nn.Module,
    student_block: nn.Module,
    calibration_data: list[torch.Tensor],
    config: ConversionConfig,
    device: torch.device,
    layer_idx: int = 0,
) -> dict[str, Any]:
    """
    Refine a converted block via activation-matching distillation.

    Freezes the teacher block, trains the student block to minimize
    MSE + cosine loss between their outputs on calibration data.

    Args:
        teacher_block: Frozen teacher module.
        student_block: Student module to train.
        calibration_data: List of input tensors for calibration.
        config: Conversion config.
        device: Target device.
        layer_idx: For logging.

    Returns:
        Refinement metrics.
    """
    metrics: dict[str, Any] = {"layer_idx": layer_idx}

    teacher_block.eval()
    for p in teacher_block.parameters():
        p.requires_grad = False

    student_block.train()
    student_block.to(device)

    optimizer = torch.optim.AdamW(
        student_block.parameters(), lr=config.refinement_lr, weight_decay=0.01
    )

    cka_scores: list[float] = []
    losses: list[float] = []

    for step in range(config.refinement_steps):
        batch = calibration_data[step % len(calibration_data)].to(device)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = teacher_block(batch)
            if isinstance(t_out, tuple):
                t_out = t_out[0]

        # Student forward
        s_out = student_block(batch)
        if isinstance(s_out, tuple):
            s_out = s_out[0]

        # Handle dimension mismatch — project if needed
        if t_out.shape[-1] != s_out.shape[-1]:
            min_d = min(t_out.shape[-1], s_out.shape[-1])
            t_out = t_out[..., :min_d]
            s_out = s_out[..., :min_d]

        # MSE + cosine loss
        mse = F.mse_loss(s_out.float(), t_out.float())
        cos_sim = F.cosine_similarity(
            s_out.float().reshape(-1, s_out.shape[-1]),
            t_out.float().reshape(-1, t_out.shape[-1]),
            dim=-1,
        )
        cos_loss = 1.0 - cos_sim.mean()
        loss = mse + 0.5 * cos_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_block.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        # CKA check
        if step > 0 and step % config.cka_check_every == 0:
            cka = _compute_block_cka(
                teacher_block, student_block, batch
            )
            cka_scores.append(cka)

            status = "OK" if cka >= config.cka_threshold else "LOW"
            logger.info(
                "  [Layer %d, Step %d/%d] loss=%.4f cka=%.4f [%s]",
                layer_idx, step, config.refinement_steps, loss.item(), cka, status,
            )

            if cka < config.cka_threshold:
                logger.warning(
                    "  CKA below threshold (%.4f < %.4f) at layer %d step %d",
                    cka, config.cka_threshold, layer_idx, step,
                )

    metrics["final_loss"] = losses[-1] if losses else 0.0
    metrics["cka_scores"] = cka_scores
    metrics["final_cka"] = cka_scores[-1] if cka_scores else None

    return metrics


# ---------------------------------------------------------------------------
# Public API: convert a single block
# ---------------------------------------------------------------------------

def convert_block(
    teacher_layer: nn.Module,
    student_block: nn.Module,
    config: ConversionConfig,
    layer_idx: int = 0,
    block_type: str = "attn",
    calibration_data: Optional[list[torch.Tensor]] = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """
    Convert a single transformer block to its Mamba/MoE equivalent.

    Args:
        teacher_layer: The transformer layer component (attention or MLP).
        student_block: The target SSMBlock or SparseMoELayer.
        config: Conversion configuration.
        layer_idx: Layer index for logging.
        block_type: "attn" for attention→SSM, "ffn" for MLP→MoE.
        calibration_data: Required for "distillation" and "hybrid" strategies.
        device: Device for refinement.

    Returns:
        Conversion metrics dict.
    """
    all_metrics: dict[str, Any] = {
        "layer_idx": layer_idx,
        "block_type": block_type,
        "strategy": config.strategy,
    }

    # Step 1: Weight mapping (for "weight_map" and "hybrid")
    if config.strategy in ("weight_map", "hybrid"):
        if block_type == "attn":
            wm_metrics = convert_attention_to_ssm(
                teacher_layer, student_block, config, layer_idx
            )
        else:
            wm_metrics = convert_ffn_to_moe(
                teacher_layer, student_block, config, layer_idx
            )
        all_metrics["weight_map"] = wm_metrics
        logger.info(
            "  [Layer %d] Weight mapping complete (%s)",
            layer_idx, block_type,
        )

    # Step 2: Distillation refinement (for "distillation" and "hybrid")
    if config.strategy in ("distillation", "hybrid"):
        if calibration_data is None:
            logger.warning(
                "  [Layer %d] No calibration data — skipping refinement",
                layer_idx,
            )
        else:
            ref_metrics = refine_block(
                teacher_layer, student_block,
                calibration_data, config, device, layer_idx,
            )
            all_metrics["refinement"] = ref_metrics
            logger.info(
                "  [Layer %d] Refinement complete: final_cka=%s",
                layer_idx, ref_metrics.get("final_cka"),
            )

    return all_metrics
