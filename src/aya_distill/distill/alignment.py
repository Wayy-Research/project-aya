"""
Stage 1: Layer Alignment

Maps transformer attention layers to Mamba SSM blocks and transformer FFN
layers to MoE blocks, training the student to reproduce teacher intermediate
representations.

Loss: MSE + cosine similarity between aligned teacher/student activations.
Monitoring: CKA similarity per layer pair with threshold warnings.

The MambaInLlama insight: attention mechanisms and SSMs are both sequence
mixers. If we can align their hidden representations, the SSM learns a
compressed version of the attention pattern. This is lossy -- CKA < 0.75
signals excessive information loss and requires investigation.

References:
    Wang et al., "The Mamba in the Llama" (2024)
    Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces" (2023)

Wayy Research, 2024-2026.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .cka import linear_cka, MinibatchCKAAccumulator
from .hooks import (
    ActivationStore,
    build_layer_mapping,
    register_student_hooks,
    register_teacher_hooks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignmentConfig:
    """Immutable configuration for Stage 1 layer alignment."""

    # Loss
    loss_type: str = "mse+cosine"
    mse_weight: float = 1.0
    cosine_weight: float = 0.5

    # CKA monitoring
    cka_threshold: float = 0.75
    cka_check_every: int = 500
    cka_kernel: str = "linear"

    # Layer mapping
    layer_mapping_strategy: str = "interleaved"

    # Training
    lr: float = 1e-4
    warmup_steps: int = 500
    total_steps: int = 10000
    batch_size: int = 4
    max_seq_len: int = 512
    gradient_accumulation: int = 8
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "checkpoints/stage1_alignment"
    save_every: int = 1000
    log_every: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        valid_losses = {"mse", "cosine", "mse+cosine"}
        if self.loss_type not in valid_losses:
            raise ValueError(
                f"loss_type must be one of {valid_losses}, "
                f"got '{self.loss_type}'"
            )


# ---------------------------------------------------------------------------
# Projection layers for dimension mismatch
# ---------------------------------------------------------------------------

class DimensionProjector(nn.Module):
    """
    Linear projection to align teacher and student hidden dimensions.

    If d_teacher != d_student, we project teacher activations down (or up)
    to the student dimension before computing alignment loss. The projector
    is learned jointly during alignment training.
    """

    def __init__(self, d_teacher: int, d_student: int) -> None:
        super().__init__()
        self.needs_projection = d_teacher != d_student
        if self.needs_projection:
            self.proj = nn.Linear(d_teacher, d_student, bias=False)
            nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Alignment loss
# ---------------------------------------------------------------------------

def alignment_loss(
    teacher_act: torch.Tensor,
    student_act: torch.Tensor,
    loss_type: str = "mse+cosine",
    mse_weight: float = 1.0,
    cosine_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute alignment loss between teacher and student activations.

    Args:
        teacher_act: (n_tokens, d) teacher activations (projected if needed).
        student_act: (n_tokens, d) student activations.
        loss_type: one of "mse", "cosine", "mse+cosine".
        mse_weight: weight for MSE component.
        cosine_weight: weight for cosine similarity component.

    Returns:
        Scalar loss.
    """
    loss = torch.tensor(0.0, device=teacher_act.device, dtype=torch.float32)

    if "mse" in loss_type:
        mse = F.mse_loss(student_act.float(), teacher_act.float())
        loss = loss + mse_weight * mse

    if "cosine" in loss_type:
        # Cosine similarity loss: 1 - cos_sim (so lower = more similar)
        cos_sim = F.cosine_similarity(
            student_act.float(), teacher_act.float(), dim=-1
        )
        cosine_loss = 1.0 - cos_sim.mean()
        loss = loss + cosine_weight * cosine_loss

    return loss


# ---------------------------------------------------------------------------
# Layer Alignment Trainer
# ---------------------------------------------------------------------------

class LayerAlignmentTrainer:
    """
    Stage 1 trainer: align student layer representations to teacher.

    Process:
        1. Forward pass through teacher (no grad) to get activations.
        2. Forward pass through student to get activations.
        3. Compute alignment loss per mapped layer pair.
        4. Periodically compute CKA to monitor representational similarity.
        5. Warn if CKA drops below threshold.

    The teacher is frozen. Only the student and projection layers are trained.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: AlignmentConfig,
        n_teacher_layers: int,
        n_student_layers: int,
        d_teacher: int,
        d_student: int,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.config = config
        self.device = device

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Build layer mapping
        self.layer_mapping = build_layer_mapping(
            n_teacher_layers,
            n_student_layers,
            strategy=config.layer_mapping_strategy,
        )
        logger.info(
            f"Layer mapping ({len(self.layer_mapping)} pairs): "
            f"{[(t, s, c) for t, s, c in self.layer_mapping[:4]]}..."
        )

        # Create projection layers for each mapped pair
        self.projectors = nn.ModuleDict()
        for t_idx, s_idx, component in self.layer_mapping:
            key = f"{t_idx}_{s_idx}_{component.replace('->', '_')}"
            self.projectors[key] = DimensionProjector(d_teacher, d_student)
        self.projectors = self.projectors.to(device)

        # CKA accumulators per layer pair
        self._cka_accumulators: dict[str, MinibatchCKAAccumulator] = {}
        for t_idx, s_idx, component in self.layer_mapping:
            key = f"{t_idx}_{s_idx}"
            self._cka_accumulators[key] = MinibatchCKAAccumulator(
                d_x=d_teacher, d_y=d_student, device="cpu"
            )

        # Metrics tracking
        self.cka_history: dict[str, list[float]] = {
            f"{t}_{s}": [] for t, s, _ in self.layer_mapping
        }
        self.loss_history: list[float] = []
        self.step = 0

    def _setup_optimizer(self) -> tuple[AdamW, Any]:
        """Create optimizer and LR scheduler."""
        # Only optimize student parameters and projectors
        params = list(self.student.parameters()) + list(
            self.projectors.parameters()
        )
        optimizer = AdamW(params, lr=self.config.lr, weight_decay=0.01)

        # Linear warmup -> cosine decay
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        decay = CosineAnnealingLR(
            optimizer,
            T_max=self.config.total_steps - self.config.warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[self.config.warmup_steps],
        )
        return optimizer, scheduler

    def _get_teacher_key(
        self, t_idx: int, component: str
    ) -> str:
        """Map component type to teacher hook key."""
        if component == "attn->ssm":
            return f"teacher.attn.{t_idx}"
        elif component == "ffn->moe":
            return f"teacher.ffn.{t_idx}"
        else:
            raise ValueError(f"Unknown component: {component}")

    def _get_student_key(self, s_idx: int) -> str:
        """Map student index to hook key."""
        layer_type = "ssm" if s_idx % 2 == 0 else "moe"
        return f"student.{layer_type}.{s_idx}"

    @torch.no_grad()
    def compute_cka_scores(
        self,
        teacher_acts: dict[str, torch.Tensor],
        student_acts: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Compute CKA scores for all layer pairs and emit warnings.

        Returns:
            {pair_key: cka_score} dict.
        """
        scores: dict[str, float] = {}

        for t_idx, s_idx, component in self.layer_mapping:
            t_key = self._get_teacher_key(t_idx, component)
            s_key = self._get_student_key(s_idx)

            if t_key not in teacher_acts or s_key not in student_acts:
                continue

            t_act = teacher_acts[t_key]
            s_act = student_acts[s_key]

            # Ensure same number of tokens
            n = min(t_act.shape[0], s_act.shape[0])
            t_act = t_act[:n]
            s_act = s_act[:n]

            pair_key = f"{t_idx}_{s_idx}"
            cka_score = linear_cka(
                t_act.cpu().float(), s_act.cpu().float()
            ).item()

            scores[pair_key] = cka_score
            self.cka_history[pair_key].append(cka_score)

            # Threshold warning
            if cka_score < self.config.cka_threshold:
                warnings.warn(
                    f"CKA below threshold at step {self.step}: "
                    f"layer pair ({t_idx}->{s_idx}, {component}) "
                    f"CKA={cka_score:.4f} < {self.config.cka_threshold}. "
                    f"Investigate information loss in this layer mapping.",
                    stacklevel=2,
                )
                logger.warning(
                    f"[Step {self.step}] CKA BELOW THRESHOLD: "
                    f"{component} ({t_idx}->{s_idx}) = {cka_score:.4f}"
                )

        return scores

    def compute_step_loss(
        self,
        teacher_acts: dict[str, torch.Tensor],
        student_acts: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute total alignment loss across all layer pairs.

        Teacher activations are projected to student dimension before
        loss computation.
        """
        total_loss = torch.tensor(
            0.0, device=self.device, dtype=torch.float32
        )
        n_pairs = 0

        for t_idx, s_idx, component in self.layer_mapping:
            t_key = self._get_teacher_key(t_idx, component)
            s_key = self._get_student_key(s_idx)

            if t_key not in teacher_acts or s_key not in student_acts:
                continue

            t_act = teacher_acts[t_key].to(self.device)
            s_act = student_acts[s_key].to(self.device)

            # Align token counts
            n = min(t_act.shape[0], s_act.shape[0])
            t_act = t_act[:n]
            s_act = s_act[:n]

            # Project teacher to student dimension
            proj_key = f"{t_idx}_{s_idx}_{component.replace('->', '_')}"
            t_projected = self.projectors[proj_key](t_act.float())

            pair_loss = alignment_loss(
                t_projected,
                s_act.float(),
                loss_type=self.config.loss_type,
                mse_weight=self.config.mse_weight,
                cosine_weight=self.config.cosine_weight,
            )
            total_loss = total_loss + pair_loss
            n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss

    def train_step(
        self,
        input_ids: torch.Tensor,
        teacher_store: ActivationStore,
        student_store: ActivationStore,
    ) -> dict[str, float]:
        """
        Execute a single alignment training step.

        Args:
            input_ids: (batch, seq_len) token ids.
            teacher_store: hook store for teacher activations.
            student_store: hook store for student activations.

        Returns:
            Metrics dict with loss and optional CKA scores.
        """
        # Teacher forward (no grad)
        teacher_store.clear()
        with torch.no_grad():
            self.teacher(input_ids)
        teacher_acts = teacher_store.collect()

        # Student forward (with grad)
        student_store.clear()
        self.student.train()
        self.student(input_ids)
        student_acts = student_store.collect()

        # Compute loss
        loss = self.compute_step_loss(teacher_acts, student_acts)

        metrics: dict[str, float] = {"loss": loss.item()}

        # CKA check
        if (
            self.step > 0
            and self.step % self.config.cka_check_every == 0
        ):
            cka_scores = self.compute_cka_scores(
                teacher_acts, student_acts
            )
            metrics["cka_mean"] = (
                sum(cka_scores.values()) / len(cka_scores)
                if cka_scores
                else 0.0
            )
            for key, score in cka_scores.items():
                metrics[f"cka/{key}"] = score

        # Cleanup
        teacher_store.clear()
        student_store.clear()

        return loss, metrics

    def train(
        self,
        dataloader: Any,
        resume_step: int = 0,
    ) -> dict[str, Any]:
        """
        Full Stage 1 training loop.

        Args:
            dataloader: yields (input_ids, labels) or input_ids tensors.
            resume_step: step to resume from (for checkpoint continuation).

        Returns:
            Final metrics dict.
        """
        torch.manual_seed(self.config.seed)

        optimizer, scheduler = self._setup_optimizer()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.student.to(self.device)
        self.teacher.to(self.device)

        # Register hooks
        teacher_store = ActivationStore(detach=True, device=self.device)
        student_store = ActivationStore(detach=False, device=self.device)

        # Determine which teacher layers to hook based on mapping
        t_layer_indices = sorted(
            set(t for t, _, _ in self.layer_mapping)
        )
        s_layer_indices = sorted(
            set(s for _, s, _ in self.layer_mapping)
        )
        register_teacher_hooks(
            self.teacher, teacher_store, t_layer_indices
        )
        register_student_hooks(
            self.student, student_store, s_layer_indices
        )

        self.step = resume_step
        accum_loss = 0.0
        best_cka_mean = 0.0

        logger.info(
            f"Starting Stage 1 alignment training: "
            f"{self.config.total_steps} steps, "
            f"lr={self.config.lr}, "
            f"accumulation={self.config.gradient_accumulation}"
        )

        optimizer.zero_grad()

        for batch in dataloader:
            if self.step >= self.config.total_steps:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
            elif isinstance(batch, (tuple, list)):
                input_ids = batch[0].to(self.device)
            else:
                input_ids = batch.to(self.device)

            loss, metrics = self.train_step(
                input_ids, teacher_store, student_store
            )

            # Gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation
            scaled_loss.backward()
            accum_loss += loss.item()

            if (self.step + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.student.parameters())
                    + list(self.projectors.parameters()),
                    self.config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / self.config.gradient_accumulation
                self.loss_history.append(avg_loss)
                accum_loss = 0.0

            # Logging
            if self.step % self.config.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                log_msg = (
                    f"[Step {self.step}/{self.config.total_steps}] "
                    f"loss={metrics['loss']:.4f} lr={lr:.2e}"
                )
                if "cka_mean" in metrics:
                    log_msg += f" cka_mean={metrics['cka_mean']:.4f}"
                logger.info(log_msg)

            # Save checkpoint
            if (
                self.step > 0
                and self.step % self.config.save_every == 0
            ):
                self._save_checkpoint(output_dir / f"step_{self.step}.pt")

            # Track best CKA
            if "cka_mean" in metrics:
                if metrics["cka_mean"] > best_cka_mean:
                    best_cka_mean = metrics["cka_mean"]
                    self._save_checkpoint(output_dir / "best.pt")
                    logger.info(
                        f"New best CKA mean: {best_cka_mean:.4f} "
                        f"(saved best.pt)"
                    )

            self.step += 1

        # Final save
        self._save_checkpoint(output_dir / "final.pt")

        # Cleanup
        teacher_store.remove_hooks()
        student_store.remove_hooks()

        return {
            "final_step": self.step,
            "best_cka_mean": best_cka_mean,
            "loss_history": self.loss_history,
            "cka_history": self.cka_history,
        }

    def _save_checkpoint(self, path: Path) -> None:
        """Save student model and projector weights."""
        checkpoint = {
            "student_state_dict": self.student.state_dict(),
            "projectors_state_dict": self.projectors.state_dict(),
            "step": self.step,
            "cka_history": self.cka_history,
            "loss_history": self.loss_history,
            "config": {
                "loss_type": self.config.loss_type,
                "cka_threshold": self.config.cka_threshold,
                "seed": self.config.seed,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
