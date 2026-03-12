"""
Stage 2: KL Distillation

Soft-target training: the student learns to match the teacher's output
distribution (not just the argmax). Temperature scaling softens the
teacher distribution, exposing inter-token relationships that hard labels
discard.

Combined loss: alpha * KL(teacher || student) + (1-alpha) * CE(hard_labels)

The KL term teaches the student "what the teacher is confused about"
(the dark knowledge). The CE term anchors the student to correct outputs.

Per-language KL tracking ensures we don't sacrifice low-resource languages
for high-resource ones — a real risk in multilingual distillation.

References:
    Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
    Kim & Rush, "Sequence-Level Knowledge Distillation" (2016)

Wayy Research, 2024-2026.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KLDistillationConfig:
    """Immutable configuration for Stage 2 KL distillation."""

    # KL parameters
    temperature: float = 2.0
    alpha: float = 0.7  # Weight on KL vs CE

    # Training
    lr: float = 5e-5
    ssm_lr_multiplier: float = 1.0  # SSM blocks get lr * this (NB11: 10x)
    warmup_steps: int = 500
    total_steps: int = 20000
    batch_size: int = 4
    max_seq_len: int = 512
    gradient_accumulation: int = 8
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Vocab projection
    teacher_vocab_size: int = 256000
    student_vocab_size: int = 256000

    # Output
    output_dir: str = "checkpoints/stage2_kl"
    save_every: int = 2000
    log_every: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}"
            )


# ---------------------------------------------------------------------------
# Vocab projection for mismatched tokenizers
# ---------------------------------------------------------------------------

class VocabProjector(nn.Module):
    """
    Projects teacher logits to student vocab space when tokenizers differ.

    If teacher and student share the same tokenizer (same vocab_size),
    this is a no-op. Otherwise, learns a linear mapping.
    """

    def __init__(
        self,
        teacher_vocab: int,
        student_vocab: int,
    ) -> None:
        super().__init__()
        self.needs_projection = teacher_vocab != student_vocab

        if self.needs_projection:
            self.proj = nn.Linear(teacher_vocab, student_vocab, bias=False)
            # Initialize close to identity where possible
            with torch.no_grad():
                min_v = min(teacher_vocab, student_vocab)
                self.proj.weight[:min_v, :min_v] = torch.eye(min_v)
        else:
            self.proj = nn.Identity()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Project (batch, seq, teacher_vocab) -> (batch, seq, student_vocab)."""
        return self.proj(logits)


# ---------------------------------------------------------------------------
# KL divergence with temperature
# ---------------------------------------------------------------------------

def kl_divergence_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    KL(P_teacher || P_student) with temperature scaling.

    Temperature > 1 softens distributions, revealing dark knowledge.
    The T^2 scaling compensates for the softening effect on gradient
    magnitudes (Hinton et al., 2015).

    Args:
        teacher_logits: (batch, seq, vocab) raw logits from teacher.
        student_logits: (batch, seq, vocab) raw logits from student.
        temperature: softening temperature (typically 1-5).
        reduction: KL reduction mode.

    Returns:
        Scalar KL divergence loss, scaled by T^2.
    """
    teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL(P || Q) = sum(P * (log P - log Q))
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        log_target=True,
        reduction=reduction,
    )

    # Scale by T^2 to maintain gradient magnitude
    return kl * (temperature ** 2)


# ---------------------------------------------------------------------------
# KL Distillation Trainer
# ---------------------------------------------------------------------------

class KLDistillationTrainer:
    """
    Stage 2 trainer: KL divergence distillation with temperature scaling.

    Process per step:
        1. Teacher forward (no grad) -> soft logits
        2. Student forward -> student logits
        3. KL divergence on softened distributions
        4. Cross-entropy on hard labels (ground truth)
        5. Combined loss: alpha * KL + (1-alpha) * CE
        6. Track per-language KL for equity analysis

    The teacher is frozen. Only student (and optional vocab projector)
    are trained.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: KLDistillationConfig,
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

        # Vocab projector (if tokenizers differ)
        self.vocab_projector = VocabProjector(
            config.teacher_vocab_size,
            config.student_vocab_size,
        ).to(device)

        # Student loss function (for hard labels)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Per-language tracking
        self.per_language_kl: dict[str, list[float]] = defaultdict(list)
        self.loss_history: list[float] = []
        self.step = 0

    def _setup_optimizer(self) -> tuple[AdamW, Any]:
        """Create optimizer and LR scheduler for student + projector.

        When ssm_lr_multiplier > 1.0, SSM layers (even-indexed) get a
        boosted learning rate. NB11 showed this is essential: SSM blocks
        receive ~27x less gradient than MoE, so 10x LR compensates.
        """
        base_lr = self.config.lr
        ssm_mult = self.config.ssm_lr_multiplier

        if ssm_mult != 1.0 and hasattr(self.student, "layers"):
            # Per-component LR: SSM layers (even indices) get boosted
            n_layers = len(self.student.layers)
            ssm_indices = set(range(0, n_layers, 2))

            ssm_params = []
            other_params = []
            for name, param in self.student.named_parameters():
                if not param.requires_grad:
                    continue
                is_ssm = any(f"layers.{i}." in name for i in ssm_indices)
                if is_ssm:
                    ssm_params.append(param)
                else:
                    other_params.append(param)

            if self.vocab_projector.needs_projection:
                other_params += list(self.vocab_projector.parameters())

            param_groups = [
                {"params": ssm_params, "lr": base_lr * ssm_mult},
                {"params": other_params, "lr": base_lr},
            ]
            logger.info(
                "Per-component LR: %d SSM params (lr=%.2e), "
                "%d other params (lr=%.2e)",
                len(ssm_params), base_lr * ssm_mult,
                len(other_params), base_lr,
            )
        else:
            params = list(self.student.parameters())
            if self.vocab_projector.needs_projection:
                params += list(self.vocab_projector.parameters())
            param_groups = [{"params": params, "lr": base_lr}]

        optimizer = AdamW(param_groups, weight_decay=0.01)

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

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        language: Optional[str] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Single distillation training step.

        Args:
            input_ids: (batch, seq_len) token ids.
            labels: (batch, seq_len) target ids (-100 for masked).
            language: optional language tag for per-language tracking.

        Returns:
            (loss, metrics) tuple.
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(input_ids)
            if isinstance(teacher_out, dict):
                teacher_logits = teacher_out["logits"]
            elif hasattr(teacher_out, "logits"):
                # HuggingFace CausalLMOutput
                teacher_logits = teacher_out.logits
            else:
                teacher_logits = teacher_out

        # Project teacher logits if needed
        teacher_logits = self.vocab_projector(teacher_logits)

        # Student forward
        self.student.train()
        student_out = self.student(input_ids)
        if isinstance(student_out, dict):
            student_logits = student_out["logits"]
            aux_loss = student_out.get("aux_loss", torch.tensor(0.0))
        else:
            student_logits = student_out
            aux_loss = torch.tensor(0.0, device=self.device)

        # Shift logits and labels for autoregressive loss
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for loss computation
        vocab_size = shift_student.shape[-1]
        flat_student = shift_student.view(-1, vocab_size)
        flat_teacher = shift_teacher.view(-1, shift_teacher.shape[-1])
        flat_labels = shift_labels.view(-1)

        # Mask: only compute loss where labels != -100
        valid_mask = flat_labels != -100

        if valid_mask.any():
            valid_student = flat_student[valid_mask]
            valid_teacher = flat_teacher[valid_mask]
            valid_labels = flat_labels[valid_mask]

            # KL divergence on soft targets
            kl_loss = kl_divergence_loss(
                valid_teacher, valid_student, self.config.temperature
            )

            # Cross-entropy on hard labels
            ce_loss = self.ce_loss_fn(valid_student, valid_labels)

            # Combined loss
            alpha = self.config.alpha
            combined_loss = alpha * kl_loss + (1 - alpha) * ce_loss

            # Add MoE auxiliary loss (load balancing)
            combined_loss = combined_loss + 0.01 * aux_loss
        else:
            kl_loss = torch.tensor(0.0, device=self.device)
            ce_loss = torch.tensor(0.0, device=self.device)
            combined_loss = torch.tensor(0.0, device=self.device)

        metrics = {
            "loss": combined_loss.item(),
            "kl_loss": kl_loss.item(),
            "ce_loss": ce_loss.item(),
            "aux_loss": aux_loss.item(),
        }

        # Per-language tracking
        if language is not None:
            self.per_language_kl[language].append(kl_loss.item())
            metrics[f"kl/{language}"] = kl_loss.item()

        return combined_loss, metrics

    def train(
        self,
        dataloader: Any,
        resume_step: int = 0,
    ) -> dict[str, Any]:
        """
        Full Stage 2 training loop.

        Args:
            dataloader: yields dicts with input_ids, labels, and
                        optionally language.
            resume_step: step to resume from.

        Returns:
            Final metrics dict.
        """
        torch.manual_seed(self.config.seed)

        optimizer, scheduler = self._setup_optimizer()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.student.to(self.device)
        self.teacher.to(self.device)

        self.step = resume_step
        accum_loss = 0.0
        best_loss = float("inf")

        logger.info(
            f"Starting Stage 2 KL distillation: "
            f"{self.config.total_steps} steps, "
            f"T={self.config.temperature}, alpha={self.config.alpha}"
        )

        optimizer.zero_grad()

        for batch in dataloader:
            if self.step >= self.config.total_steps:
                break

            # Unpack batch
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                language = batch.get("language", None)
                # language may be a list for the batch; take first
                if isinstance(language, (list, tuple)):
                    language = language[0]
            elif isinstance(batch, (tuple, list)):
                input_ids = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                language = batch[2] if len(batch) > 2 else None
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            loss, metrics = self.train_step(input_ids, labels, language)

            # Gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation
            scaled_loss.backward()
            accum_loss += loss.item()

            if (self.step + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    self.config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / self.config.gradient_accumulation
                self.loss_history.append(avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save_checkpoint(output_dir / "best.pt")

                accum_loss = 0.0

            # Logging
            if self.step % self.config.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Step {self.step}/{self.config.total_steps}] "
                    f"loss={metrics['loss']:.4f} "
                    f"kl={metrics['kl_loss']:.4f} "
                    f"ce={metrics['ce_loss']:.4f} "
                    f"lr={lr:.2e}"
                )

            # Save checkpoint
            if (
                self.step > 0
                and self.step % self.config.save_every == 0
            ):
                self._save_checkpoint(
                    output_dir / f"step_{self.step}.pt"
                )
                self._log_language_summary()

            self.step += 1

        # Final save
        self._save_checkpoint(output_dir / "final.pt")
        self._log_language_summary()

        return {
            "final_step": self.step,
            "best_loss": best_loss,
            "loss_history": self.loss_history,
            "per_language_kl": dict(self.per_language_kl),
        }

    def _log_language_summary(self) -> None:
        """Log per-language KL divergence summary for equity analysis."""
        if not self.per_language_kl:
            return

        logger.info("=== Per-Language KL Divergence ===")
        for lang, scores in sorted(self.per_language_kl.items()):
            if scores:
                recent = scores[-100:]  # Last 100 steps
                mean_kl = sum(recent) / len(recent)
                logger.info(f"  {lang}: mean_kl={mean_kl:.4f} (n={len(scores)})")

    def _save_checkpoint(self, path: Path) -> None:
        """Save student model checkpoint."""
        checkpoint = {
            "student_state_dict": self.student.state_dict(),
            "step": self.step,
            "loss_history": self.loss_history,
            "per_language_kl": dict(self.per_language_kl),
            "config": {
                "temperature": self.config.temperature,
                "alpha": self.config.alpha,
                "seed": self.config.seed,
            },
        }
        if self.vocab_projector.needs_projection:
            checkpoint["vocab_projector_state_dict"] = (
                self.vocab_projector.state_dict()
            )
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
