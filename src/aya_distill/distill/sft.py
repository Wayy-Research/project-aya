"""
Stage 3: Supervised Fine-Tuning

Recovers multilingual capability and tool calling lost during distillation
stages 1-2. Uses language-balanced sampling and assistant-only loss masking
(matching the Aetheris data.py pattern).

Three recovery targets:
    1. Multilingual fluency (all 10 target languages)
    2. Tool calling (structured JSON generation)
    3. Chat format compliance (user/assistant turn structure)

Loss is computed ONLY on assistant tokens -- we don't train on user prompts
or system messages. This prevents the model from memorizing inputs and
forces it to learn the response distribution.

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SFTConfig:
    """Immutable configuration for Stage 3 supervised fine-tuning."""

    # Training
    lr: float = 2e-5
    warmup_steps: int = 200
    total_steps: int = 5000
    batch_size: int = 4
    max_seq_len: int = 1024
    gradient_accumulation: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Output
    output_dir: str = "checkpoints/stage3_sft"
    save_every: int = 500
    log_every: int = 25
    eval_every: int = 500
    seed: int = 42


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

ROLE_TAGS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "tool_call": "<tool_call>",
    "tool_result": "<tool_result>",
}


def format_chat(
    messages: list[dict[str, str]],
    eos_token: str = "<|endoftext|>",
) -> str:
    """
    Format a list of chat messages into a single string with role tags.

    Args:
        messages: list of {role, content} dicts.
        eos_token: end-of-sequence token.

    Returns:
        Formatted chat string.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tag = ROLE_TAGS.get(role, f"<|{role}|>")
        parts.append(f"{tag}{content}")
    parts.append(eos_token)
    return "".join(parts)


def find_assistant_spans(text: str) -> list[tuple[int, int]]:
    """
    Find character spans of assistant responses in formatted chat text.

    Mirrors the logic in aetheris/data.py _find_assistant_spans.

    Args:
        text: formatted chat string with role tags.

    Returns:
        List of (char_start, char_end) tuples for assistant content.
    """
    spans: list[tuple[int, int]] = []
    search_from = 0
    assistant_tag = "<|assistant|>"

    while True:
        start = text.find(assistant_tag, search_from)
        if start == -1:
            break

        content_start = start + len(assistant_tag)

        # End at next role tag or end of text
        end = len(text)
        for tag in [
            "<|user|>",
            "<|system|>",
            "<|tool|>",
            "<|endoftext|>",
        ]:
            pos = text.find(tag, content_start)
            if pos != -1:
                end = min(end, pos)

        spans.append((content_start, end))
        search_from = end

    return spans


def build_sft_labels(
    text: str,
    tokenizer: Any,
    max_seq_len: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenize chat text and build labels with assistant-only masking.

    Non-assistant tokens get label=-100 (ignored by CrossEntropyLoss).

    Args:
        text: formatted chat string.
        tokenizer: HuggingFace tokenizer.
        max_seq_len: maximum sequence length.

    Returns:
        (input_ids, labels) tensors, or None if text is too short.
    """
    if len(text) < 10:
        return None

    # Pre-truncate to avoid slow tokenization
    max_chars = max_seq_len * 5
    if len(text) > max_chars:
        text = text[:max_chars]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0]

    if len(input_ids) < 2:
        return None

    # Build labels: -100 everywhere except assistant spans
    labels = torch.full_like(input_ids, -100)
    assistant_spans = find_assistant_spans(text)

    for char_start, char_end in assistant_spans:
        in_span = False
        for tok_idx in range(len(input_ids)):
            token_span = enc.token_to_chars(0, tok_idx)
            if token_span is None:
                # Special token — include if neighbors are in span
                if in_span:
                    labels[tok_idx] = input_ids[tok_idx]
                continue
            tok_start, tok_end = token_span
            if tok_end > char_start and tok_start < char_end:
                labels[tok_idx] = input_ids[tok_idx]
                in_span = True
            else:
                in_span = False

    # Train on EOS token
    if input_ids[-1] == tokenizer.eos_token_id:
        labels[-1] = input_ids[-1]

    # Pad to max_seq_len
    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids = torch.cat([
            input_ids,
            torch.full(
                (pad_len,), tokenizer.pad_token_id, dtype=torch.long
            ),
        ])
        labels = torch.cat([
            labels,
            torch.full((pad_len,), -100, dtype=torch.long),
        ])

    return input_ids, labels


# ---------------------------------------------------------------------------
# SFT Trainer
# ---------------------------------------------------------------------------

class SFTTrainer:
    """
    Stage 3 trainer: supervised fine-tuning for capability recovery.

    After KL distillation, the student model may have degraded on:
        - Low-resource languages (imbalanced training data)
        - Tool calling (structured output generation)
        - Chat format adherence

    SFT with targeted, language-balanced data restores these capabilities.
    Loss is computed only on assistant tokens (label masking).
    """

    def __init__(
        self,
        student: nn.Module,
        config: SFTConfig,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.student = student
        self.config = config
        self.device = device

        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Per-language tracking
        self.per_language_loss: dict[str, list[float]] = defaultdict(list)
        self.loss_history: list[float] = []
        self.eval_history: list[dict[str, float]] = []
        self.step = 0

    def _setup_optimizer(self) -> tuple[AdamW, Any]:
        """Create optimizer and LR scheduler."""
        optimizer = AdamW(
            self.student.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

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
        Single SFT training step.

        Args:
            input_ids: (batch, seq_len) token ids.
            labels: (batch, seq_len) with -100 for non-assistant tokens.
            language: optional language tag for tracking.

        Returns:
            (loss, metrics) tuple.
        """
        self.student.train()
        output = self.student(input_ids, labels=labels)

        if isinstance(output, dict):
            loss = output["loss"]
            ce_loss = output.get("ce_loss", loss)
            aux_loss = output.get("aux_loss", torch.tensor(0.0))
        else:
            # Compute loss manually
            logits = output.logits if hasattr(output, "logits") else output
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.ce_loss_fn(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
            ce_loss = loss
            aux_loss = torch.tensor(0.0)

        metrics = {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "aux_loss": aux_loss.item(),
        }

        if language is not None:
            self.per_language_loss[language].append(ce_loss.item())
            metrics[f"loss/{language}"] = ce_loss.item()

        return loss, metrics

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataloader: Any,
        max_batches: int = 50,
    ) -> dict[str, float]:
        """
        Evaluate on held-out data.

        Args:
            eval_dataloader: validation data.
            max_batches: cap evaluation length.

        Returns:
            Metrics dict with eval_loss and per-language breakdown.
        """
        self.student.eval()
        total_loss = 0.0
        n_batches = 0
        lang_losses: dict[str, list[float]] = defaultdict(list)

        for batch in eval_dataloader:
            if n_batches >= max_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                language = batch.get("language", None)
                if isinstance(language, (list, tuple)):
                    language = language[0]
            else:
                input_ids = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                language = None

            output = self.student(input_ids, labels=labels)
            if isinstance(output, dict):
                loss = output["loss"]
            else:
                logits = output.logits if hasattr(output, "logits") else output
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.ce_loss_fn(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                )

            total_loss += loss.item()
            n_batches += 1

            if language is not None:
                lang_losses[language].append(loss.item())

        eval_loss = total_loss / max(n_batches, 1)
        metrics = {"eval_loss": eval_loss}

        for lang, losses in lang_losses.items():
            metrics[f"eval_loss/{lang}"] = sum(losses) / len(losses)

        return metrics

    def train(
        self,
        train_dataloader: Any,
        eval_dataloader: Optional[Any] = None,
        resume_step: int = 0,
    ) -> dict[str, Any]:
        """
        Full Stage 3 SFT training loop.

        Args:
            train_dataloader: training data (chat format, language-balanced).
            eval_dataloader: optional validation data.
            resume_step: step to resume from.

        Returns:
            Final metrics dict.
        """
        torch.manual_seed(self.config.seed)

        optimizer, scheduler = self._setup_optimizer()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.student.to(self.device)
        self.step = resume_step
        accum_loss = 0.0
        best_eval_loss = float("inf")

        logger.info(
            f"Starting Stage 3 SFT: "
            f"{self.config.total_steps} steps, lr={self.config.lr}"
        )

        optimizer.zero_grad()

        for batch in train_dataloader:
            if self.step >= self.config.total_steps:
                break

            # Unpack
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                language = batch.get("language", None)
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
                accum_loss = 0.0

            # Logging
            if self.step % self.config.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Step {self.step}/{self.config.total_steps}] "
                    f"loss={metrics['loss']:.4f} lr={lr:.2e}"
                )

            # Evaluation
            if (
                eval_dataloader is not None
                and self.step > 0
                and self.step % self.config.eval_every == 0
            ):
                eval_metrics = self.evaluate(eval_dataloader)
                self.eval_history.append(eval_metrics)
                logger.info(
                    f"[Eval Step {self.step}] "
                    f"eval_loss={eval_metrics['eval_loss']:.4f}"
                )

                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    self._save_checkpoint(output_dir / "best.pt")
                    logger.info(
                        f"New best eval loss: {best_eval_loss:.4f}"
                    )

            # Save checkpoint
            if (
                self.step > 0
                and self.step % self.config.save_every == 0
            ):
                self._save_checkpoint(
                    output_dir / f"step_{self.step}.pt"
                )

            self.step += 1

        # Final save
        self._save_checkpoint(output_dir / "final.pt")
        self._log_language_summary()

        return {
            "final_step": self.step,
            "best_eval_loss": best_eval_loss,
            "loss_history": self.loss_history,
            "eval_history": self.eval_history,
            "per_language_loss": dict(self.per_language_loss),
        }

    def _log_language_summary(self) -> None:
        """Log per-language loss summary."""
        if not self.per_language_loss:
            return

        logger.info("=== Per-Language SFT Loss ===")
        for lang, losses in sorted(self.per_language_loss.items()):
            if losses:
                recent = losses[-100:]
                mean_loss = sum(recent) / len(recent)
                logger.info(
                    f"  {lang}: mean_loss={mean_loss:.4f} "
                    f"(n={len(losses)})"
                )

    def _save_checkpoint(self, path: Path) -> None:
        """Save student model checkpoint."""
        checkpoint = {
            "student_state_dict": self.student.state_dict(),
            "step": self.step,
            "loss_history": self.loss_history,
            "eval_history": self.eval_history,
            "per_language_loss": dict(self.per_language_loss),
            "config": {
                "lr": self.config.lr,
                "seed": self.config.seed,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
