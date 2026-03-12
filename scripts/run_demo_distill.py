#!/usr/bin/env python3
"""
Unified demo distillation training script for Project Aya.

Runs all 3 distillation stages on CPU with synthetic data from fixtures.
No external dataset downloads required -- uses MULTILINGUAL_PROMPTS as
training data and self-distillation (student = teacher) for KL stage.

Stages:
    0: Initialize student model from demo config
    1: Layer alignment (CKA-guided, MSE+cosine loss)
    2: KL distillation (temperature-scaled, SSM 10x LR boost)
    3: Supervised fine-tuning (assistant-only masking)

Usage:
    python scripts/run_demo_distill.py
    python scripts/run_demo_distill.py --config configs/demo_distill.yaml

Wayy Research, 2024-2026.
"""
from __future__ import annotations

import copy
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, IterableDataset

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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("demo_distill")


# ---------------------------------------------------------------------------
# Simple byte-level tokenizer (no downloads required)
# ---------------------------------------------------------------------------
class SimpleTokenizer:
    """Byte-level tokenizer with a small vocab for demo purposes.

    Encodes text as UTF-8 bytes mapped to token IDs 0..255.
    Special tokens are assigned above the byte range.
    """

    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"

        # Special token string -> ID mapping
        self._special: dict[str, int] = {
            "<|pad|>": 0,
            "<|endoftext|>": 1,
            "<|bos|>": 2,
            "<|system|>": 3,
            "<|user|>": 4,
            "<|assistant|>": 5,
            "<|tool|>": 6,
            "<tool_call>": 7,
            "</tool_call>": 8,
            "<tool_result>": 9,
            "</tool_result>": 10,
        }
        self._byte_offset = 16  # bytes mapped to 16..271

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> Any:
        """Encode text to token IDs."""
        ids: list[int] = []

        # Replace special tokens first
        remaining = text
        while remaining:
            found = False
            for token_str, token_id in self._special.items():
                if remaining.startswith(token_str):
                    ids.append(token_id)
                    remaining = remaining[len(token_str):]
                    found = True
                    break
            if not found:
                byte_val = remaining[0].encode("utf-8")[0]
                ids.append((byte_val + self._byte_offset) % self.vocab_size)
                remaining = remaining[1:]

        if truncation and max_length is not None:
            ids = ids[:max_length]

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(
        self,
        ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        special_ids = set(self._special.values()) if skip_special_tokens else set()
        id_to_special = {v: k for k, v in self._special.items()}

        chars: list[str] = []
        for tid in ids:
            if tid in special_ids:
                continue
            if tid in id_to_special and not skip_special_tokens:
                chars.append(id_to_special[tid])
            elif self._byte_offset <= tid < self._byte_offset + 256:
                byte_val = tid - self._byte_offset
                try:
                    chars.append(bytes([byte_val]).decode("utf-8", errors="replace"))
                except Exception:
                    chars.append("?")
            else:
                chars.append("?")
        return "".join(chars)

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> dict[str, Any]:
        """HuggingFace-compatible __call__."""
        ids = self.encode(text, max_length=max_length, truncation=truncation)
        if return_tensors == "pt":
            input_ids = torch.tensor([ids], dtype=torch.long)
        else:
            input_ids = ids
        return {"input_ids": input_ids}

    def __len__(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# Synthetic demo dataset from fixtures
# ---------------------------------------------------------------------------
class SyntheticDemoDataset(IterableDataset):
    """Infinite-cycling dataset built from MULTILINGUAL_PROMPTS fixtures.

    For each language in the config, collects prompts across all categories,
    tokenizes them, and yields (input_ids, labels) pairs forever with
    shuffling. In SFT mode, wraps prompts in chat format.
    """

    def __init__(
        self,
        languages: list[str],
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 128,
        mode: str = "alignment",
        seed: int = 42,
    ) -> None:
        from aya_distill.testing.fixtures import MULTILINGUAL_PROMPTS

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.seed = seed

        # Collect all prompts for requested languages
        self.examples: list[dict[str, Any]] = []
        for category, lang_prompts in MULTILINGUAL_PROMPTS.items():
            for lang in languages:
                if lang in lang_prompts:
                    prompt = lang_prompts[lang]
                    self.examples.append({
                        "text": prompt,
                        "language": lang,
                        "category": category,
                    })

        if not self.examples:
            raise ValueError(
                f"No prompts found for languages {languages}. "
                "Check MULTILINGUAL_PROMPTS in fixtures.py."
            )
        logger.info(
            "SyntheticDemoDataset: %d examples across %d languages, mode=%s",
            len(self.examples), len(languages), mode,
        )

    def _format_sft(self, text: str) -> str:
        """Wrap prompt in chat format for SFT mode."""
        return (
            f"<|system|>You are a helpful assistant.<|user|>{text}"
            f"<|assistant|>This is a response about: {text[:50]}<|endoftext|>"
        )

    def _tokenize(
        self, text: str
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Tokenize text and create (input_ids, labels) pair."""
        ids = self.tokenizer.encode(text, truncation=True, max_length=self.max_seq_len)
        if len(ids) < 2:
            return None

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()

        # Pad to max_seq_len
        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long),
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=torch.long),
            ])

        return input_ids, labels

    def _tokenize_sft(
        self, text: str
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Tokenize with assistant-only masking for SFT."""
        formatted = self._format_sft(text)
        ids = self.tokenizer.encode(formatted, truncation=True, max_length=self.max_seq_len)
        if len(ids) < 2:
            return None

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)

        # Find assistant token (ID 5) and unmask everything after it
        asst_positions = (input_ids == self.tokenizer._special["<|assistant|>"]).nonzero(as_tuple=True)[0]
        if len(asst_positions) > 0:
            start = asst_positions[0].item() + 1
            labels[start:] = input_ids[start:]

        # Pad
        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long),
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=torch.long),
            ])

        return input_ids, labels

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield examples infinitely with shuffling."""
        rng = random.Random(self.seed)
        examples = list(self.examples)

        while True:
            rng.shuffle(examples)
            for ex in examples:
                if self.mode == "sft":
                    result = self._tokenize_sft(ex["text"])
                else:
                    result = self._tokenize(ex["text"])

                if result is None:
                    continue

                input_ids, labels = result
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "language": ex["language"],
                }


def create_demo_dataloader(
    languages: list[str],
    tokenizer: SimpleTokenizer,
    max_seq_len: int = 128,
    batch_size: int = 1,
    mode: str = "alignment",
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader from synthetic demo data."""
    dataset = SyntheticDemoDataset(
        languages=languages,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        mode=mode,
        seed=seed,
    )

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "language": [b["language"] for b in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load demo distillation config."""
    if path is None:
        path = str(PROJECT_ROOT / "configs" / "demo_distill.yaml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
def init_student(
    config: dict[str, Any],
    vocab_size: int,
) -> nn.Module:
    """Initialize the Aetheris student model."""
    from aetheris.config import AetherisConfig
    from aetheris.model import HybridMambaMoE

    config_path = config["student"]["config_path"]
    # Resolve relative to project root
    if not Path(config_path).is_absolute():
        config_path = str(PROJECT_ROOT / config_path)

    student_config = AetherisConfig.from_yaml(config_path)
    student_config.vocab_size = vocab_size
    # Force float32 and disable gradient checkpointing for CPU
    student_config.gradient_checkpointing = False

    model = HybridMambaMoE(student_config)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        "Student initialized: d_model=%d, n_layer=%d, vocab=%d, %.1fM params",
        student_config.d_model, student_config.n_layer,
        student_config.vocab_size, param_count,
    )
    return model


# ---------------------------------------------------------------------------
# Stage 0: Initialization (no conversion for demo -- random init is fine)
# ---------------------------------------------------------------------------
def run_stage0(
    config: dict[str, Any],
    tokenizer: SimpleTokenizer,
) -> nn.Module:
    """Stage 0: Initialize the student model.

    For demo purposes, we skip block conversion (which would require a
    real teacher model). The student starts from random initialization,
    which is sufficient to prove the pipeline works end-to-end.
    """
    logger.info("=== Stage 0: Model Initialization ===")
    student = init_student(config, vocab_size=tokenizer.vocab_size)

    output_dir = Path(PROJECT_ROOT / "checkpoints" / "demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"student_state_dict": student.state_dict(), "step": 0},
        output_dir / "stage0_init.pt",
    )
    logger.info("Stage 0 complete: saved initial weights")
    return student


# ---------------------------------------------------------------------------
# Stage 1: Layer Alignment (self-distillation with mock teacher)
# ---------------------------------------------------------------------------
def run_stage1(
    student: nn.Module,
    config: dict[str, Any],
    tokenizer: SimpleTokenizer,
    device: torch.device,
) -> nn.Module:
    """Stage 1: Layer alignment using self-distillation.

    Uses a frozen copy of the student as the 'teacher'. The student
    learns to align its own layer representations through the alignment
    trainer, proving the pipeline works without needing the real Aya model.
    """
    stage_cfg = config.get("stage1", {})
    if not stage_cfg.get("enabled", True):
        logger.info("Stage 1 disabled, skipping")
        return student

    logger.info("=== Stage 1: Layer Alignment ===")

    from aya_distill.distill.hooks import (
        ActivationStore,
        build_layer_mapping,
        register_student_hooks,
    )

    # Create a frozen copy as mock teacher
    mock_teacher = copy.deepcopy(student)
    mock_teacher.eval()
    for p in mock_teacher.parameters():
        p.requires_grad = False

    n_layers = len(student.layers)
    d_model = student.config.d_model

    # Build layer mapping (student -> student for self-distillation)
    layer_mapping = build_layer_mapping(n_layers, n_layers, strategy="interleaved")
    logger.info("Layer mapping: %d pairs", len(layer_mapping))

    # Create projection layers (identity since same dimensions)
    from aya_distill.distill.alignment import DimensionProjector, alignment_loss

    projectors = nn.ModuleDict()
    for t_idx, s_idx, component in layer_mapping:
        key = f"{t_idx}_{s_idx}_{component.replace('->', '_')}"
        projectors[key] = DimensionProjector(d_model, d_model)
    projectors = projectors.to(device)

    # Create data
    dataloader = create_demo_dataloader(
        languages=config["languages"],
        tokenizer=tokenizer,
        max_seq_len=stage_cfg.get("max_seq_len", 128),
        batch_size=stage_cfg.get("batch_size", 1),
        mode="alignment",
        seed=config.get("seed", 42),
    )

    # Setup optimizer
    total_steps = stage_cfg.get("total_steps", 200)
    warmup_steps = stage_cfg.get("warmup_steps", 20)
    lr = float(stage_cfg.get("lr", 1e-4))
    grad_accum = stage_cfg.get("gradient_accumulation", 4)
    log_every = 25

    params = list(student.parameters()) + list(projectors.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=0.01)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    decay = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])

    # Register hooks on both models
    teacher_store = ActivationStore(detach=True, device="cpu")
    student_store = ActivationStore(detach=False, device="cpu")
    register_student_hooks(mock_teacher, teacher_store)
    register_student_hooks(student, student_store)

    student.to(device)
    mock_teacher.to(device)

    optimizer.zero_grad()
    step = 0
    accum_loss = 0.0
    loss_history: list[float] = []
    t_start = time.time()

    for batch in dataloader:
        if step >= total_steps:
            break

        input_ids = batch["input_ids"].to(device)

        # Teacher forward
        teacher_store.clear()
        with torch.no_grad():
            mock_teacher(input_ids)
        teacher_acts = teacher_store.collect()

        # Student forward
        student_store.clear()
        student.train()
        student(input_ids)
        student_acts = student_store.collect()

        # Compute alignment loss
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        n_pairs = 0
        for t_idx, s_idx, component in layer_mapping:
            layer_type = "ssm" if s_idx % 2 == 0 else "moe"
            t_key = f"student.{layer_type}.{t_idx}"
            s_key = f"student.{layer_type}.{s_idx}"

            if t_key not in teacher_acts or s_key not in student_acts:
                continue

            t_act = teacher_acts[t_key].to(device)
            s_act = student_acts[s_key].to(device)
            n = min(t_act.shape[0], s_act.shape[0])
            t_act, s_act = t_act[:n], s_act[:n]

            proj_key = f"{t_idx}_{s_idx}_{component.replace('->', '_')}"
            t_projected = projectors[proj_key](t_act.float())

            pair_loss = alignment_loss(
                t_projected, s_act.float(),
                loss_type=stage_cfg.get("loss_type", "mse+cosine"),
            )
            total_loss = total_loss + pair_loss
            n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        # Gradient accumulation
        scaled = total_loss / grad_accum
        scaled.backward()
        accum_loss += total_loss.item()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            avg_loss = accum_loss / grad_accum
            loss_history.append(avg_loss)
            accum_loss = 0.0

        if step % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "[Stage 1][Step %d/%d] loss=%.4f lr=%.2e",
                step, total_steps, total_loss.item(), current_lr,
            )

        teacher_store.clear()
        student_store.clear()
        step += 1

    # Cleanup hooks
    teacher_store.remove_hooks()
    student_store.remove_hooks()

    elapsed = time.time() - t_start
    logger.info(
        "Stage 1 complete: %d steps in %.1fs (%.2f steps/s)",
        step, elapsed, step / max(elapsed, 0.01),
    )

    # Save checkpoint
    output_dir = Path(PROJECT_ROOT / stage_cfg.get("output_dir", "checkpoints/demo/stage1_alignment"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "step": step,
            "loss_history": loss_history,
        },
        output_dir / "final.pt",
    )
    logger.info("Stage 1 checkpoint saved to %s", output_dir / "final.pt")

    del mock_teacher
    return student


# ---------------------------------------------------------------------------
# Stage 2: KL Distillation with per-component LR
# ---------------------------------------------------------------------------
def run_stage2(
    student: nn.Module,
    config: dict[str, Any],
    tokenizer: SimpleTokenizer,
    device: torch.device,
) -> nn.Module:
    """Stage 2: KL distillation with SSM 10x LR boost.

    Uses self-distillation: a frozen snapshot of the student serves as
    the teacher. The student trains with KL divergence loss against the
    teacher's softened logits.

    Per-component LR: SSM layers (even indices) get 10x the base LR,
    following NB11 findings that SSM blocks need faster adaptation.
    """
    stage_cfg = config.get("stage2", {})
    if not stage_cfg.get("enabled", True):
        logger.info("Stage 2 disabled, skipping")
        return student

    logger.info("=== Stage 2: KL Distillation ===")

    from aya_distill.distill.kl_distillation import kl_divergence_loss

    # Freeze a snapshot as teacher
    mock_teacher = copy.deepcopy(student)
    mock_teacher.eval()
    for p in mock_teacher.parameters():
        p.requires_grad = False

    # Training params
    total_steps = stage_cfg.get("total_steps", 500)
    warmup_steps = stage_cfg.get("warmup_steps", 50)
    base_lr = float(stage_cfg.get("lr", 3e-4))
    ssm_lr_mult = float(stage_cfg.get("ssm_lr_multiplier", 10.0))
    grad_accum = stage_cfg.get("gradient_accumulation", 4)
    temperature = float(stage_cfg.get("temperature", 2.0))
    alpha = float(stage_cfg.get("alpha", 0.7))
    log_every = 25

    # Per-component LR: SSM layers get boosted LR
    n_layer = len(student.layers)
    ssm_layer_indices = set(range(0, n_layer, 2))

    ssm_params = []
    other_params = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        is_ssm = any(f"layers.{i}." in name for i in ssm_layer_indices)
        if is_ssm:
            ssm_params.append(param)
        else:
            other_params.append(param)

    logger.info(
        "Per-component LR: %d SSM params (lr=%.2e), %d other params (lr=%.2e)",
        len(ssm_params), base_lr * ssm_lr_mult,
        len(other_params), base_lr,
    )

    optimizer = AdamW([
        {"params": ssm_params, "lr": base_lr * ssm_lr_mult},
        {"params": other_params, "lr": base_lr},
    ], weight_decay=0.01)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    decay = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Data
    dataloader = create_demo_dataloader(
        languages=config["languages"],
        tokenizer=tokenizer,
        max_seq_len=stage_cfg.get("max_seq_len", 128),
        batch_size=stage_cfg.get("batch_size", 1),
        mode="distillation",
        seed=config.get("seed", 42),
    )

    student.to(device)
    mock_teacher.to(device)

    optimizer.zero_grad()
    step = 0
    accum_loss = 0.0
    loss_history: list[float] = []
    per_language_kl: dict[str, list[float]] = {}
    t_start = time.time()

    for batch in dataloader:
        if step >= total_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        language = batch["language"][0] if isinstance(batch["language"], list) else batch["language"]

        # Teacher forward
        with torch.no_grad():
            teacher_out = mock_teacher(input_ids)
            teacher_logits = teacher_out["logits"]

        # Student forward
        student.train()
        student_out = student(input_ids)
        student_logits = student_out["logits"]
        aux_loss = student_out.get("aux_loss", torch.tensor(0.0, device=device))

        # Shift for autoregressive
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_student.shape[-1]
        flat_student = shift_student.view(-1, vocab_size)
        flat_teacher = shift_teacher.view(-1, shift_teacher.shape[-1])
        flat_labels = shift_labels.view(-1)

        valid_mask = flat_labels != -100
        if valid_mask.any():
            valid_student = flat_student[valid_mask]
            valid_teacher = flat_teacher[valid_mask]
            valid_labels = flat_labels[valid_mask]

            kl_loss = kl_divergence_loss(valid_teacher, valid_student, temperature)
            ce_loss = ce_loss_fn(valid_student, valid_labels)
            combined = alpha * kl_loss + (1 - alpha) * ce_loss + 0.01 * aux_loss
        else:
            kl_loss = torch.tensor(0.0, device=device)
            ce_loss = torch.tensor(0.0, device=device)
            combined = torch.tensor(0.0, device=device)

        # Track per-language KL
        per_language_kl.setdefault(language, []).append(kl_loss.item())

        # Gradient accumulation
        scaled = combined / grad_accum
        scaled.backward()
        accum_loss += combined.item()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            avg_loss = accum_loss / grad_accum
            loss_history.append(avg_loss)
            accum_loss = 0.0

        if step % log_every == 0:
            ssm_lr = optimizer.param_groups[0]["lr"]
            other_lr = optimizer.param_groups[1]["lr"]
            logger.info(
                "[Stage 2][Step %d/%d] loss=%.4f kl=%.4f ce=%.4f "
                "ssm_lr=%.2e other_lr=%.2e",
                step, total_steps, combined.item(),
                kl_loss.item(), ce_loss.item(), ssm_lr, other_lr,
            )

        step += 1

    elapsed = time.time() - t_start
    logger.info(
        "Stage 2 complete: %d steps in %.1fs (%.2f steps/s)",
        step, elapsed, step / max(elapsed, 0.01),
    )

    # Log per-language KL summary
    logger.info("=== Per-Language KL Divergence ===")
    for lang, scores in sorted(per_language_kl.items()):
        if scores:
            recent = scores[-50:]
            mean_kl = sum(recent) / len(recent)
            logger.info("  %s: mean_kl=%.4f (n=%d)", lang, mean_kl, len(scores))

    # Save checkpoint
    output_dir = Path(PROJECT_ROOT / stage_cfg.get("output_dir", "checkpoints/demo/stage2_kl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "step": step,
            "loss_history": loss_history,
            "per_language_kl": per_language_kl,
            "config": {"temperature": temperature, "alpha": alpha},
        },
        output_dir / "final.pt",
    )
    logger.info("Stage 2 checkpoint saved to %s", output_dir / "final.pt")

    del mock_teacher
    return student


# ---------------------------------------------------------------------------
# Stage 3: Supervised Fine-Tuning
# ---------------------------------------------------------------------------
def run_stage3(
    student: nn.Module,
    config: dict[str, Any],
    tokenizer: SimpleTokenizer,
    device: torch.device,
) -> nn.Module:
    """Stage 3: SFT with assistant-only masking.

    Uses chat-formatted synthetic data with loss computed only on
    assistant tokens.
    """
    stage_cfg = config.get("stage3", {})
    if not stage_cfg.get("enabled", True):
        logger.info("Stage 3 disabled, skipping")
        return student

    logger.info("=== Stage 3: Supervised Fine-Tuning ===")

    # Training params
    total_steps = stage_cfg.get("total_steps", 200)
    warmup_steps = stage_cfg.get("warmup_steps", 20)
    lr = float(stage_cfg.get("lr", 2e-5))
    grad_accum = stage_cfg.get("gradient_accumulation", 4)
    log_every = 25

    # Data in SFT mode
    dataloader = create_demo_dataloader(
        languages=config["languages"],
        tokenizer=tokenizer,
        max_seq_len=stage_cfg.get("max_seq_len", 128),
        batch_size=stage_cfg.get("batch_size", 1),
        mode="sft",
        seed=config.get("seed", 42),
    )

    # Optimizer
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    decay = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])

    student.to(device)
    optimizer.zero_grad()
    step = 0
    accum_loss = 0.0
    loss_history: list[float] = []
    per_language_loss: dict[str, list[float]] = {}
    t_start = time.time()

    for batch in dataloader:
        if step >= total_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        language = batch["language"][0] if isinstance(batch["language"], list) else batch["language"]

        student.train()
        output = student(input_ids, labels=labels)
        loss = output["loss"]
        ce_loss = output.get("ce_loss", loss)

        per_language_loss.setdefault(language, []).append(ce_loss.item())

        # Gradient accumulation
        scaled = loss / grad_accum
        scaled.backward()
        accum_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            avg_loss = accum_loss / grad_accum
            loss_history.append(avg_loss)
            accum_loss = 0.0

        if step % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "[Stage 3][Step %d/%d] loss=%.4f ce=%.4f lr=%.2e",
                step, total_steps, loss.item(), ce_loss.item(), current_lr,
            )

        step += 1

    elapsed = time.time() - t_start
    logger.info(
        "Stage 3 complete: %d steps in %.1fs (%.2f steps/s)",
        step, elapsed, step / max(elapsed, 0.01),
    )

    # Per-language summary
    logger.info("=== Per-Language SFT Loss ===")
    for lang, losses in sorted(per_language_loss.items()):
        if losses:
            recent = losses[-50:]
            mean_loss = sum(recent) / len(recent)
            logger.info("  %s: mean_loss=%.4f (n=%d)", lang, mean_loss, len(losses))

    # Save checkpoint
    output_dir = Path(PROJECT_ROOT / stage_cfg.get("output_dir", "checkpoints/demo/stage3_sft"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "step": step,
            "loss_history": loss_history,
            "per_language_loss": per_language_loss,
        },
        output_dir / "final.pt",
    )
    logger.info("Stage 3 checkpoint saved to %s", output_dir / "final.pt")

    return student


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full demo distillation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Demo Distillation Pipeline")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: configs/demo_distill.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cpu")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("  Project Aya -- Demo Distillation Pipeline")
    logger.info("  Device: %s", device)
    logger.info("  Languages: %s", config["languages"])
    logger.info("=" * 60)

    t_total = time.time()

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=8192)
    logger.info("SimpleTokenizer initialized: vocab_size=%d", tokenizer.vocab_size)

    # Stage 0: Initialize
    student = run_stage0(config, tokenizer)

    # Stage 1: Alignment
    student = run_stage1(student, config, tokenizer, device)

    # Stage 2: KL Distillation
    student = run_stage2(student, config, tokenizer, device)

    # Stage 3: SFT
    student = run_stage3(student, config, tokenizer, device)

    # Save final model
    final_dir = Path(PROJECT_ROOT / "checkpoints" / "demo")
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "config": config,
            "vocab_size": tokenizer.vocab_size,
        },
        final_dir / "final_model.pt",
    )

    total_time = time.time() - t_total
    logger.info("=" * 60)
    logger.info("  Demo distillation complete!")
    logger.info("  Total time: %.1fs (%.1f minutes)", total_time, total_time / 60)
    logger.info("  Final model: %s", final_dir / "final_model.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
