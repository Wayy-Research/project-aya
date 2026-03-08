"""
Multilingual distillation data pipeline.

Handles three data modes across the distillation stages:
    - alignment: raw multilingual text for layer alignment (Stage 1)
    - distillation: text + teacher logit caching for KL (Stage 2)
    - sft: chat-format data for supervised fine-tuning (Stage 3)

Key design decisions:
    - Language-balanced sampling: equal representation across 10 languages
    - Streaming from HuggingFace datasets: no full dataset materialization
    - Tokenizer strategy: use the student tokenizer (Aya's tokenizer with
      256k vocab) for both models, avoiding lossy token projection

Target languages: en, es, hi, zh, ar, sw, tr, ja, id, te

Wayy Research, 2024-2026.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset, Sampler

logger = logging.getLogger(__name__)

# Target languages for Aya distillation
TARGET_LANGUAGES = ["en", "es", "hi", "zh", "ar", "sw", "tr", "ja", "id", "te"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DistillDataConfig:
    """Configuration for the distillation data pipeline."""

    mode: str = "alignment"  # alignment | distillation | sft
    languages: list[str] = field(default_factory=lambda: list(TARGET_LANGUAGES))
    max_seq_len: int = 512
    batch_size: int = 4
    buffer_size: int = 200
    seed: int = 42

    # HuggingFace dataset settings
    dataset_name: Optional[str] = None
    dataset_split: str = "train"
    hf_token: Optional[str] = None
    streaming: bool = True

    # Local data paths (alternative to HF datasets)
    local_data_dir: Optional[str] = None

    # Teacher logit caching (Stage 2)
    cache_teacher_logits: bool = False
    logit_cache_dir: Optional[str] = None
    logit_top_k: int = 64  # Only cache top-k logits to save disk

    def __post_init__(self) -> None:
        valid_modes = {"alignment", "distillation", "sft"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}, got '{self.mode}'"
            )
        for lang in self.languages:
            if lang not in TARGET_LANGUAGES:
                logger.warning(
                    f"Language '{lang}' not in target set. "
                    f"Expected: {TARGET_LANGUAGES}"
                )


# ---------------------------------------------------------------------------
# Tokenizer handling
# ---------------------------------------------------------------------------

def get_shared_tokenizer(
    tokenizer_name: str = "CohereForAI/aya-expanse-8b",
    add_special_tokens: bool = True,
) -> Any:
    """
    Load the shared tokenizer for distillation.

    Strategy: use the teacher's (Aya) tokenizer for both models. The
    student's embedding layer is resized to match. This avoids lossy
    token-level projection between different vocabularies.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier.
        add_special_tokens: whether to add tool/chat special tokens.

    Returns:
        HuggingFace tokenizer instance.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if add_special_tokens:
        special_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<tool_result>",
            "</tool_result>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|tool|>",
            "<|endoftext|>",
        ]
        # Only add tokens not already in vocab
        new_tokens = [
            t for t in special_tokens
            if t not in tokenizer.get_vocab()
        ]
        if new_tokens:
            num_added = tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens}
            )
            logger.info(
                f"Added {num_added} special tokens, "
                f"vocab_size={len(tokenizer)}"
            )

    return tokenizer


# ---------------------------------------------------------------------------
# Language-balanced batch sampler
# ---------------------------------------------------------------------------

class LanguageBalancedSampler:
    """
    Yields language tags in round-robin order to ensure equal
    representation across languages in each batch.

    For streaming datasets, this controls which language source to
    draw from next. Each batch contains examples from a single language,
    and languages rotate every batch.
    """

    def __init__(
        self,
        languages: list[str],
        seed: int = 42,
    ) -> None:
        self.languages = list(languages)
        self.rng = random.Random(seed)
        self.rng.shuffle(self.languages)
        self._idx = 0

    def next_language(self) -> str:
        """Get next language in rotation."""
        lang = self.languages[self._idx % len(self.languages)]
        self._idx += 1
        return lang

    def reset(self) -> None:
        self._idx = 0
        self.rng.shuffle(self.languages)


# ---------------------------------------------------------------------------
# Multilingual streaming dataset
# ---------------------------------------------------------------------------

class MultilingualDistillDataset(IterableDataset):
    """
    Streaming dataset for multilingual distillation.

    Supports three modes:
        - alignment: yields (input_ids,) for layer alignment
        - distillation: yields (input_ids, labels, language) for KL
        - sft: yields (input_ids, labels, language) with assistant masking

    Language balancing is done via a round-robin sampler that cycles
    through per-language data iterators.
    """

    def __init__(
        self,
        config: DistillDataConfig,
        tokenizer: Any,
        per_language_datasets: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.per_language_datasets = per_language_datasets or {}
        self.sampler = LanguageBalancedSampler(
            config.languages, seed=config.seed
        )

    def _tokenize_text(
        self, text: str
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Tokenize raw text, returning (input_ids, labels)."""
        if len(text) < 10:
            return None

        max_chars = self.config.max_seq_len * 5
        if len(text) > max_chars:
            text = text[:max_chars]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]

        if len(input_ids) < 2:
            return None

        # For alignment/distillation: loss on all non-pad tokens
        labels = input_ids.clone()

        # Pad
        if len(input_ids) < self.config.max_seq_len:
            pad_len = self.config.max_seq_len - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full(
                    (pad_len,),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                ),
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=torch.long),
            ])

        return input_ids, labels

    def _prepare_sft_example(
        self, example: dict[str, Any]
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Prepare SFT example with assistant-only label masking."""
        from .sft import format_chat, build_sft_labels

        if "messages" in example:
            text = format_chat(
                example["messages"],
                eos_token=self.tokenizer.eos_token or "<|endoftext|>",
            )
        elif "text" in example:
            text = example["text"]
        else:
            return None

        return build_sft_labels(text, self.tokenizer, self.config.max_seq_len)

    def _get_text_from_example(self, example: dict[str, Any]) -> str:
        """Extract text from various dataset formats."""
        if "text" in example:
            return example["text"]
        if "content" in example:
            return example["content"]
        if "sentence" in example:
            return example["sentence"]
        if "messages" in example:
            # Concatenate message contents for alignment mode
            parts = [
                msg.get("content", "")
                for msg in example["messages"]
            ]
            return " ".join(parts)
        # Last resort: concatenate all string values
        texts = [str(v) for v in example.values() if isinstance(v, str)]
        return " ".join(texts)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield language-balanced examples."""
        # Build per-language iterators
        lang_iters: dict[str, Iterator] = {}
        for lang in self.config.languages:
            if lang in self.per_language_datasets:
                lang_iters[lang] = iter(self.per_language_datasets[lang])

        if not lang_iters:
            logger.warning(
                "No per-language datasets available. "
                "Falling back to unified iteration."
            )
            return

        buffer: list[dict[str, Any]] = []
        exhausted: set[str] = set()

        while len(exhausted) < len(lang_iters):
            lang = self.sampler.next_language()

            if lang in exhausted or lang not in lang_iters:
                continue

            try:
                example = next(lang_iters[lang])
            except StopIteration:
                exhausted.add(lang)
                continue

            # Process based on mode
            if self.config.mode == "sft":
                result = self._prepare_sft_example(example)
            else:
                text = self._get_text_from_example(example)
                result = self._tokenize_text(text)

            if result is None:
                continue

            input_ids, labels = result
            item = {
                "input_ids": input_ids,
                "labels": labels,
                "language": lang,
            }

            buffer.append(item)

            if len(buffer) >= self.config.buffer_size:
                random.shuffle(buffer)
                while len(buffer) > self.config.buffer_size // 2:
                    yield buffer.pop()

        # Yield remaining
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()


# ---------------------------------------------------------------------------
# Teacher logit caching (Stage 2 optimization)
# ---------------------------------------------------------------------------

class TeacherLogitCache:
    """
    Cache teacher logits to disk for Stage 2, avoiding repeated
    teacher forward passes.

    Only stores top-k logits per position to manage disk usage.
    A 256k vocab with 512 seq_len would be ~512MB per batch without
    top-k filtering.
    """

    def __init__(
        self,
        cache_dir: str,
        top_k: int = 64,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k

    def _cache_path(self, batch_idx: int) -> Path:
        return self.cache_dir / f"logits_{batch_idx:08d}.pt"

    def has_cache(self, batch_idx: int) -> bool:
        return self._cache_path(batch_idx).exists()

    @torch.no_grad()
    def save(
        self,
        batch_idx: int,
        logits: torch.Tensor,
    ) -> None:
        """
        Save top-k teacher logits for a batch.

        Args:
            batch_idx: batch index for cache key.
            logits: (batch, seq, vocab) teacher output logits.
        """
        # Keep only top-k values and indices
        top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
        torch.save(
            {
                "values": top_values.half().cpu(),
                "indices": top_indices.cpu(),
                "vocab_size": logits.shape[-1],
            },
            self._cache_path(batch_idx),
        )

    def load(
        self,
        batch_idx: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Load cached logits, reconstructing full vocab distribution.

        Missing positions get -inf (zero probability after softmax).

        Returns:
            (batch, seq, vocab) logits tensor.
        """
        data = torch.load(self._cache_path(batch_idx), map_location="cpu")
        values = data["values"].float()
        indices = data["indices"]
        vocab_size = data["vocab_size"]

        # Reconstruct sparse logits
        batch, seq, k = values.shape
        logits = torch.full(
            (batch, seq, vocab_size),
            float("-inf"),
            dtype=torch.float32,
        )
        logits.scatter_(-1, indices, values)
        return logits.to(device)


# ---------------------------------------------------------------------------
# Dataset loading utilities
# ---------------------------------------------------------------------------

def load_per_language_datasets(
    config: DistillDataConfig,
) -> dict[str, Any]:
    """
    Load per-language datasets from HuggingFace or local files.

    For HuggingFace: expects dataset with a 'language' or 'lang' column,
    or separate dataset configs per language.

    For local: expects {local_data_dir}/{lang}.jsonl files.

    Returns:
        {language_code: iterable_dataset} dict.
    """
    from datasets import load_dataset

    per_lang: dict[str, Any] = {}

    if config.local_data_dir is not None:
        # Local JSONL files
        data_dir = Path(config.local_data_dir)
        for lang in config.languages:
            path = data_dir / f"{lang}.jsonl"
            if path.exists():
                per_lang[lang] = _load_jsonl_stream(str(path))
                logger.info(f"Loaded local dataset: {path}")
            else:
                logger.warning(f"No data file for {lang}: {path}")
    elif config.dataset_name is not None:
        # HuggingFace dataset — try language-specific configs first
        for lang in config.languages:
            try:
                ds = load_dataset(
                    config.dataset_name,
                    lang,
                    split=config.dataset_split,
                    streaming=config.streaming,
                    trust_remote_code=True,
                    token=config.hf_token,
                )
                per_lang[lang] = ds
                logger.info(f"Loaded HF dataset config '{lang}'")
            except Exception:
                logger.debug(
                    f"No config '{lang}' for {config.dataset_name}, "
                    f"will try filtering."
                )

        # Fallback: load full dataset and filter by language column
        if not per_lang:
            logger.info(
                "Loading full dataset and filtering by language column..."
            )
            full_ds = load_dataset(
                config.dataset_name,
                split=config.dataset_split,
                streaming=config.streaming,
                trust_remote_code=True,
                token=config.hf_token,
            )
            for lang in config.languages:
                filtered = full_ds.filter(
                    lambda x: x.get("language", x.get("lang", "")) == lang
                )
                per_lang[lang] = filtered
    else:
        raise ValueError(
            "Must specify either dataset_name or local_data_dir"
        )

    loaded = list(per_lang.keys())
    missing = [l for l in config.languages if l not in per_lang]
    logger.info(
        f"Loaded {len(loaded)} languages: {loaded}. "
        f"Missing: {missing or 'none'}"
    )
    return per_lang


def _load_jsonl_stream(path: str) -> Any:
    """Load a local JSONL file as a streaming iterable."""
    from datasets import IterableDataset as HFIterableDataset

    def gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    return HFIterableDataset.from_generator(gen)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def create_distill_dataloader(
    config: DistillDataConfig,
    tokenizer: Any,
    per_language_datasets: Optional[dict[str, Any]] = None,
) -> DataLoader:
    """
    Create a DataLoader for distillation training.

    Args:
        config: data pipeline configuration.
        tokenizer: shared tokenizer for both models.
        per_language_datasets: pre-loaded per-language datasets
            (if None, will be loaded from config).

    Returns:
        PyTorch DataLoader yielding language-balanced batches.
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    if per_language_datasets is None:
        per_language_datasets = load_per_language_datasets(config)

    dataset = MultilingualDistillDataset(
        config=config,
        tokenizer=tokenizer,
        per_language_datasets=per_language_datasets,
    )

    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Collate into batched tensors, keeping language as metadata."""
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        languages = [b["language"] for b in batch]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "language": languages,
        }

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Streaming datasets don't support multi-worker
        pin_memory=True,
    )
