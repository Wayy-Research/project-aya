"""
ClimbMix data loader for distillation pre-training.

NVIDIA ClimbMix is a 400B-token dataset created via CLIMB (CLustering-based
Iterative Data Mixture Bootstrapping). It outperforms FineWeb-Edu, DCLM, and
Olmo datasets under equal token budgets.

The dataset ships pre-tokenized with GPT-2 tokenizer. For distillation into
Aetheris (which uses the Aya 256k tokenizer), we provide both:
    1. Pre-tokenized mode: use raw token IDs directly (GPT-2 vocab only)
    2. Detokenized mode: decode to text, re-tokenize with Aya tokenizer

Reference:
    Diao et al., "CLIMB: CLustering-based Iterative Data Mixture Bootstrapping
    for Language Model Pre-training" (arXiv:2504.13161)

    Karpathy nanochat result: GPT-2 quality in 2 hours on 8xH100 with ClimbMix
    https://x.com/karpathy/status/2029701092347630069

Dataset: https://huggingface.co/datasets/nvidia/ClimbMix
License: CC BY-NC 4.0

Wayy Research, 2024-2026.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

# GPT-2 tokenizer vocab size (ClimbMix is pre-tokenized with this)
GPT2_VOCAB_SIZE = 50257


@dataclass
class ClimbMixConfig:
    """Configuration for ClimbMix data loading."""

    # Dataset
    dataset_name: str = "nvidia/ClimbMix"
    streaming: bool = True
    max_seq_len: int = 1024
    batch_size: int = 4

    # Tokenizer mode
    # "pretokenized": use GPT-2 token IDs directly (fast, no re-encoding)
    # "retokenize": decode to text, re-tokenize with target tokenizer
    mode: str = "pretokenized"
    target_tokenizer_name: Optional[str] = None  # for retokenize mode

    # Filtering
    min_tokens: int = 32
    cluster_ids: Optional[list[int]] = None  # filter to specific clusters

    # Training
    seed: int = 42
    buffer_size: int = 500
    num_workers: int = 0  # streaming doesn't support multi-worker


class ClimbMixDataset(IterableDataset):
    """
    Streaming dataset for NVIDIA ClimbMix.

    ClimbMix rows have: {cluster_id: int, tokens: list[int], token_count: int}

    In pretokenized mode, we chunk/pad the token sequences directly.
    In retokenize mode, we decode with GPT-2 and re-encode with the target
    tokenizer (needed when the student model uses a different vocabulary).
    """

    def __init__(
        self,
        config: ClimbMixConfig,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer

        if config.mode == "retokenize" and tokenizer is None:
            raise ValueError(
                "retokenize mode requires a tokenizer. "
                "Pass the target tokenizer or use pretokenized mode."
            )

        # For retokenize mode, we need a GPT-2 tokenizer to decode
        self._gpt2_tokenizer: Optional[Any] = None

    def _get_gpt2_tokenizer(self) -> Any:
        """Lazy-load GPT-2 tokenizer for decoding."""
        if self._gpt2_tokenizer is None:
            from transformers import AutoTokenizer

            self._gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self._gpt2_tokenizer

    def _load_stream(self) -> Any:
        """Load ClimbMix as a streaming dataset from HuggingFace."""
        from datasets import load_dataset

        ds = load_dataset(
            self.config.dataset_name,
            split="train",
            streaming=self.config.streaming,
        )

        # Filter by cluster if specified
        if self.config.cluster_ids is not None:
            allowed = set(self.config.cluster_ids)
            ds = ds.filter(lambda x: x["cluster_id"] in allowed)

        # Filter by minimum length
        ds = ds.filter(lambda x: x["token_count"] >= self.config.min_tokens)

        return ds

    def _chunk_tokens(
        self, tokens: list[int]
    ) -> list[torch.Tensor]:
        """Split a token sequence into max_seq_len chunks."""
        seq_len = self.config.max_seq_len
        chunks = []
        for start in range(0, len(tokens) - 1, seq_len):
            chunk = tokens[start : start + seq_len + 1]  # +1 for labels
            if len(chunk) > seq_len // 2:  # skip tiny tail chunks
                chunks.append(torch.tensor(chunk, dtype=torch.long))
        return chunks

    def _process_pretokenized(
        self, example: dict[str, Any]
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Process a pre-tokenized ClimbMix row into training examples."""
        tokens = example["tokens"]
        chunks = self._chunk_tokens(tokens)

        for chunk in chunks:
            seq_len = self.config.max_seq_len

            # input_ids = chunk[:-1], labels = chunk[1:]
            if len(chunk) > seq_len:
                input_ids = chunk[:seq_len]
                labels = chunk[1 : seq_len + 1]
            else:
                input_ids = chunk[:-1]
                labels = chunk[1:]

            # Pad if needed
            if len(input_ids) < seq_len:
                pad_len = seq_len - len(input_ids)
                input_ids = torch.cat([
                    input_ids,
                    torch.zeros(pad_len, dtype=torch.long),
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }

    def _process_retokenized(
        self, example: dict[str, Any]
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Decode with GPT-2, re-tokenize with target tokenizer."""
        gpt2_tok = self._get_gpt2_tokenizer()
        tokens = example["tokens"]

        # Decode to text
        text = gpt2_tok.decode(tokens, skip_special_tokens=True)

        if len(text) < 20:
            return

        # Re-tokenize with target tokenizer
        enc = self.tokenizer(
            text,
            truncation=False,
            return_tensors=None,
        )
        new_tokens = enc["input_ids"]

        # Chunk the re-tokenized sequence
        chunks = self._chunk_tokens(new_tokens)
        for chunk in chunks:
            seq_len = self.config.max_seq_len

            if len(chunk) > seq_len:
                input_ids = chunk[:seq_len]
                labels = chunk[1 : seq_len + 1]
            else:
                input_ids = chunk[:-1]
                labels = chunk[1:]

            if len(input_ids) < seq_len:
                pad_len = seq_len - len(input_ids)
                pad_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else 0
                )
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), pad_id, dtype=torch.long),
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Stream ClimbMix examples with optional shuffling buffer."""
        ds = self._load_stream()
        buffer: list[dict[str, torch.Tensor]] = []

        process_fn = (
            self._process_pretokenized
            if self.config.mode == "pretokenized"
            else self._process_retokenized
        )

        for example in ds:
            for item in process_fn(example):
                buffer.append(item)

                if len(buffer) >= self.config.buffer_size:
                    random.shuffle(buffer)
                    while len(buffer) > self.config.buffer_size // 2:
                        yield buffer.pop()

        # Flush remaining
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_climbmix_dataloader(
    config: ClimbMixConfig,
    tokenizer: Optional[Any] = None,
) -> DataLoader:
    """
    Create a DataLoader for ClimbMix distillation training.

    Usage (pretokenized, GPT-2 vocab):
        config = ClimbMixConfig(mode="pretokenized", max_seq_len=1024)
        loader = create_climbmix_dataloader(config)

    Usage (retokenized, Aya vocab):
        from distill.data import get_shared_tokenizer
        tokenizer = get_shared_tokenizer()
        config = ClimbMixConfig(
            mode="retokenize",
            target_tokenizer_name="CohereForAI/aya-expanse-8b",
        )
        loader = create_climbmix_dataloader(config, tokenizer=tokenizer)
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset = ClimbMixDataset(config=config, tokenizer=tokenizer)

    def collate_fn(
        batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
