"""
Tests for the distillation pipeline modules with low coverage.

Targets:
    - block_surgery.py: _get_teacher_layers, _get_attn, _get_mlp, helper logic
    - data.py: DistillDataConfig, LanguageBalancedSampler, TeacherLogitCache,
               MultilingualDistillDataset, create_distill_dataloader
    - sft.py: format_chat, find_assistant_spans, build_sft_labels,
              SFTTrainer.train_step, SFTTrainer.evaluate, SFTTrainer._setup_optimizer,
              SFTTrainer._save_checkpoint, SFTTrainer._log_language_summary
    - kl_distillation.py: KLDistillationTrainer._setup_optimizer,
                          KLDistillationTrainer.train_step,
                          KLDistillationTrainer._log_language_summary,
                          KLDistillationTrainer._save_checkpoint
    - alignment.py: DimensionProjector, alignment_loss,
                    LayerAlignmentTrainer.compute_step_loss,
                    LayerAlignmentTrainer.compute_cka_scores,
                    LayerAlignmentTrainer._setup_optimizer,
                    LayerAlignmentTrainer._save_checkpoint

All tests use synthetic tensors and mock models -- NO real model downloads.

Wayy Research, 2024-2026.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Path setup for local packages
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "aetheris"))

from aya_distill.distill.block_surgery import (
    _get_attn,
    _get_mlp,
    _get_teacher_layers,
)
from aya_distill.distill.data import (
    DistillDataConfig,
    LanguageBalancedSampler,
    MultilingualDistillDataset,
    TeacherLogitCache,
    create_distill_dataloader,
)
from aya_distill.distill.sft import (
    SFTConfig,
    SFTTrainer,
    build_sft_labels,
    find_assistant_spans,
    format_chat,
)
from aya_distill.distill.kl_distillation import (
    KLDistillationConfig,
    KLDistillationTrainer,
    VocabProjector,
    kl_divergence_loss,
)
from aya_distill.distill.alignment import (
    AlignmentConfig,
    DimensionProjector,
    LayerAlignmentTrainer,
    alignment_loss,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight mock models (same as test_training.py)
# ---------------------------------------------------------------------------

class TinyTeacher(nn.Module):
    """Minimal transformer-like teacher for testing. Has .model.layers."""

    def __init__(
        self, n_layers: int = 4, d_model: int = 32, vocab_size: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.model = nn.Module()
        layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.self_attn = nn.Linear(d_model, d_model)
            layer.mlp = nn.Linear(d_model, d_model)
            layers.append(layer)
        self.model.layers = layers
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer.self_attn(x) + x
            x = layer.mlp(x) + x
        return self.head(x)


class TinyStudent(nn.Module):
    """Minimal student model with .layers attribute (alternating types)."""

    def __init__(
        self, n_layers: int = 4, d_model: int = 16, vocab_size: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x) + x
        logits = self.head(x)
        return {
            "logits": logits,
            "loss": torch.tensor(0.0),
            "aux_loss": torch.tensor(0.0),
        }


# ===================================================================
# block_surgery.py tests
# ===================================================================

class TestGetTeacherLayers:
    """Tests for _get_teacher_layers extraction from different model shapes."""

    def test_model_model_layers_path(self):
        """Standard HuggingFace path: model.model.layers."""
        teacher = TinyTeacher(n_layers=4, d_model=32)
        layers = _get_teacher_layers(teacher)
        assert len(layers) == 4

    def test_direct_layers_path(self):
        """Fallback path: model.layers directly."""
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        layers = _get_teacher_layers(model)
        assert len(layers) == 3

    def test_no_layers_raises(self):
        """Model without recognizable layer paths raises ValueError."""
        model = nn.Module()
        model.some_other_thing = nn.Linear(8, 8)
        with pytest.raises(ValueError, match="Cannot find transformer layers"):
            _get_teacher_layers(model)

    def test_empty_layers(self):
        """Model with zero layers returns empty list."""
        teacher = TinyTeacher(n_layers=0, d_model=16)
        layers = _get_teacher_layers(teacher)
        assert layers == []


class TestGetAttn:
    """Tests for _get_attn extraction."""

    def test_self_attn_found(self):
        layer = nn.Module()
        layer.self_attn = nn.Linear(8, 8)
        result = _get_attn(layer)
        assert result is layer.self_attn

    def test_no_attn_raises(self):
        layer = nn.Module()
        layer.some_module = nn.Linear(8, 8)
        with pytest.raises(ValueError, match="Cannot find attention"):
            _get_attn(layer)


class TestGetMlp:
    """Tests for _get_mlp extraction."""

    def test_mlp_found(self):
        layer = nn.Module()
        layer.mlp = nn.Linear(8, 8)
        result = _get_mlp(layer)
        assert result is layer.mlp

    def test_no_mlp_raises(self):
        layer = nn.Module()
        layer.some_module = nn.Linear(8, 8)
        with pytest.raises(ValueError, match="Cannot find MLP"):
            _get_mlp(layer)


# ===================================================================
# data.py tests -- TeacherLogitCache
# ===================================================================

class TestTeacherLogitCache:
    """Tests for TeacherLogitCache save/load round-trip."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Cached logits should reconstruct to match originals at top-k positions."""
        cache = TeacherLogitCache(str(tmp_path), top_k=4)
        logits = torch.randn(2, 5, 20)

        cache.save(0, logits)
        assert cache.has_cache(0)
        assert not cache.has_cache(1)

        loaded = cache.load(0)
        assert loaded.shape == (2, 5, 20)

        # Top-k values should be approximately preserved (half precision rounding)
        top_vals, top_idx = torch.topk(logits, 4, dim=-1)
        loaded_top = loaded.gather(-1, top_idx)
        assert torch.allclose(loaded_top, top_vals, atol=0.01)

    def test_non_topk_positions_are_neg_inf(self, tmp_path):
        """Positions not in top-k should be -inf."""
        cache = TeacherLogitCache(str(tmp_path), top_k=2)
        logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        cache.save(0, logits)
        loaded = cache.load(0)

        # Bottom 3 values should be -inf
        assert (loaded[0, 0, :3] == float("-inf")).sum() == 3

    def test_cache_path_format(self, tmp_path):
        cache = TeacherLogitCache(str(tmp_path), top_k=8)
        path = cache._cache_path(42)
        assert path.name == "logits_00000042.pt"

    def test_multiple_batches(self, tmp_path):
        """Multiple batch caches coexist without collision."""
        cache = TeacherLogitCache(str(tmp_path), top_k=4)
        for i in range(3):
            cache.save(i, torch.randn(1, 4, 10))
        for i in range(3):
            assert cache.has_cache(i)
            loaded = cache.load(i)
            assert loaded.shape == (1, 4, 10)


# ===================================================================
# data.py tests -- MultilingualDistillDataset & DataLoader
# ===================================================================

class TestMultilingualDistillDataset:
    """Tests for the streaming multilingual dataset."""

    def _make_mock_tokenizer(self):
        """Create a minimal mock tokenizer."""
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.eos_token = "<|endoftext|>"

        def tokenize_fn(text, **kwargs):
            # Produce fake token ids based on text length
            length = min(len(text) // 2, kwargs.get("max_length", 512))
            length = max(length, 2)
            ids = torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0)
            return {"input_ids": ids}

        tok.side_effect = tokenize_fn
        tok.__call__ = tokenize_fn
        return tok

    def test_empty_datasets_yields_nothing(self):
        """No per-language datasets means no items."""
        cfg = DistillDataConfig(languages=["en", "es"])
        tok = self._make_mock_tokenizer()
        ds = MultilingualDistillDataset(cfg, tok, per_language_datasets={})
        items = list(ds)
        assert items == []

    def test_alignment_mode_yields_items(self):
        """Dataset with data should yield items in alignment mode."""
        cfg = DistillDataConfig(
            mode="alignment", languages=["en"], max_seq_len=16, buffer_size=4
        )
        tok = self._make_mock_tokenizer()
        fake_data = [{"text": f"This is sentence number {i} for testing."} for i in range(10)]
        ds = MultilingualDistillDataset(cfg, tok, per_language_datasets={"en": fake_data})
        items = list(ds)
        assert len(items) > 0
        for item in items:
            assert "input_ids" in item
            assert "labels" in item
            assert item["language"] == "en"

    def test_get_text_from_example_variants(self):
        """_get_text_from_example handles multiple formats."""
        cfg = DistillDataConfig(languages=["en"])
        tok = self._make_mock_tokenizer()
        ds = MultilingualDistillDataset(cfg, tok)

        assert ds._get_text_from_example({"text": "hello"}) == "hello"
        assert ds._get_text_from_example({"content": "world"}) == "world"
        assert ds._get_text_from_example({"sentence": "foo"}) == "foo"

        # messages format
        msgs_example = {"messages": [{"content": "a"}, {"content": "b"}]}
        assert "a" in ds._get_text_from_example(msgs_example)

        # Fallback: string values
        assert ds._get_text_from_example({"x": "val", "y": 123}) == "val"

    def test_tokenize_text_short_returns_none(self):
        """Text shorter than 10 chars returns None."""
        cfg = DistillDataConfig(languages=["en"], max_seq_len=32)
        tok = self._make_mock_tokenizer()
        ds = MultilingualDistillDataset(cfg, tok)
        assert ds._tokenize_text("short") is None

    def test_tokenize_text_valid(self):
        """Valid text tokenizes to (input_ids, labels) tuple."""
        cfg = DistillDataConfig(languages=["en"], max_seq_len=32)
        tok = self._make_mock_tokenizer()
        ds = MultilingualDistillDataset(cfg, tok)
        result = ds._tokenize_text("This is a sufficiently long text for testing.")
        assert result is not None
        input_ids, labels = result
        assert input_ids.shape[0] == 32  # padded to max_seq_len
        assert labels.shape[0] == 32


class TestCreateDistillDataloader:
    """Tests for create_distill_dataloader factory."""

    def test_creates_dataloader(self):
        cfg = DistillDataConfig(
            mode="alignment", languages=["en"], max_seq_len=16, batch_size=2
        )
        tok = MagicMock()
        tok.pad_token_id = 0

        def tokenize_fn(text, **kwargs):
            length = min(len(text) // 2, kwargs.get("max_length", 16))
            length = max(length, 2)
            ids = torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0)
            return {"input_ids": ids}

        tok.__call__ = tokenize_fn
        tok.eos_token = "<|endoftext|>"

        fake_data = [{"text": f"Test sentence number {i} is here."} for i in range(5)]
        dl = create_distill_dataloader(cfg, tok, {"en": fake_data})
        assert dl is not None
        assert dl.batch_size == 2


# ===================================================================
# sft.py tests -- build_sft_labels, SFTTrainer methods
# ===================================================================

class _MockEncoding:
    """Mock encoding that provides token_to_chars mapping (1 token per char)."""

    def __init__(self, ids: torch.Tensor, length: int):
        self._ids = ids
        self._length = length

    def __getitem__(self, key: str):
        if key == "input_ids":
            return self._ids
        raise KeyError(key)

    def token_to_chars(self, batch_idx: int, tok_idx: int):
        if tok_idx < self._length:
            return (tok_idx, tok_idx + 1)
        return None


class _FakeTokenizer:
    """Minimal callable tokenizer for build_sft_labels tests."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.eos_token = "<|endoftext|>"

    def __call__(self, text, **kwargs):
        max_len = kwargs.get("max_length", 512)
        length = min(len(text), max_len)
        ids = torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0)
        return _MockEncoding(ids, length)


class TestBuildSftLabels:
    """Tests for build_sft_labels with assistant-only masking."""

    def test_short_text_returns_none(self):
        tok = _FakeTokenizer()
        assert build_sft_labels("hi", tok, 32) is None

    def test_valid_text_returns_tensors(self):
        tok = _FakeTokenizer()
        text = "<|user|>Hello there<|assistant|>Hi back<|endoftext|>"
        result = build_sft_labels(text, tok, 128)
        assert result is not None
        input_ids, labels = result
        assert input_ids.shape[0] == 128
        assert labels.shape[0] == 128

    def test_non_assistant_tokens_masked(self):
        tok = _FakeTokenizer()
        text = "<|user|>Hello<|assistant|>Reply<|endoftext|>"
        result = build_sft_labels(text, tok, 128)
        assert result is not None
        _, labels = result
        # Many positions should be masked (-100)
        assert (labels == -100).sum().item() > 0


class TestSFTTrainerSetupOptimizer:
    """Tests for SFTTrainer._setup_optimizer."""

    def test_setup_optimizer_returns_optimizer_and_scheduler(self):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig(warmup_steps=10, total_steps=100)
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))
        optimizer, scheduler = trainer._setup_optimizer()
        assert optimizer is not None
        assert scheduler is not None
        assert len(optimizer.param_groups) > 0

    def test_optimizer_lr_matches_config(self):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig(lr=3e-4, warmup_steps=5, total_steps=50)
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))
        optimizer, _ = trainer._setup_optimizer()
        # Initial LR is scaled by start_factor (0.01)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 * 0.01, rel=0.1)


class TestSFTTrainerTrainStep:
    """Tests for SFTTrainer.train_step with synthetic data."""

    def test_train_step_dict_output(self):
        """Student returning dict with 'loss' key."""
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))

        input_ids = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))
        loss, metrics = trainer.train_step(input_ids, labels, language="en")

        assert "loss" in metrics
        assert "ce_loss" in metrics
        assert "aux_loss" in metrics
        assert isinstance(loss, torch.Tensor)

    def test_train_step_tracks_language(self):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))

        input_ids = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))
        trainer.train_step(input_ids, labels, language="hi")
        assert "hi" in trainer.per_language_loss
        assert len(trainer.per_language_loss["hi"]) == 1

    def test_train_step_no_language(self):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))

        input_ids = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))
        loss, metrics = trainer.train_step(input_ids, labels, language=None)
        assert len(trainer.per_language_loss) == 0

    def test_train_step_non_dict_output(self):
        """Student returning raw tensor triggers manual loss computation."""

        class RawOutputStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 16)
                self.head = nn.Linear(16, 32)

            def forward(self, input_ids, **kwargs):
                return self.head(self.embed(input_ids))

        model = RawOutputStudent()
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))

        input_ids = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))
        loss, metrics = trainer.train_step(input_ids, labels)
        assert loss.item() > 0
        assert metrics["ce_loss"] == metrics["loss"]


class TestSFTTrainerEvaluate:
    """Tests for SFTTrainer.evaluate."""

    def test_evaluate_dict_batch(self):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))

        eval_data = [
            {
                "input_ids": torch.randint(0, 32, (2, 8)),
                "labels": torch.randint(0, 32, (2, 8)),
                "language": ["en", "es"],
            }
            for _ in range(3)
        ]
        metrics = trainer.evaluate(eval_data, max_batches=2)
        assert "eval_loss" in metrics

    def test_evaluate_tuple_batch(self):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))

        eval_data = [
            (torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8)))
            for _ in range(3)
        ]
        metrics = trainer.evaluate(eval_data, max_batches=2)
        assert "eval_loss" in metrics

    def test_evaluate_max_batches_respected(self):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))

        eval_data = [
            {
                "input_ids": torch.randint(0, 32, (1, 8)),
                "labels": torch.randint(0, 32, (1, 8)),
            }
            for _ in range(100)
        ]
        metrics = trainer.evaluate(eval_data, max_batches=5)
        assert "eval_loss" in metrics

    def test_evaluate_raw_output_model(self):
        """Evaluate with model that returns raw tensors."""

        class RawModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 16)
                self.head = nn.Linear(16, 32)

            def forward(self, input_ids, **kwargs):
                return self.head(self.embed(input_ids))

        model = RawModel()
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))

        eval_data = [
            (torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8)))
        ]
        metrics = trainer.evaluate(eval_data, max_batches=1)
        assert metrics["eval_loss"] > 0


class TestSFTTrainerSaveCheckpoint:
    """Tests for SFTTrainer._save_checkpoint."""

    def test_save_checkpoint(self, tmp_path):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))
        trainer.step = 10
        trainer.loss_history = [1.0, 0.9, 0.8]

        path = tmp_path / "ckpt.pt"
        trainer._save_checkpoint(path)
        assert path.exists()

        loaded = torch.load(path, map_location="cpu")
        assert loaded["step"] == 10
        assert loaded["loss_history"] == [1.0, 0.9, 0.8]
        assert "student_state_dict" in loaded


class TestSFTTrainerLogLanguageSummary:
    """Tests for SFTTrainer._log_language_summary."""

    def test_empty_language_loss_no_crash(self):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))
        trainer._log_language_summary()  # Should not raise

    def test_with_language_data(self):
        model = TinyStudent(n_layers=2, d_model=16, vocab_size=32)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))
        trainer.per_language_loss["en"] = [1.0, 0.9, 0.8]
        trainer.per_language_loss["es"] = [1.1, 1.0]
        trainer._log_language_summary()  # Should not raise


# ===================================================================
# kl_distillation.py tests -- deeper coverage
# ===================================================================

class TestKLDistillationTrainerSetupOptimizer:
    """Tests for KLDistillationTrainer._setup_optimizer."""

    def test_setup_without_projection(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(
            teacher_vocab_size=64, student_vocab_size=64,
            warmup_steps=5, total_steps=50,
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )
        optimizer, scheduler = trainer._setup_optimizer()
        assert optimizer is not None
        assert scheduler is not None

    def test_setup_with_projection_includes_projector_params(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=100)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=50)
        cfg = KLDistillationConfig(
            teacher_vocab_size=100, student_vocab_size=50,
            warmup_steps=5, total_steps=50,
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )
        optimizer, scheduler = trainer._setup_optimizer()
        # Should include projector params
        total_params = sum(
            len(pg["params"]) for pg in optimizer.param_groups
        )
        student_params = len(list(student.parameters()))
        projector_params = len(list(trainer.vocab_projector.parameters()))
        assert total_params == student_params + projector_params


class TestKLDistillationTrainerTrainStep:
    """Tests for KLDistillationTrainer.train_step."""

    def test_train_step_basic(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(
            teacher_vocab_size=64, student_vocab_size=64
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )

        input_ids = torch.randint(0, 64, (2, 8))
        labels = torch.randint(0, 64, (2, 8))
        loss, metrics = trainer.train_step(input_ids, labels)

        assert "loss" in metrics
        assert "kl_loss" in metrics
        assert "ce_loss" in metrics
        assert "aux_loss" in metrics
        assert isinstance(loss, torch.Tensor)

    def test_train_step_with_language_tracking(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(
            teacher_vocab_size=64, student_vocab_size=64
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )

        input_ids = torch.randint(0, 64, (2, 8))
        labels = torch.randint(0, 64, (2, 8))
        trainer.train_step(input_ids, labels, language="sw")
        assert "sw" in trainer.per_language_kl
        assert "kl/sw" in trainer.train_step(
            input_ids, labels, language="sw"
        )[1]

    def test_train_step_all_masked_labels(self):
        """When all labels are -100, loss should be zero."""
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(
            teacher_vocab_size=64, student_vocab_size=64
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )

        input_ids = torch.randint(0, 64, (2, 8))
        labels = torch.full((2, 8), -100, dtype=torch.long)
        loss, metrics = trainer.train_step(input_ids, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_train_step_with_vocab_projection(self):
        """Train step with mismatched vocab sizes."""
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=100)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=50)
        cfg = KLDistillationConfig(
            teacher_vocab_size=100, student_vocab_size=50
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )

        input_ids = torch.randint(0, 50, (2, 8))
        labels = torch.randint(0, 50, (2, 8))
        loss, metrics = trainer.train_step(input_ids, labels)
        assert isinstance(loss, torch.Tensor)

    def test_train_step_gradient_flows(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(
            teacher_vocab_size=64, student_vocab_size=64
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )

        input_ids = torch.randint(0, 64, (2, 8))
        labels = torch.randint(0, 64, (2, 8))
        loss, _ = trainer.train_step(input_ids, labels)
        loss.backward()

        # Student gradients should exist
        has_grad = any(
            p.grad is not None for p in student.parameters()
        )
        assert has_grad


class TestKLDistillationTrainerSaveCheckpoint:
    """Tests for KLDistillationTrainer._save_checkpoint."""

    def test_save_without_projection(self, tmp_path):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(
            teacher_vocab_size=64, student_vocab_size=64
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )
        trainer.step = 5
        path = tmp_path / "kl_ckpt.pt"
        trainer._save_checkpoint(path)
        loaded = torch.load(path, map_location="cpu")
        assert loaded["step"] == 5
        assert "vocab_projector_state_dict" not in loaded

    def test_save_with_projection(self, tmp_path):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=100)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=50)
        cfg = KLDistillationConfig(
            teacher_vocab_size=100, student_vocab_size=50
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )
        path = tmp_path / "kl_ckpt_proj.pt"
        trainer._save_checkpoint(path)
        loaded = torch.load(path, map_location="cpu")
        assert "vocab_projector_state_dict" in loaded


class TestKLDistillationTrainerLogLanguageSummary:
    """Tests for KLDistillationTrainer._log_language_summary."""

    def test_empty(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(
            teacher_vocab_size=64, student_vocab_size=64
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )
        trainer._log_language_summary()  # Should not raise

    def test_with_data(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(
            teacher_vocab_size=64, student_vocab_size=64
        )
        trainer = KLDistillationTrainer(
            teacher, student, cfg, device=torch.device("cpu")
        )
        trainer.per_language_kl["en"] = [0.5, 0.4, 0.3]
        trainer.per_language_kl["hi"] = [0.6]
        trainer._log_language_summary()


# ===================================================================
# alignment.py tests -- deeper coverage
# ===================================================================

class TestAlignmentTrainerSetupOptimizer:
    """Tests for LayerAlignmentTrainer._setup_optimizer."""

    def test_setup_includes_student_and_projector_params(self):
        teacher = TinyTeacher(n_layers=4, d_model=32)
        student = TinyStudent(n_layers=4, d_model=16)
        cfg = AlignmentConfig(warmup_steps=5, total_steps=50)
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=4, n_student_layers=4,
            d_teacher=32, d_student=16,
            device=torch.device("cpu"),
        )
        optimizer, scheduler = trainer._setup_optimizer()
        assert optimizer is not None
        assert scheduler is not None


class TestAlignmentTrainerComputeStepLoss:
    """Tests for LayerAlignmentTrainer.compute_step_loss."""

    def test_loss_with_matching_activations(self):
        teacher = TinyTeacher(n_layers=4, d_model=32)
        student = TinyStudent(n_layers=4, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=4, n_student_layers=4,
            d_teacher=32, d_student=16,
            device=torch.device("cpu"),
        )

        # Build synthetic activation dicts matching expected keys
        teacher_acts = {}
        student_acts = {}
        for t_idx, s_idx, component in trainer.layer_mapping:
            t_key = trainer._get_teacher_key(t_idx, component)
            s_key = trainer._get_student_key(s_idx)
            teacher_acts[t_key] = torch.randn(20, 32)
            student_acts[s_key] = torch.randn(20, 16)

        loss = trainer.compute_step_loss(teacher_acts, student_acts)
        assert loss.item() > 0
        assert loss.dim() == 0

    def test_loss_with_empty_activations(self):
        """No matching activations means zero loss."""
        teacher = TinyTeacher(n_layers=2, d_model=16)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=16, d_student=16,
            device=torch.device("cpu"),
        )
        loss = trainer.compute_step_loss({}, {})
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_with_different_token_counts(self):
        """Activations with different token counts should be truncated."""
        teacher = TinyTeacher(n_layers=2, d_model=32)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=32, d_student=16,
            device=torch.device("cpu"),
        )

        teacher_acts = {}
        student_acts = {}
        for t_idx, s_idx, component in trainer.layer_mapping:
            t_key = trainer._get_teacher_key(t_idx, component)
            s_key = trainer._get_student_key(s_idx)
            teacher_acts[t_key] = torch.randn(30, 32)
            student_acts[s_key] = torch.randn(20, 16)  # Different count

        loss = trainer.compute_step_loss(teacher_acts, student_acts)
        assert loss.item() > 0


class TestAlignmentTrainerComputeCKAScores:
    """Tests for LayerAlignmentTrainer.compute_cka_scores."""

    def test_cka_scores_returned(self):
        teacher = TinyTeacher(n_layers=2, d_model=32)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig(cka_threshold=0.99)  # High threshold to trigger warning
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=32, d_student=16,
            device=torch.device("cpu"),
        )

        teacher_acts = {}
        student_acts = {}
        for t_idx, s_idx, component in trainer.layer_mapping:
            t_key = trainer._get_teacher_key(t_idx, component)
            s_key = trainer._get_student_key(s_idx)
            teacher_acts[t_key] = torch.randn(20, 32)
            student_acts[s_key] = torch.randn(20, 16)

        scores = trainer.compute_cka_scores(teacher_acts, student_acts)
        assert len(scores) > 0
        for key, score in scores.items():
            assert 0.0 <= score <= 1.0

    def test_cka_threshold_warning(self):
        """CKA below threshold should emit warning."""
        teacher = TinyTeacher(n_layers=2, d_model=32)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig(cka_threshold=0.99)
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=32, d_student=16,
            device=torch.device("cpu"),
        )

        teacher_acts = {}
        student_acts = {}
        for t_idx, s_idx, component in trainer.layer_mapping:
            t_key = trainer._get_teacher_key(t_idx, component)
            s_key = trainer._get_student_key(s_idx)
            teacher_acts[t_key] = torch.randn(20, 32)
            student_acts[s_key] = torch.randn(20, 16)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trainer.compute_cka_scores(teacher_acts, student_acts)
            cka_warnings = [x for x in w if "CKA below threshold" in str(x.message)]
            assert len(cka_warnings) > 0

    def test_cka_with_missing_keys(self):
        """Missing keys should be skipped, not crash."""
        teacher = TinyTeacher(n_layers=2, d_model=16)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=16, d_student=16,
            device=torch.device("cpu"),
        )
        scores = trainer.compute_cka_scores({}, {})
        assert scores == {}


class TestAlignmentTrainerSaveCheckpoint:
    """Tests for LayerAlignmentTrainer._save_checkpoint."""

    def test_save_checkpoint(self, tmp_path):
        teacher = TinyTeacher(n_layers=2, d_model=16)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=16, d_student=16,
            device=torch.device("cpu"),
        )
        trainer.step = 42
        trainer.loss_history = [2.0, 1.5, 1.0]

        path = tmp_path / "align_ckpt.pt"
        trainer._save_checkpoint(path)
        assert path.exists()

        loaded = torch.load(path, map_location="cpu")
        assert loaded["step"] == 42
        assert loaded["loss_history"] == [2.0, 1.5, 1.0]
        assert "student_state_dict" in loaded
        assert "projectors_state_dict" in loaded
        assert loaded["config"]["loss_type"] == "mse+cosine"


class TestAlignmentTrainerHelperKeys:
    """Tests for _get_teacher_key and _get_student_key."""

    def test_teacher_key_attn(self):
        teacher = TinyTeacher(n_layers=2, d_model=16)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=16, d_student=16,
            device=torch.device("cpu"),
        )
        assert trainer._get_teacher_key(0, "attn->ssm") == "teacher.attn.0"
        assert trainer._get_teacher_key(3, "ffn->moe") == "teacher.ffn.3"

    def test_teacher_key_unknown_raises(self):
        teacher = TinyTeacher(n_layers=2, d_model=16)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=16, d_student=16,
            device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="Unknown component"):
            trainer._get_teacher_key(0, "unknown")

    def test_student_key(self):
        teacher = TinyTeacher(n_layers=2, d_model=16)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher, student, cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=16, d_student=16,
            device=torch.device("cpu"),
        )
        assert trainer._get_student_key(0) == "student.ssm.0"
        assert trainer._get_student_key(1) == "student.moe.1"
        assert trainer._get_student_key(2) == "student.ssm.2"


# ===================================================================
# kl_distillation.py edge cases
# ===================================================================

class TestKLDivergenceLossEdgeCases:
    """Additional edge cases for kl_divergence_loss."""

    def test_single_element_vocab(self):
        """Single-element vocab should give zero KL."""
        t = torch.tensor([[[1.0]]])
        s = torch.tensor([[[1.0]]])
        loss = kl_divergence_loss(t, s, temperature=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_batch_dim_1(self):
        """Works with batch size 1."""
        t = torch.randn(1, 5, 20)
        s = torch.randn(1, 5, 20)
        loss = kl_divergence_loss(t, s, temperature=2.0)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_large_temperature(self):
        """Very high temperature produces valid loss."""
        t = torch.randn(2, 5, 20)
        s = torch.randn(2, 5, 20)
        loss = kl_divergence_loss(t, s, temperature=100.0)
        assert torch.isfinite(loss)

    def test_small_temperature(self):
        """Very small temperature produces valid loss."""
        t = torch.randn(2, 5, 20)
        s = torch.randn(2, 5, 20)
        loss = kl_divergence_loss(t, s, temperature=0.1)
        assert torch.isfinite(loss)


# ===================================================================
# data.py -- DistillDataConfig additional validation
# ===================================================================

class TestDistillDataConfigExtended:
    """Extended tests for DistillDataConfig."""

    def test_logit_cache_settings(self):
        cfg = DistillDataConfig(
            cache_teacher_logits=True,
            logit_cache_dir="/tmp/cache",
            logit_top_k=32,
        )
        assert cfg.cache_teacher_logits is True
        assert cfg.logit_cache_dir == "/tmp/cache"
        assert cfg.logit_top_k == 32

    def test_local_data_dir_setting(self):
        cfg = DistillDataConfig(local_data_dir="/some/path")
        assert cfg.local_data_dir == "/some/path"

    def test_streaming_default_true(self):
        cfg = DistillDataConfig()
        assert cfg.streaming is True

    def test_hf_token(self):
        cfg = DistillDataConfig(hf_token="fake-token")
        assert cfg.hf_token == "fake-token"

    def test_custom_languages(self):
        cfg = DistillDataConfig(languages=["en", "es"])
        assert cfg.languages == ["en", "es"]


# ===================================================================
# VocabProjector edge cases
# ===================================================================

class TestVocabProjectorEdgeCases:
    """Additional edge cases for VocabProjector."""

    def test_identity_init_small_to_large(self):
        """When student vocab > teacher vocab, identity portion should be correct."""
        proj = VocabProjector(50, 80)
        assert proj.needs_projection
        with torch.no_grad():
            diag = torch.diag(proj.proj.weight[:50, :50])
        assert torch.allclose(diag, torch.ones(50))

    def test_forward_preserves_batch_dims(self):
        proj = VocabProjector(30, 50)
        x = torch.randn(4, 12, 30)
        out = proj(x)
        assert out.shape == (4, 12, 50)
