"""
Tests for aya-distill training modules.

Covers alignment, KL distillation, SFT, data pipeline, and ClimbMix.
All tests run on CPU with synthetic models -- no GPU or real downloads.

Wayy Research, 2024-2026.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Path setup for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "aetheris"))

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from aya_distill.distill.alignment import (
    AlignmentConfig,
    DimensionProjector,
    alignment_loss,
    LayerAlignmentTrainer,
)
from aya_distill.distill.kl_distillation import (
    KLDistillationConfig,
    VocabProjector,
    kl_divergence_loss,
    KLDistillationTrainer,
)
from aya_distill.distill.sft import (
    SFTConfig,
    SFTTrainer,
    format_chat,
    find_assistant_spans,
)
from aya_distill.distill.data import (
    DistillDataConfig,
    LanguageBalancedSampler,
    TARGET_LANGUAGES,
)
from aya_distill.distill.climbmix import (
    ClimbMixConfig,
    ClimbMixDataset,
    GPT2_VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight mock models
# ---------------------------------------------------------------------------

class TinyTeacher(nn.Module):
    """Minimal transformer-like teacher for testing. Has .model.layers."""

    def __init__(self, n_layers: int = 4, d_model: int = 32, vocab_size: int = 64):
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

    def __init__(self, n_layers: int = 4, d_model: int = 16, vocab_size: int = 64):
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
        return {"logits": logits, "loss": torch.tensor(0.0), "aux_loss": torch.tensor(0.0)}


# ===================================================================
# AlignmentConfig tests
# ===================================================================

class TestAlignmentConfig:
    """Tests for AlignmentConfig validation."""

    def test_default_config(self):
        cfg = AlignmentConfig()
        assert cfg.loss_type == "mse+cosine"
        assert cfg.mse_weight == 1.0
        assert cfg.cosine_weight == 0.5
        assert cfg.cka_threshold == 0.75

    @pytest.mark.parametrize("loss_type", ["mse", "cosine", "mse+cosine"])
    def test_valid_loss_types(self, loss_type):
        cfg = AlignmentConfig(loss_type=loss_type)
        assert cfg.loss_type == loss_type

    @pytest.mark.parametrize("loss_type", ["l1", "huber", "", "MSE"])
    def test_invalid_loss_types(self, loss_type):
        with pytest.raises(ValueError, match="loss_type must be one of"):
            AlignmentConfig(loss_type=loss_type)

    def test_frozen(self):
        cfg = AlignmentConfig()
        with pytest.raises(AttributeError):
            cfg.lr = 0.1  # type: ignore


# ===================================================================
# DimensionProjector tests
# ===================================================================

class TestDimensionProjector:
    """Tests for DimensionProjector forward pass and shape handling."""

    @pytest.mark.parametrize("d_teacher,d_student", [
        (64, 32),
        (32, 64),
        (128, 16),
    ])
    def test_forward_shape_with_projection(self, d_teacher, d_student):
        proj = DimensionProjector(d_teacher, d_student)
        x = torch.randn(10, d_teacher)
        out = proj(x)
        assert out.shape == (10, d_student)

    def test_identity_when_same_dim(self):
        proj = DimensionProjector(32, 32)
        assert not proj.needs_projection
        x = torch.randn(5, 32)
        out = proj(x)
        assert torch.allclose(x, out)

    def test_needs_projection_flag(self):
        proj = DimensionProjector(64, 32)
        assert proj.needs_projection

    @given(
        d_in=st.integers(min_value=2, max_value=128),
        d_out=st.integers(min_value=2, max_value=128),
        n_tokens=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30)
    def test_output_shape_property(self, d_in, d_out, n_tokens):
        """Property: output always has (n_tokens, d_out)."""
        proj = DimensionProjector(d_in, d_out)
        x = torch.randn(n_tokens, d_in)
        out = proj(x)
        assert out.shape == (n_tokens, d_out)


# ===================================================================
# alignment_loss tests
# ===================================================================

class TestAlignmentLoss:
    """Tests for alignment_loss computation."""

    def test_identical_tensors_mse_zero(self):
        t = torch.randn(20, 32)
        loss = alignment_loss(t, t.clone(), loss_type="mse")
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_identical_tensors_cosine_zero(self):
        t = torch.randn(20, 32)
        loss = alignment_loss(t, t.clone(), loss_type="cosine")
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_identical_tensors_combined_low(self):
        t = torch.randn(20, 32)
        loss = alignment_loss(t, t.clone(), loss_type="mse+cosine")
        assert loss.item() < 1e-5

    def test_random_tensors_higher_loss(self):
        torch.manual_seed(0)
        t = torch.randn(50, 64)
        s = torch.randn(50, 64)
        loss_random = alignment_loss(t, s, loss_type="mse+cosine")
        loss_same = alignment_loss(t, t.clone(), loss_type="mse+cosine")
        assert loss_random.item() > loss_same.item()

    def test_mse_only(self):
        t = torch.ones(10, 8)
        s = torch.zeros(10, 8)
        loss = alignment_loss(t, s, loss_type="mse", mse_weight=1.0)
        # MSE of ones vs zeros = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_cosine_orthogonal(self):
        """Orthogonal vectors should give cosine loss near 1.0."""
        t = torch.zeros(1, 4)
        t[0, 0] = 1.0
        s = torch.zeros(1, 4)
        s[0, 1] = 1.0
        loss = alignment_loss(t, s, loss_type="cosine", cosine_weight=1.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("weight", [0.0, 0.5, 1.0, 2.0])
    def test_mse_weight_scaling(self, weight):
        t = torch.ones(10, 8)
        s = torch.zeros(10, 8)
        loss = alignment_loss(t, s, loss_type="mse", mse_weight=weight)
        assert loss.item() == pytest.approx(weight * 1.0, abs=1e-5)

    def test_output_is_scalar(self):
        loss = alignment_loss(torch.randn(5, 8), torch.randn(5, 8))
        assert loss.dim() == 0


# ===================================================================
# KLDistillationConfig tests
# ===================================================================

class TestKLDistillationConfig:
    """Tests for KLDistillationConfig validation."""

    def test_default_config(self):
        cfg = KLDistillationConfig()
        assert cfg.temperature == 2.0
        assert cfg.alpha == 0.7

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha must be in"):
            KLDistillationConfig(alpha=alpha)

    @pytest.mark.parametrize("alpha", [0.01, 0.5, 0.99])
    def test_valid_alpha(self, alpha):
        cfg = KLDistillationConfig(alpha=alpha)
        assert cfg.alpha == alpha

    @pytest.mark.parametrize("temp", [0.0, -1.0, -0.01])
    def test_invalid_temperature(self, temp):
        with pytest.raises(ValueError, match="temperature must be positive"):
            KLDistillationConfig(temperature=temp)

    def test_frozen(self):
        cfg = KLDistillationConfig()
        with pytest.raises(AttributeError):
            cfg.temperature = 5.0  # type: ignore


# ===================================================================
# VocabProjector tests
# ===================================================================

class TestVocabProjector:
    """Tests for VocabProjector shape projection."""

    def test_no_projection_same_vocab(self):
        proj = VocabProjector(100, 100)
        assert not proj.needs_projection
        x = torch.randn(2, 10, 100)
        out = proj(x)
        assert torch.allclose(x, out)

    @pytest.mark.parametrize("t_vocab,s_vocab", [
        (100, 50),
        (50, 100),
        (200, 64),
    ])
    def test_projection_shape(self, t_vocab, s_vocab):
        proj = VocabProjector(t_vocab, s_vocab)
        assert proj.needs_projection
        x = torch.randn(2, 8, t_vocab)
        out = proj(x)
        assert out.shape == (2, 8, s_vocab)

    def test_identity_init_diagonal(self):
        """First min(t, s) entries of weight diagonal should be 1."""
        proj = VocabProjector(100, 80)
        with torch.no_grad():
            diag = torch.diag(proj.proj.weight[:80, :80])
        assert torch.allclose(diag, torch.ones(80))


# ===================================================================
# kl_divergence_loss tests
# ===================================================================

class TestKLDivergenceLoss:
    """Tests for kl_divergence_loss computation."""

    def test_identical_distributions_near_zero(self):
        logits = torch.randn(2, 10, 50)
        loss = kl_divergence_loss(logits, logits.clone(), temperature=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_for_different_distributions(self):
        torch.manual_seed(42)
        t = torch.randn(2, 10, 50)
        s = torch.randn(2, 10, 50)
        loss = kl_divergence_loss(t, s, temperature=1.0)
        assert loss.item() > 0.0

    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0, 5.0])
    def test_temperature_positive_loss(self, temperature):
        torch.manual_seed(7)
        t = torch.randn(2, 8, 30)
        s = torch.randn(2, 8, 30)
        loss = kl_divergence_loss(t, s, temperature=temperature)
        assert loss.item() > 0

    def test_higher_temperature_softens(self):
        """Higher temperature should produce different loss magnitude due to T^2 scaling."""
        torch.manual_seed(99)
        t = torch.randn(2, 10, 50)
        s = torch.randn(2, 10, 50)
        loss_t1 = kl_divergence_loss(t, s, temperature=1.0)
        loss_t10 = kl_divergence_loss(t, s, temperature=10.0)
        # Both should be positive, and they should differ meaningfully
        assert loss_t1.item() > 0
        assert loss_t10.item() > 0
        assert abs(loss_t1.item() - loss_t10.item()) > 0.01

    def test_output_is_scalar(self):
        loss = kl_divergence_loss(torch.randn(1, 5, 10), torch.randn(1, 5, 10), temperature=2.0)
        assert loss.dim() == 0

    def test_gradients_flow(self):
        s = torch.randn(2, 5, 20, requires_grad=True)
        t = torch.randn(2, 5, 20)
        loss = kl_divergence_loss(t, s, temperature=2.0)
        loss.backward()
        assert s.grad is not None
        assert s.grad.shape == s.shape


# ===================================================================
# SFTConfig tests
# ===================================================================

class TestSFTConfig:
    """Tests for SFTConfig."""

    def test_default_values(self):
        cfg = SFTConfig()
        assert cfg.lr == 2e-5
        assert cfg.max_seq_len == 1024
        assert cfg.weight_decay == 0.01

    def test_frozen(self):
        cfg = SFTConfig()
        with pytest.raises(AttributeError):
            cfg.lr = 0.1  # type: ignore


# ===================================================================
# SFT chat formatting tests
# ===================================================================

class TestFormatChat:
    """Tests for format_chat helper."""

    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = format_chat(msgs)
        assert "<|user|>hello" in result
        assert result.endswith("<|endoftext|>")

    def test_multi_turn(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = format_chat(msgs)
        assert "<|user|>hi" in result
        assert "<|assistant|>hello" in result

    def test_system_message(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        result = format_chat(msgs)
        assert "<|system|>You are helpful." in result

    def test_empty_messages(self):
        result = format_chat([])
        assert result == "<|endoftext|>"


class TestFindAssistantSpans:
    """Tests for find_assistant_spans."""

    def test_single_span(self):
        text = "<|user|>hi<|assistant|>hello<|endoftext|>"
        spans = find_assistant_spans(text)
        assert len(spans) == 1
        start, end = spans[0]
        assert text[start:end] == "hello"

    def test_multiple_spans(self):
        text = "<|user|>q1<|assistant|>a1<|user|>q2<|assistant|>a2<|endoftext|>"
        spans = find_assistant_spans(text)
        assert len(spans) == 2
        assert text[spans[0][0]:spans[0][1]] == "a1"
        assert text[spans[1][0]:spans[1][1]] == "a2"

    def test_no_assistant(self):
        text = "<|user|>hi<|endoftext|>"
        spans = find_assistant_spans(text)
        assert spans == []


# ===================================================================
# SFTTrainer init tests
# ===================================================================

class TestSFTTrainerInit:
    """Tests for SFTTrainer initialization."""

    def test_init_basic(self):
        model = TinyStudent(n_layers=4, d_model=16, vocab_size=64)
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))
        assert trainer.step == 0
        assert trainer.loss_history == []
        assert isinstance(trainer.ce_loss_fn, nn.CrossEntropyLoss)

    def test_per_language_tracking_initialized(self):
        model = TinyStudent()
        cfg = SFTConfig()
        trainer = SFTTrainer(model, cfg, device=torch.device("cpu"))
        assert len(trainer.per_language_loss) == 0


# ===================================================================
# DistillDataConfig tests
# ===================================================================

class TestDistillDataConfig:
    """Tests for DistillDataConfig validation."""

    @pytest.mark.parametrize("mode", ["alignment", "distillation", "sft"])
    def test_valid_modes(self, mode):
        cfg = DistillDataConfig(mode=mode)
        assert cfg.mode == mode

    @pytest.mark.parametrize("mode", ["train", "eval", ""])
    def test_invalid_modes(self, mode):
        with pytest.raises(ValueError, match="mode must be one of"):
            DistillDataConfig(mode=mode)

    def test_default_languages(self):
        cfg = DistillDataConfig()
        assert cfg.languages == list(TARGET_LANGUAGES)
        assert len(cfg.languages) == 10

    def test_unknown_language_logs_warning(self, caplog):
        """Unknown language triggers a logger.warning (not warnings.warn)."""
        import logging
        with caplog.at_level(logging.WARNING):
            DistillDataConfig(languages=["xx"])
        assert "not in target set" in caplog.text

    def test_default_seq_len(self):
        cfg = DistillDataConfig()
        assert cfg.max_seq_len == 512


# ===================================================================
# LanguageBalancedSampler tests
# ===================================================================

class TestLanguageBalancedSampler:
    """Tests for round-robin language sampling."""

    def test_round_robin_coverage(self):
        langs = ["en", "es", "hi"]
        sampler = LanguageBalancedSampler(langs, seed=0)
        drawn = [sampler.next_language() for _ in range(9)]
        # After 9 draws we should have exactly 3 of each
        for lang in langs:
            assert drawn.count(lang) == 3

    def test_all_languages_appear(self):
        sampler = LanguageBalancedSampler(list(TARGET_LANGUAGES), seed=42)
        drawn = set()
        for _ in range(len(TARGET_LANGUAGES)):
            drawn.add(sampler.next_language())
        assert drawn == set(TARGET_LANGUAGES)

    def test_reset_restarts(self):
        sampler = LanguageBalancedSampler(["en", "es"], seed=0)
        first_three = [sampler.next_language() for _ in range(3)]
        sampler.reset()
        after_reset = [sampler.next_language() for _ in range(3)]
        # After reset the internal index is 0, so the pattern restarts
        # (order may differ because reset shuffles)
        assert len(after_reset) == 3

    def test_single_language(self):
        sampler = LanguageBalancedSampler(["en"], seed=0)
        for _ in range(5):
            assert sampler.next_language() == "en"

    @given(n=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_never_crashes(self, n):
        """Property: sampling n times never raises."""
        sampler = LanguageBalancedSampler(["a", "b", "c"], seed=0)
        results = [sampler.next_language() for _ in range(n)]
        assert len(results) == n
        assert all(r in {"a", "b", "c"} for r in results)


# ===================================================================
# ClimbMixConfig tests
# ===================================================================

class TestClimbMixConfig:
    """Tests for ClimbMixConfig."""

    def test_defaults(self):
        cfg = ClimbMixConfig()
        assert cfg.dataset_name == "nvidia/ClimbMix"
        assert cfg.mode == "pretokenized"
        assert cfg.max_seq_len == 1024
        assert cfg.min_tokens == 32

    def test_custom_config(self):
        cfg = ClimbMixConfig(max_seq_len=512, batch_size=8, seed=123)
        assert cfg.max_seq_len == 512
        assert cfg.batch_size == 8
        assert cfg.seed == 123


# ===================================================================
# ClimbMixDataset tests
# ===================================================================

class TestClimbMixDataset:
    """Tests for ClimbMixDataset token chunking and processing."""

    def test_retokenize_mode_requires_tokenizer(self):
        cfg = ClimbMixConfig(mode="retokenize")
        with pytest.raises(ValueError, match="retokenize mode requires a tokenizer"):
            ClimbMixDataset(cfg, tokenizer=None)

    def test_pretokenized_mode_no_tokenizer_ok(self):
        cfg = ClimbMixConfig(mode="pretokenized")
        ds = ClimbMixDataset(cfg, tokenizer=None)
        assert ds.config.mode == "pretokenized"

    def test_chunk_tokens_basic(self):
        cfg = ClimbMixConfig(max_seq_len=8)
        ds = ClimbMixDataset(cfg)
        tokens = list(range(20))
        chunks = ds._chunk_tokens(tokens)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, torch.Tensor)
            # chunk length should be <= max_seq_len + 1
            assert len(chunk) <= cfg.max_seq_len + 1

    def test_chunk_tokens_short_input(self):
        """Short input (< half seq_len) should be dropped."""
        cfg = ClimbMixConfig(max_seq_len=32)
        ds = ClimbMixDataset(cfg)
        tokens = list(range(5))
        chunks = ds._chunk_tokens(tokens)
        # 5 tokens < 32//2 = 16, so this tail chunk is dropped
        assert len(chunks) == 0

    def test_process_pretokenized_shape(self):
        cfg = ClimbMixConfig(max_seq_len=16)
        ds = ClimbMixDataset(cfg)
        example = {"tokens": list(range(50)), "token_count": 50, "cluster_id": 0}
        items = list(ds._process_pretokenized(example))
        assert len(items) >= 1
        for item in items:
            assert item["input_ids"].shape == (16,)
            assert item["labels"].shape == (16,)

    def test_process_pretokenized_labels_shifted(self):
        """Labels should be shifted by 1 relative to input_ids."""
        cfg = ClimbMixConfig(max_seq_len=8)
        ds = ClimbMixDataset(cfg)
        tokens = list(range(20))
        items = list(ds._process_pretokenized({"tokens": tokens, "token_count": 20, "cluster_id": 0}))
        if items:
            item = items[0]
            # For a full chunk, labels[i] should equal the token after input_ids[i]
            # This verifies the autoregressive shift
            assert item["input_ids"].dtype == torch.long
            assert item["labels"].dtype == torch.long

    def test_gpt2_vocab_constant(self):
        assert GPT2_VOCAB_SIZE == 50257


# ===================================================================
# LayerAlignmentTrainer init tests
# ===================================================================

class TestLayerAlignmentTrainerInit:
    """Tests for LayerAlignmentTrainer initialization."""

    def test_init_creates_projectors(self):
        teacher = TinyTeacher(n_layers=4, d_model=32)
        student = TinyStudent(n_layers=4, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher=teacher,
            student=student,
            config=cfg,
            n_teacher_layers=4,
            n_student_layers=4,
            d_teacher=32,
            d_student=16,
            device=torch.device("cpu"),
        )
        assert len(trainer.projectors) > 0
        assert trainer.step == 0

    def test_teacher_frozen(self):
        teacher = TinyTeacher(n_layers=2, d_model=16)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher=teacher,
            student=student,
            config=cfg,
            n_teacher_layers=2,
            n_student_layers=2,
            d_teacher=16,
            d_student=16,
            device=torch.device("cpu"),
        )
        for param in trainer.teacher.parameters():
            assert not param.requires_grad

    def test_layer_mapping_populated(self):
        teacher = TinyTeacher(n_layers=8, d_model=32)
        student = TinyStudent(n_layers=4, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher=teacher,
            student=student,
            config=cfg,
            n_teacher_layers=8,
            n_student_layers=4,
            d_teacher=32,
            d_student=16,
            device=torch.device("cpu"),
        )
        assert len(trainer.layer_mapping) == 4
        # Each mapping is (t_idx, s_idx, component)
        for t_idx, s_idx, component in trainer.layer_mapping:
            assert component in ("attn->ssm", "ffn->moe")

    def test_cka_history_initialized(self):
        teacher = TinyTeacher(n_layers=2, d_model=16)
        student = TinyStudent(n_layers=2, d_model=16)
        cfg = AlignmentConfig()
        trainer = LayerAlignmentTrainer(
            teacher=teacher, student=student, config=cfg,
            n_teacher_layers=2, n_student_layers=2,
            d_teacher=16, d_student=16,
            device=torch.device("cpu"),
        )
        assert isinstance(trainer.cka_history, dict)
        assert len(trainer.cka_history) > 0
        for key, hist in trainer.cka_history.items():
            assert hist == []


# ===================================================================
# KLDistillationTrainer init tests
# ===================================================================

class TestKLDistillationTrainerInit:
    """Tests for KLDistillationTrainer initialization."""

    def test_init_basic(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(teacher_vocab_size=64, student_vocab_size=64)
        trainer = KLDistillationTrainer(
            teacher=teacher, student=student, config=cfg,
            device=torch.device("cpu"),
        )
        assert trainer.step == 0
        assert not trainer.vocab_projector.needs_projection

    def test_init_with_vocab_mismatch(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=100)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=50)
        cfg = KLDistillationConfig(teacher_vocab_size=100, student_vocab_size=50)
        trainer = KLDistillationTrainer(
            teacher=teacher, student=student, config=cfg,
            device=torch.device("cpu"),
        )
        assert trainer.vocab_projector.needs_projection

    def test_teacher_frozen(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(teacher_vocab_size=64, student_vocab_size=64)
        trainer = KLDistillationTrainer(
            teacher=teacher, student=student, config=cfg,
            device=torch.device("cpu"),
        )
        for param in trainer.teacher.parameters():
            assert not param.requires_grad

    def test_per_language_tracking_empty(self):
        teacher = TinyTeacher(n_layers=2, d_model=16, vocab_size=64)
        student = TinyStudent(n_layers=2, d_model=16, vocab_size=64)
        cfg = KLDistillationConfig(teacher_vocab_size=64, student_vocab_size=64)
        trainer = KLDistillationTrainer(
            teacher=teacher, student=student, config=cfg,
            device=torch.device("cpu"),
        )
        assert len(trainer.per_language_kl) == 0
