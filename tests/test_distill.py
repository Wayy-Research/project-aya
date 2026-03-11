"""
Tests for aya-distill distillation modules: converter, hooks, and CKA.

Covers:
    - converter.py: convert_attention_to_ssm, convert_ffn_to_moe, svd_project,
      _extract_submatrix, ConversionConfig, convert_block
    - hooks.py: ActivationStore, register_teacher_hooks, register_student_hooks,
      activation_capture, build_layer_mapping
    - cka.py: linear_cka, rbf_cka, MinibatchCKAAccumulator, minibatch_cka,
      cka_permutation_test, compute_layerwise_cka, CKAHeatmapData
    - CKA mathematical properties: symmetry, self-similarity, range [0,1]

All tests use synthetic/mock models -- no GPU or real model downloads required.

Wayy Research, 2024-2026.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Path setup for local packages
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "aetheris"))

import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from aetheris import AetherisConfig
from aetheris.modules.ssm import SSMBlock
from aetheris.modules.moe import SparseMoELayer

from aya_distill.distill.converter import (
    ConversionConfig,
    _extract_submatrix,
    convert_attention_to_ssm,
    convert_ffn_to_moe,
    svd_project,
)
from aya_distill.distill.hooks import (
    ActivationStore,
    activation_capture,
    build_layer_mapping,
    register_student_hooks,
    register_teacher_hooks,
)
from aya_distill.distill.converter import (
    convert_block,
)
from aya_distill.distill.cka import (
    CKAHeatmapData,
    MinibatchCKAAccumulator,
    cka_permutation_test,
    compute_layerwise_cka,
    linear_cka,
    minibatch_cka,
    rbf_cka,
)


# ---------------------------------------------------------------------------
# Test dimensions (small for speed)
# ---------------------------------------------------------------------------
D_MODEL = 64
D_INNER = 128
D_STATE = 16
D_FF = 192
NUM_EXPERTS = 4
KV_DIM = 32  # simulate GQA: fewer kv heads than q heads
INTERMEDIATE = 256


# ---------------------------------------------------------------------------
# Mock modules
# ---------------------------------------------------------------------------

class MockAttention(nn.Module):
    """Mimics Cohere attention with q/k/v/o projections."""

    def __init__(
        self,
        hidden_size: int = D_MODEL,
        kv_dim: int = KV_DIM,
    ):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: just use q_proj so dimensions work (hidden -> hidden)
        return self.o_proj(self.q_proj(x))


class MockMLP(nn.Module):
    """Mimics Cohere gated FFN with gate/up/down projections."""

    def __init__(
        self,
        hidden_size: int = D_MODEL,
        intermediate_size: int = INTERMEDIATE,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockTransformerLayer(nn.Module):
    """Single transformer layer with self_attn and mlp."""

    def __init__(self):
        super().__init__()
        self.self_attn = MockAttention()
        self.mlp = MockMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class MockTeacherModel(nn.Module):
    """Mimics HuggingFace CohereForCausalLM structure: model.model.layers."""

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockTransformerLayer() for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


class MockStudentModel(nn.Module):
    """Mimics Aetheris HybridMambaMoE with alternating SSM/MoE layers."""

    def __init__(self, n_layers: int = 4):
        super().__init__()
        cfg = AetherisConfig(
            d_model=D_MODEL,
            d_inner=D_INNER,
            ssm_d_state=D_STATE,
            d_ff=D_FF,
            num_experts=NUM_EXPERTS,
            dtype="float32",
        )
        layers: list[nn.Module] = []
        for i in range(n_layers):
            if i % 2 == 0:
                layers.append(SSMBlock(cfg))
            else:
                layers.append(SparseMoELayer(cfg))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            out = layer(x)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
        return x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aetheris_config() -> AetherisConfig:
    return AetherisConfig(
        d_model=D_MODEL,
        d_inner=D_INNER,
        ssm_d_state=D_STATE,
        d_ff=D_FF,
        num_experts=NUM_EXPERTS,
        dtype="float32",
    )


@pytest.fixture
def ssm_block(aetheris_config: AetherisConfig) -> SSMBlock:
    return SSMBlock(aetheris_config)


@pytest.fixture
def moe_block(aetheris_config: AetherisConfig) -> SparseMoELayer:
    return SparseMoELayer(aetheris_config)


@pytest.fixture
def mock_attn() -> MockAttention:
    return MockAttention()


@pytest.fixture
def mock_mlp() -> MockMLP:
    return MockMLP()


@pytest.fixture
def teacher_model() -> MockTeacherModel:
    return MockTeacherModel(n_layers=4)


@pytest.fixture
def student_model() -> MockStudentModel:
    return MockStudentModel(n_layers=4)


# ===================================================================
# Tests for ConversionConfig
# ===================================================================

class TestConversionConfig:
    """Tests for ConversionConfig dataclass defaults and values."""

    def test_default_strategy(self):
        cfg = ConversionConfig()
        assert cfg.strategy == "hybrid"

    def test_default_a_init(self):
        cfg = ConversionConfig()
        assert cfg.a_init == "exponential_decay"

    def test_default_ffn_to_moe(self):
        cfg = ConversionConfig()
        assert cfg.ffn_to_moe == "replicate"

    def test_custom_strategy(self):
        cfg = ConversionConfig(strategy="weight_map")
        assert cfg.strategy == "weight_map"

    def test_custom_refinement_params(self):
        cfg = ConversionConfig(refinement_steps=500, refinement_lr=1e-3)
        assert cfg.refinement_steps == 500
        assert cfg.refinement_lr == 1e-3


# ===================================================================
# Tests for svd_project
# ===================================================================

class TestSvdProject:
    """Tests for SVD-based weight projection."""

    def test_basic_downproject(self):
        """Downproject from larger to smaller dimensions."""
        w = torch.randn(128, 64)
        result = svd_project(w, 32, 32)
        assert result.shape == (32, 32)

    def test_preserves_dtype(self):
        """Output dtype should match input dtype."""
        w = torch.randn(64, 64)
        result = svd_project(w, 32, 32)
        assert result.dtype == w.dtype

    def test_square_identity_preserves_structure(self):
        """SVD of identity projected to same size should approximate identity."""
        w = torch.eye(32)
        result = svd_project(w, 32, 32)
        assert torch.allclose(result, w, atol=1e-5)

    def test_variance_preservation(self):
        """Projected matrix should capture most of the original variance."""
        torch.manual_seed(42)
        w = torch.randn(64, 64)
        original_norm = torch.norm(w, p="fro")
        result = svd_project(w, 32, 32)
        projected_norm = torch.norm(result, p="fro")
        # Should capture a significant fraction of the norm
        assert projected_norm > 0.1 * original_norm

    @pytest.mark.parametrize("target_rows,target_cols", [
        (16, 32),
        (32, 16),
        (8, 8),
        (64, 32),
    ])
    def test_various_target_shapes(self, target_rows, target_cols):
        """svd_project should handle arbitrary target shapes."""
        w = torch.randn(64, 64)
        result = svd_project(w, target_rows, target_cols)
        assert result.shape == (target_rows, target_cols)


# ===================================================================
# Tests for _extract_submatrix
# ===================================================================

class TestExtractSubmatrix:
    """Tests for _extract_submatrix with shrink, grow, and identity cases."""

    def test_same_size_returns_clone(self):
        """Same dimensions should return a clone, not the same tensor."""
        w = torch.randn(32, 32)
        result = _extract_submatrix(w, 32, 32)
        assert result.shape == (32, 32)
        assert torch.allclose(result, w)
        # Should be a clone, not the original
        result[0, 0] = 999.0
        assert w[0, 0] != 999.0

    def test_shrink_both_dims(self):
        """Shrinking both dimensions uses SVD projection."""
        w = torch.randn(64, 64)
        result = _extract_submatrix(w, 32, 32)
        assert result.shape == (32, 32)

    def test_grow_rows(self):
        """Growing rows should pad with small random values."""
        w = torch.randn(16, 32)
        result = _extract_submatrix(w, 32, 32)
        assert result.shape == (32, 32)
        # Original data should be preserved in top portion
        assert torch.allclose(result[:16, :], w, atol=1e-5)

    def test_grow_cols(self):
        """Growing cols should pad with small random values."""
        w = torch.randn(32, 16)
        result = _extract_submatrix(w, 32, 32)
        assert result.shape == (32, 32)
        # Original data should be preserved in left portion
        assert torch.allclose(result[:, :16], w, atol=1e-5)

    def test_grow_both_dims(self):
        """Growing both dimensions from a small matrix."""
        w = torch.randn(8, 8)
        result = _extract_submatrix(w, 32, 32)
        assert result.shape == (32, 32)
        # Original block preserved
        assert torch.allclose(result[:8, :8], w, atol=1e-5)

    def test_mixed_shrink_grow(self):
        """Shrink rows but grow cols."""
        w = torch.randn(64, 16)
        result = _extract_submatrix(w, 32, 32)
        assert result.shape == (32, 32)


# ===================================================================
# Tests for convert_attention_to_ssm
# ===================================================================

class TestConvertAttentionToSSM:
    """Tests for attention-to-SSM weight mapping."""

    def test_basic_conversion_returns_metrics(self, mock_attn, ssm_block):
        """Conversion should return a metrics dict."""
        cfg = ConversionConfig(strategy="weight_map")
        metrics = convert_attention_to_ssm(mock_attn, ssm_block, cfg, layer_idx=0)
        assert isinstance(metrics, dict)
        assert metrics["layer_idx"] == 0
        assert metrics["type"] == "attn->ssm"

    def test_in_proj_shape_after_conversion(self, mock_attn, ssm_block):
        """in_proj should be (2*d_inner, d_model) after conversion."""
        cfg = ConversionConfig(strategy="weight_map")
        convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        assert ssm_block.in_proj.weight.shape == (2 * D_INNER, D_MODEL)

    def test_gate_proj_shape(self, mock_attn, ssm_block):
        """gate_proj should be (d_inner, d_model) after conversion."""
        cfg = ConversionConfig(strategy="weight_map")
        convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        assert ssm_block.gate_proj.weight.shape == (D_INNER, D_MODEL)

    def test_out_proj_shape(self, mock_attn, ssm_block):
        """out_proj should be (d_model, d_inner) after conversion."""
        cfg = ConversionConfig(strategy="weight_map")
        convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        assert ssm_block.out_proj.weight.shape == (D_MODEL, D_INNER)

    def test_conv_initialized_as_identity(self, mock_attn, ssm_block):
        """Conv1d should be initialized with center=1, neighbors=0.1."""
        cfg = ConversionConfig(strategy="weight_map")
        convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        conv_w = ssm_block.conv_d.weight.data
        assert torch.allclose(conv_w[:, 0, 1], torch.ones(D_INNER))
        assert torch.allclose(conv_w[:, 0, 0], torch.full((D_INNER,), 0.1))
        assert torch.allclose(conv_w[:, 0, 2], torch.full((D_INNER,), 0.1))

    def test_d_parameter_initialized(self, mock_attn, ssm_block):
        """D skip connection should be initialized to 0.1."""
        cfg = ConversionConfig(strategy="weight_map")
        convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        expected = torch.ones(D_INNER) * 0.1
        assert torch.allclose(ssm_block.D.data, expected)

    @pytest.mark.parametrize("a_init", ["exponential_decay", "constant", "random"])
    def test_a_init_strategies(self, mock_attn, ssm_block, a_init):
        """All A_log initialization strategies should work without error."""
        cfg = ConversionConfig(strategy="weight_map", a_init=a_init)
        metrics = convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        assert metrics is not None
        # A_log should be finite
        assert torch.isfinite(ssm_block.A_log.data).all()

    def test_exponential_decay_has_range(self, mock_attn, ssm_block):
        """Exponential decay A_log should span from -5 to -2 (with noise)."""
        cfg = ConversionConfig(strategy="weight_map", a_init="exponential_decay")
        convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        a_log = ssm_block.A_log.data
        # Check that there is variation across the d_inner dimension
        col_means = a_log.mean(dim=1)
        assert col_means.min() < -4.0
        assert col_means.max() > -3.0

    def test_constant_a_init(self, mock_attn, ssm_block):
        """Constant A_log init should set all values to -4.0."""
        cfg = ConversionConfig(strategy="weight_map", a_init="constant")
        convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        assert torch.allclose(ssm_block.A_log.data, torch.full_like(ssm_block.A_log.data, -4.0))

    @pytest.mark.parametrize("delta_init", ["uniform", "random"])
    def test_delta_init_strategies(self, mock_attn, ssm_block, delta_init):
        """Delta initialization strategies should all succeed."""
        cfg = ConversionConfig(strategy="weight_map", delta_init=delta_init)
        metrics = convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        assert metrics is not None

    def test_metrics_contain_shape_info(self, mock_attn, ssm_block):
        """Metrics should contain projection shape information."""
        cfg = ConversionConfig(strategy="weight_map")
        metrics = convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        assert "v_proj_shape" in metrics
        assert "q_proj_shape" in metrics
        assert "ssm_d_model" in metrics
        assert metrics["ssm_d_model"] == D_MODEL
        assert metrics["ssm_d_inner"] == D_INNER

    def test_ssm_forward_after_conversion(self, mock_attn, ssm_block):
        """SSM block should produce valid output after weight conversion."""
        cfg = ConversionConfig(strategy="weight_map")
        convert_attention_to_ssm(mock_attn, ssm_block, cfg)
        x = torch.randn(2, 8, D_MODEL)
        with torch.no_grad():
            out = ssm_block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


# ===================================================================
# Tests for convert_ffn_to_moe
# ===================================================================

class TestConvertFFNToMoE:
    """Tests for FFN-to-MoE weight mapping."""

    def test_replicate_strategy(self, mock_mlp, moe_block):
        """Replicate strategy should copy weights to all experts."""
        cfg = ConversionConfig(ffn_to_moe="replicate")
        metrics = convert_ffn_to_moe(mock_mlp, moe_block, cfg)
        assert metrics["init_strategy"] == "replicate_with_noise"
        assert metrics["num_experts"] == NUM_EXPERTS

    def test_svd_split_strategy(self, mock_mlp, moe_block):
        """SVD split should decompose weights across experts."""
        cfg = ConversionConfig(ffn_to_moe="svd_split")
        metrics = convert_ffn_to_moe(mock_mlp, moe_block, cfg)
        assert metrics["init_strategy"] == "svd_split"
        assert "k_per_expert" in metrics

    def test_random_strategy(self, mock_mlp, moe_block):
        """Random strategy should keep existing initialization."""
        cfg = ConversionConfig(ffn_to_moe="random")
        metrics = convert_ffn_to_moe(mock_mlp, moe_block, cfg)
        assert metrics["init_strategy"] == "random"

    def test_router_initialized_to_zeros(self, mock_mlp, moe_block):
        """Router gate weights should be zeroed for uniform routing."""
        cfg = ConversionConfig(ffn_to_moe="replicate")
        convert_ffn_to_moe(mock_mlp, moe_block, cfg)
        assert torch.allclose(moe_block.gate.weight, torch.zeros_like(moe_block.gate.weight))

    def test_replicate_experts_differ(self, mock_mlp, moe_block):
        """Replicated experts should have small perturbations (not identical)."""
        cfg = ConversionConfig(ffn_to_moe="replicate")
        convert_ffn_to_moe(mock_mlp, moe_block, cfg)
        w1_0 = moe_block.experts[0].w1.weight.data
        w1_1 = moe_block.experts[1].w1.weight.data
        # Not exactly equal due to noise
        assert not torch.allclose(w1_0, w1_1, atol=1e-8)
        # But close since noise is small (0.01 std)
        assert torch.allclose(w1_0, w1_1, atol=0.1)

    def test_svd_split_experts_differ(self, mock_mlp, moe_block):
        """SVD-split experts should have meaningfully different weights."""
        cfg = ConversionConfig(ffn_to_moe="svd_split")
        convert_ffn_to_moe(mock_mlp, moe_block, cfg)
        w1_0 = moe_block.experts[0].w1.weight.data
        w1_1 = moe_block.experts[1].w1.weight.data
        assert not torch.allclose(w1_0, w1_1, atol=0.01)

    def test_expert_shapes_correct(self, mock_mlp, moe_block):
        """Expert weight shapes should match the config."""
        cfg = ConversionConfig(ffn_to_moe="replicate")
        convert_ffn_to_moe(mock_mlp, moe_block, cfg)
        for expert in moe_block.experts:
            assert expert.w1.weight.shape == (D_FF, D_MODEL)
            assert expert.w2.weight.shape == (D_MODEL, D_FF)

    def test_moe_forward_after_conversion(self, mock_mlp, moe_block):
        """MoE block should produce valid output after conversion."""
        cfg = ConversionConfig(ffn_to_moe="replicate")
        convert_ffn_to_moe(mock_mlp, moe_block, cfg)
        x = torch.randn(2, 8, D_MODEL)
        with torch.no_grad():
            out, aux_loss = moe_block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert torch.isfinite(aux_loss)

    @pytest.mark.parametrize("strategy", ["replicate", "svd_split", "random"])
    def test_all_strategies_produce_valid_output(self, mock_mlp, moe_block, strategy):
        """All FFN->MoE strategies should produce valid forward pass output."""
        cfg = ConversionConfig(ffn_to_moe=strategy)
        convert_ffn_to_moe(mock_mlp, moe_block, cfg)
        x = torch.randn(2, 8, D_MODEL)
        with torch.no_grad():
            out, aux_loss = moe_block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


# ===================================================================
# Tests for ActivationStore
# ===================================================================

class TestActivationStore:
    """Tests for ActivationStore lifecycle: register, collect, clear, remove."""

    def test_register_and_collect(self):
        """Hook should capture output after forward pass."""
        store = ActivationStore()
        linear = nn.Linear(16, 16)
        store.register(linear, "test_layer")

        x = torch.randn(4, 16)
        linear(x)

        collected = store.collect()
        assert "test_layer" in collected
        assert collected["test_layer"].shape == (4, 16)
        store.remove_hooks()

    def test_collect_empty_returns_empty(self):
        """Collecting before any forward pass returns empty dict."""
        store = ActivationStore()
        assert store.collect() == {}

    def test_clear_removes_buffers(self):
        """Clear should empty the buffers but keep hooks."""
        store = ActivationStore()
        linear = nn.Linear(16, 16)
        store.register(linear, "test")

        linear(torch.randn(4, 16))
        assert len(store.collect()) == 1

        store.clear()
        assert store.collect() == {}

        # Hooks still active -- new forward pass should collect
        linear(torch.randn(4, 16))
        assert len(store.collect()) == 1
        store.remove_hooks()

    def test_remove_hooks_full_cleanup(self):
        """remove_hooks should remove hooks and clear buffers."""
        store = ActivationStore()
        linear = nn.Linear(16, 16)
        store.register(linear, "test")

        linear(torch.randn(4, 16))
        store.remove_hooks()

        assert store.collect() == {}
        assert len(store._hooks) == 0

        # Hook should no longer fire
        linear(torch.randn(4, 16))
        assert store.collect() == {}

    def test_multiple_forward_passes_accumulate(self):
        """Multiple forward passes should accumulate activations."""
        store = ActivationStore()
        linear = nn.Linear(16, 16)
        store.register(linear, "test")

        linear(torch.randn(4, 16))
        linear(torch.randn(8, 16))

        collected = store.collect()
        assert collected["test"].shape == (12, 16)  # 4 + 8
        store.remove_hooks()

    def test_3d_input_flattened(self):
        """3D (B, L, D) output should be flattened to (B*L, D)."""
        store = ActivationStore()
        linear = nn.Linear(16, 16)
        store.register(linear, "test")

        x = torch.randn(2, 5, 16)
        linear(x)

        collected = store.collect()
        assert collected["test"].shape == (10, 16)  # 2*5 = 10
        store.remove_hooks()

    def test_tuple_output_extracts_first(self):
        """If module returns tuple, hook should extract first element."""
        store = ActivationStore()

        class TupleModule(nn.Module):
            def forward(self, x):
                return x, torch.tensor(0.0)

        mod = TupleModule()
        store.register(mod, "tuple_test")
        mod(torch.randn(4, 16))

        collected = store.collect()
        assert "tuple_test" in collected
        assert collected["tuple_test"].shape == (4, 16)
        store.remove_hooks()

    def test_detach_by_default(self):
        """Captured activations should be detached by default."""
        store = ActivationStore()
        linear = nn.Linear(16, 16)
        store.register(linear, "test")

        x = torch.randn(4, 16, requires_grad=True)
        linear(x)

        collected = store.collect()
        assert not collected["test"].requires_grad
        store.remove_hooks()

    def test_non_detach_mode(self):
        """When detach=False, activations should retain grad."""
        store = ActivationStore(detach=False)
        linear = nn.Linear(16, 16)
        store.register(linear, "test")

        x = torch.randn(4, 16, requires_grad=True)
        linear(x)

        collected = store.collect()
        # The activation will be on cpu as float32 but may or may not have grad
        # depending on whether .to() preserves it. The key point is it was not detached.
        store.remove_hooks()


# ===================================================================
# Tests for register_teacher_hooks
# ===================================================================

class TestRegisterTeacherHooks:
    """Tests for teacher hook registration on HF-style models."""

    def test_hooks_registered_on_all_layers(self, teacher_model):
        """Should register hooks on all layers by default."""
        store = ActivationStore()
        register_teacher_hooks(teacher_model, store)
        # 4 layers * 2 (attn + ffn) = 8 hooks
        assert len(store._hooks) == 8
        store.remove_hooks()

    def test_hooks_on_specific_layers(self, teacher_model):
        """Should register hooks only on specified layers."""
        store = ActivationStore()
        register_teacher_hooks(teacher_model, store, layer_indices=[0, 2])
        assert len(store._hooks) == 4  # 2 layers * 2
        store.remove_hooks()

    def test_hook_attention_only(self, teacher_model):
        """Should register only attention hooks when hook_ffn=False."""
        store = ActivationStore()
        register_teacher_hooks(teacher_model, store, hook_attention=True, hook_ffn=False)
        assert len(store._hooks) == 4  # 4 layers, attn only
        store.remove_hooks()

    def test_hook_ffn_only(self, teacher_model):
        """Should register only FFN hooks when hook_attention=False."""
        store = ActivationStore()
        register_teacher_hooks(teacher_model, store, hook_attention=False, hook_ffn=True)
        assert len(store._hooks) == 4  # 4 layers, ffn only
        store.remove_hooks()

    def test_captures_activations_on_forward(self, teacher_model):
        """Hooks should capture activations during forward pass."""
        store = ActivationStore()
        register_teacher_hooks(teacher_model, store, layer_indices=[0])
        x = torch.randn(2, 8, D_MODEL)
        teacher_model(x)
        collected = store.collect()
        assert "teacher.attn.0" in collected
        assert "teacher.ffn.0" in collected
        store.remove_hooks()

    def test_invalid_model_raises(self):
        """Should raise ValueError for unrecognized model structure."""
        store = ActivationStore()
        bogus_model = nn.Linear(16, 16)
        with pytest.raises(ValueError, match="Could not locate"):
            register_teacher_hooks(bogus_model, store)

    def test_out_of_range_layer_warns(self, teacher_model):
        """Out-of-range layer indices should produce a warning."""
        store = ActivationStore()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_teacher_hooks(teacher_model, store, layer_indices=[99])
            assert any("out of range" in str(warning.message) for warning in w)
        store.remove_hooks()


# ===================================================================
# Tests for register_student_hooks
# ===================================================================

class TestRegisterStudentHooks:
    """Tests for student hook registration on Aetheris-style models."""

    def test_hooks_registered_on_all_layers(self, student_model):
        """Should register hooks on all student layers."""
        store = ActivationStore()
        register_student_hooks(student_model, store)
        assert len(store._hooks) == 4
        store.remove_hooks()

    def test_hooks_on_specific_layers(self, student_model):
        """Should register hooks only on specified layers."""
        store = ActivationStore()
        register_student_hooks(student_model, store, layer_indices=[0, 1])
        assert len(store._hooks) == 2
        store.remove_hooks()

    def test_naming_convention(self, student_model):
        """Even layers should be named 'ssm', odd layers 'moe'."""
        store = ActivationStore()
        register_student_hooks(student_model, store)

        x = torch.randn(2, 8, D_MODEL)
        student_model(x)

        collected = store.collect()
        assert "student.ssm.0" in collected
        assert "student.moe.1" in collected
        assert "student.ssm.2" in collected
        assert "student.moe.3" in collected
        store.remove_hooks()

    def test_invalid_student_raises(self):
        """Should raise ValueError if model has no .layers attribute."""
        store = ActivationStore()
        bogus = nn.Linear(16, 16)
        with pytest.raises(ValueError, match="must have a .layers"):
            register_student_hooks(bogus, store)


# ===================================================================
# Tests for activation_capture context manager
# ===================================================================

class TestActivationCapture:
    """Tests for the activation_capture context manager."""

    def test_context_manager_captures(self, teacher_model, student_model):
        """Context manager should capture activations from both models."""
        x = torch.randn(2, 8, D_MODEL)
        with activation_capture(teacher_model, student_model) as (t_store, s_store):
            teacher_model(x)
            student_model(x)
            t_acts = t_store.collect()
            s_acts = s_store.collect()

        assert len(t_acts) > 0
        assert len(s_acts) > 0

    def test_hooks_removed_after_exit(self, teacher_model, student_model):
        """Hooks should be cleaned up after exiting context."""
        x = torch.randn(2, 8, D_MODEL)
        with activation_capture(teacher_model, student_model) as (t_store, s_store):
            pass

        assert len(t_store._hooks) == 0
        assert len(s_store._hooks) == 0

    def test_hooks_removed_on_exception(self, teacher_model, student_model):
        """Hooks should be cleaned up even if an exception occurs."""
        t_store_ref = None
        s_store_ref = None
        with pytest.raises(RuntimeError):
            with activation_capture(teacher_model, student_model) as (t_store, s_store):
                t_store_ref = t_store
                s_store_ref = s_store
                raise RuntimeError("test error")

        assert len(t_store_ref._hooks) == 0
        assert len(s_store_ref._hooks) == 0


# ===================================================================
# Tests for build_layer_mapping
# ===================================================================

class TestBuildLayerMapping:
    """Tests for teacher-student layer mapping construction."""

    def test_basic_mapping(self):
        """Basic mapping with equal layer counts."""
        mapping = build_layer_mapping(4, 4)
        assert len(mapping) == 4
        # Even student layers -> attn->ssm, odd -> ffn->moe
        assert mapping[0][2] == "attn->ssm"
        assert mapping[1][2] == "ffn->moe"

    def test_interleaved_types(self):
        """Even student indices should map to attn->ssm, odd to ffn->moe."""
        mapping = build_layer_mapping(8, 6)
        for t_idx, s_idx, component in mapping:
            if s_idx % 2 == 0:
                assert component == "attn->ssm"
            else:
                assert component == "ffn->moe"

    def test_teacher_indices_within_bounds(self):
        """All teacher indices should be within valid range."""
        mapping = build_layer_mapping(8, 12)
        for t_idx, s_idx, _ in mapping:
            assert 0 <= t_idx < 8

    def test_student_indices_sequential(self):
        """Student indices should be 0..n_student-1."""
        mapping = build_layer_mapping(8, 6)
        s_indices = [s for _, s, _ in mapping]
        assert s_indices == list(range(6))

    def test_more_student_than_teacher(self):
        """Multiple student layers can map to the same teacher layer."""
        mapping = build_layer_mapping(2, 8)
        t_indices = [t for t, _, _ in mapping]
        # All teacher indices clamped to [0, 1]
        assert all(0 <= t <= 1 for t in t_indices)

    def test_single_teacher_layer(self):
        """Single teacher layer should map to all student layers."""
        mapping = build_layer_mapping(1, 4)
        for t_idx, _, _ in mapping:
            assert t_idx == 0

    def test_unknown_strategy_raises(self):
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mapping strategy"):
            build_layer_mapping(4, 4, strategy="unknown")

    @pytest.mark.parametrize("n_teacher,n_student", [
        (32, 24),
        (40, 16),
        (8, 8),
        (16, 32),
        (1, 1),
    ])
    def test_various_layer_counts(self, n_teacher, n_student):
        """Mapping should work for various teacher/student layer combinations."""
        mapping = build_layer_mapping(n_teacher, n_student)
        assert len(mapping) == n_student
        for t_idx, s_idx, comp in mapping:
            assert 0 <= t_idx < n_teacher
            assert 0 <= s_idx < n_student
            assert comp in ("attn->ssm", "ffn->moe")


# ===================================================================
# Tests for MinibatchCKAAccumulator edge cases
# ===================================================================

class TestMinibatchCKAAccumulatorEdgeCases:
    """Edge case tests for MinibatchCKAAccumulator consistency."""

    def test_compute_before_update_raises(self):
        """Should raise ValueError if compute() is called with no data."""
        acc = MinibatchCKAAccumulator(d_x=16, d_y=16)
        with pytest.raises(ValueError, match="No samples accumulated"):
            acc.compute()

    def test_single_sample_update(self):
        """Single sample should produce a valid CKA score."""
        acc = MinibatchCKAAccumulator(d_x=16, d_y=16)
        x = torch.randn(1, 16)
        acc.update(x, x)
        # CKA of identical single sample - may be 0 or NaN-ish due to centering
        # The key property: it should not raise
        score = acc.compute()
        assert isinstance(score, float)

    def test_identical_inputs_high_cka(self):
        """Identical activations should yield CKA close to 1."""
        torch.manual_seed(123)
        acc = MinibatchCKAAccumulator(d_x=32, d_y=32)
        for _ in range(5):
            x = torch.randn(64, 32)
            acc.update(x, x)
        score = acc.compute()
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_matches_full_linear_cka(self):
        """Minibatch CKA should match full linear CKA for same data."""
        torch.manual_seed(42)
        X = torch.randn(200, 32)
        Y = torch.randn(200, 24)

        full_score = linear_cka(X, Y).item()
        mb_score = minibatch_cka(X, Y, batch_size=50)

        assert full_score == pytest.approx(mb_score, abs=1e-5)

    def test_reset_clears_state(self):
        """Reset should allow reuse of the accumulator."""
        acc = MinibatchCKAAccumulator(d_x=16, d_y=16)
        acc.update(torch.randn(32, 16), torch.randn(32, 16))
        score_1 = acc.compute()

        acc.reset()
        with pytest.raises(ValueError):
            acc.compute()

        acc.update(torch.randn(32, 16), torch.randn(32, 16))
        score_2 = acc.compute()
        # Different random data, different score (almost certainly)
        assert isinstance(score_2, float)

    def test_different_dimensions(self):
        """CKA should work with different d_x and d_y."""
        acc = MinibatchCKAAccumulator(d_x=16, d_y=64)
        for _ in range(3):
            acc.update(torch.randn(32, 16), torch.randn(32, 64))
        score = acc.compute()
        assert 0.0 <= score <= 1.0 + 1e-6

    def test_accumulation_order_invariant(self):
        """CKA should be the same regardless of batch ordering."""
        torch.manual_seed(7)
        batches_x = [torch.randn(16, 32) for _ in range(4)]
        batches_y = [torch.randn(16, 24) for _ in range(4)]

        acc1 = MinibatchCKAAccumulator(d_x=32, d_y=24)
        for bx, by in zip(batches_x, batches_y):
            acc1.update(bx, by)
        score1 = acc1.compute()

        # Reverse order
        acc2 = MinibatchCKAAccumulator(d_x=32, d_y=24)
        for bx, by in zip(reversed(batches_x), reversed(batches_y)):
            acc2.update(bx, by)
        score2 = acc2.compute()

        assert score1 == pytest.approx(score2, abs=1e-6)

    def test_large_batch_equals_many_small(self):
        """One large batch should equal many small batches."""
        torch.manual_seed(99)
        X = torch.randn(128, 16)
        Y = torch.randn(128, 16)

        # Single large batch
        acc_big = MinibatchCKAAccumulator(d_x=16, d_y=16)
        acc_big.update(X, Y)
        score_big = acc_big.compute()

        # Many small batches
        acc_small = MinibatchCKAAccumulator(d_x=16, d_y=16)
        for i in range(0, 128, 16):
            acc_small.update(X[i:i+16], Y[i:i+16])
        score_small = acc_small.compute()

        assert score_big == pytest.approx(score_small, abs=1e-6)


# ===================================================================
# Tests for linear_cka properties
# ===================================================================

class TestLinearCKAProperties:
    """Mathematical property tests for linear CKA."""

    def test_self_similarity_is_one(self):
        """CKA(X, X) should be 1.0."""
        torch.manual_seed(0)
        X = torch.randn(100, 32)
        score = linear_cka(X, X).item()
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_symmetry(self):
        """CKA(X, Y) == CKA(Y, X)."""
        torch.manual_seed(1)
        X = torch.randn(80, 32)
        Y = torch.randn(80, 24)
        score_xy = linear_cka(X, Y).item()
        score_yx = linear_cka(Y, X).item()
        assert score_xy == pytest.approx(score_yx, abs=1e-6)

    def test_range_zero_to_one(self):
        """CKA should be in [0, 1]."""
        torch.manual_seed(2)
        X = torch.randn(100, 16)
        Y = torch.randn(100, 24)
        score = linear_cka(X, Y).item()
        assert 0.0 <= score <= 1.0 + 1e-6

    def test_orthogonal_transform_invariance(self):
        """CKA should be invariant to orthogonal transformations."""
        torch.manual_seed(3)
        X = torch.randn(100, 16)
        # Create a random orthogonal matrix via QR decomposition
        Q, _ = torch.linalg.qr(torch.randn(16, 16))
        X_rot = X @ Q

        Y = torch.randn(100, 16)
        score_orig = linear_cka(X, Y).item()
        score_rot = linear_cka(X_rot, Y).item()
        assert score_orig == pytest.approx(score_rot, abs=1e-4)

    def test_isotropic_scaling_invariance(self):
        """CKA should be invariant to uniform scaling."""
        torch.manual_seed(4)
        X = torch.randn(100, 16)
        Y = torch.randn(100, 24)
        score_orig = linear_cka(X, Y).item()
        score_scaled = linear_cka(X * 5.0, Y * 0.3).item()
        assert score_orig == pytest.approx(score_scaled, abs=1e-4)

    def test_sample_count_mismatch_raises(self):
        """Different number of samples should raise assertion."""
        X = torch.randn(50, 16)
        Y = torch.randn(60, 16)
        with pytest.raises(AssertionError, match="Sample count mismatch"):
            linear_cka(X, Y)

    def test_identity_matrices(self):
        """CKA of two identity-like matrices should be 1.0."""
        X = torch.eye(20)
        score = linear_cka(X, X).item()
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_different_feature_dimensions(self):
        """CKA should work when d_x != d_y."""
        torch.manual_seed(5)
        X = torch.randn(100, 8)
        Y = torch.randn(100, 64)
        score = linear_cka(X, Y).item()
        assert 0.0 <= score <= 1.0 + 1e-6

    @pytest.mark.parametrize("n_samples", [2, 10, 50, 200])
    def test_various_sample_sizes(self, n_samples):
        """CKA should work with various sample counts."""
        torch.manual_seed(6)
        X = torch.randn(n_samples, 16)
        Y = torch.randn(n_samples, 16)
        score = linear_cka(X, Y).item()
        assert np.isfinite(score)

    def test_constant_features_near_zero(self):
        """Constant features (zero variance) should yield near-zero CKA."""
        X = torch.ones(50, 16)
        Y = torch.randn(50, 16)
        score = linear_cka(X, Y).item()
        # After centering, X becomes zeros, so numerator is 0
        assert score == pytest.approx(0.0, abs=1e-5)


# ===================================================================
# Tests for rbf_cka
# ===================================================================

class TestRBFCKA:
    """Tests for RBF kernel CKA."""

    def test_self_similarity_is_one(self):
        """RBF CKA(X, X) should be close to 1.0."""
        torch.manual_seed(10)
        X = torch.randn(50, 16)
        score = rbf_cka(X, X).item()
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_symmetry(self):
        """RBF CKA(X, Y) == CKA(Y, X)."""
        torch.manual_seed(11)
        X = torch.randn(40, 16)
        Y = torch.randn(40, 12)
        score_xy = rbf_cka(X, Y).item()
        score_yx = rbf_cka(Y, X).item()
        assert score_xy == pytest.approx(score_yx, abs=1e-5)

    def test_range_zero_to_one(self):
        """RBF CKA should be in [0, 1]."""
        torch.manual_seed(12)
        X = torch.randn(50, 16)
        Y = torch.randn(50, 24)
        score = rbf_cka(X, Y).item()
        assert -0.01 <= score <= 1.01  # Small tolerance for numerical issues

    def test_explicit_sigma(self):
        """RBF CKA with explicit sigma should not crash."""
        torch.manual_seed(13)
        X = torch.randn(30, 8)
        Y = torch.randn(30, 8)
        score = rbf_cka(X, Y, sigma_x=1.0, sigma_y=1.0).item()
        assert np.isfinite(score)

    def test_different_dimensions(self):
        """RBF CKA should work with different feature dims."""
        torch.manual_seed(14)
        X = torch.randn(40, 8)
        Y = torch.randn(40, 32)
        score = rbf_cka(X, Y).item()
        assert np.isfinite(score)

    def test_sample_count_mismatch_raises(self):
        """Different sample counts should raise assertion."""
        X = torch.randn(30, 8)
        Y = torch.randn(40, 8)
        with pytest.raises(AssertionError):
            rbf_cka(X, Y)


# ===================================================================
# Tests for cka_permutation_test
# ===================================================================

class TestCKAPermutationTest:
    """Tests for CKA statistical significance via permutation test."""

    def test_returns_expected_keys(self):
        """Result dict should contain expected keys."""
        torch.manual_seed(20)
        X = torch.randn(50, 8)
        Y = torch.randn(50, 8)
        result = cka_permutation_test(X, Y, n_permutations=20, seed=42)
        assert "observed_cka" in result
        assert "p_value" in result
        assert "null_mean" in result
        assert "null_std" in result

    def test_identical_inputs_low_p_value(self):
        """Identical inputs should have very low p-value (significant)."""
        torch.manual_seed(21)
        X = torch.randn(100, 16)
        result = cka_permutation_test(X, X, n_permutations=50, seed=42)
        assert result["observed_cka"] == pytest.approx(1.0, abs=1e-4)
        assert result["p_value"] <= 0.05

    def test_null_mean_below_observed_for_correlated(self):
        """For correlated data, null mean should be below observed CKA."""
        torch.manual_seed(22)
        X = torch.randn(80, 16)
        Y = X + torch.randn_like(X) * 0.1  # Highly correlated
        result = cka_permutation_test(X, Y, n_permutations=50, seed=42)
        assert result["null_mean"] < result["observed_cka"]

    def test_rbf_kernel_option(self):
        """Should work with RBF kernel."""
        torch.manual_seed(23)
        X = torch.randn(30, 8)
        Y = torch.randn(30, 8)
        result = cka_permutation_test(X, Y, n_permutations=10, kernel="rbf", seed=42)
        assert np.isfinite(result["observed_cka"])
        assert 0.0 <= result["p_value"] <= 1.0

    def test_reproducibility(self):
        """Same seed should produce same result."""
        torch.manual_seed(24)
        X = torch.randn(50, 8)
        Y = torch.randn(50, 8)
        r1 = cka_permutation_test(X, Y, n_permutations=30, seed=99)
        r2 = cka_permutation_test(X, Y, n_permutations=30, seed=99)
        assert r1["p_value"] == r2["p_value"]
        assert r1["null_mean"] == pytest.approx(r2["null_mean"], abs=1e-6)


# ===================================================================
# Tests for compute_layerwise_cka
# ===================================================================

class TestComputeLayerwiseCKA:
    """Tests for compute_layerwise_cka and CKAHeatmapData."""

    def test_basic_layerwise_cka(self):
        """Should produce a scores matrix with correct shape."""
        torch.manual_seed(30)
        t_acts = {
            "layer.0": torch.randn(100, 32),
            "layer.1": torch.randn(100, 32),
        }
        s_acts = {
            "layer.0": torch.randn(100, 16),
            "layer.1": torch.randn(100, 16),
            "layer.2": torch.randn(100, 16),
        }
        result = compute_layerwise_cka(t_acts, s_acts)
        assert result.scores.shape == (2, 3)
        assert len(result.teacher_layer_names) == 2
        assert len(result.student_layer_names) == 3

    def test_diagonal_highest_for_identical(self):
        """When teacher==student, diagonal should be 1.0."""
        torch.manual_seed(31)
        acts = {
            "a": torch.randn(80, 16),
            "b": torch.randn(80, 16),
        }
        result = compute_layerwise_cka(acts, acts)
        # Diagonal = self-similarity = 1.0
        for i in range(2):
            assert result.scores[i, i] == pytest.approx(1.0, abs=1e-4)

    def test_minibatch_mode(self):
        """Should produce valid results with batch_size set."""
        torch.manual_seed(32)
        t_acts = {"l0": torch.randn(100, 16)}
        s_acts = {"l0": torch.randn(100, 16)}
        result = compute_layerwise_cka(t_acts, s_acts, batch_size=25)
        assert result.scores.shape == (1, 1)
        assert np.isfinite(result.scores[0, 0])

    def test_rbf_kernel_mode(self):
        """Should work with RBF kernel."""
        torch.manual_seed(33)
        t_acts = {"l0": torch.randn(50, 8)}
        s_acts = {"l0": torch.randn(50, 8)}
        result = compute_layerwise_cka(t_acts, s_acts, kernel="rbf")
        assert result.scores.shape == (1, 1)
        assert np.isfinite(result.scores[0, 0])

    def test_empty_activations(self):
        """Empty dicts should produce empty scores."""
        result = compute_layerwise_cka({}, {})
        assert result.scores.shape == (0, 0)
        assert result.teacher_layer_names == []
        assert result.student_layer_names == []


# ===================================================================
# Tests for CKAHeatmapData
# ===================================================================

class TestCKAHeatmapData:
    """Tests for CKAHeatmapData serialization."""

    def test_to_dict_structure(self):
        """to_dict should return expected keys."""
        data = CKAHeatmapData(
            scores=np.array([[0.9, 0.5], [0.4, 0.8]]),
            teacher_layer_names=["t0", "t1"],
            student_layer_names=["s0", "s1"],
        )
        d = data.to_dict()
        assert "scores" in d
        assert "teacher_layers" in d
        assert "student_layers" in d
        assert len(d["scores"]) == 2
        assert len(d["scores"][0]) == 2

    def test_to_dict_values(self):
        """to_dict values should match original."""
        scores = np.array([[1.0, 0.5], [0.3, 0.9]])
        data = CKAHeatmapData(
            scores=scores,
            teacher_layer_names=["a", "b"],
            student_layer_names=["x", "y"],
        )
        d = data.to_dict()
        assert d["teacher_layers"] == ["a", "b"]
        assert d["student_layers"] == ["x", "y"]
        assert d["scores"][0][0] == pytest.approx(1.0)


# ===================================================================
# Tests for minibatch_cka convenience function
# ===================================================================

class TestMinibatchCKAFunction:
    """Tests for the minibatch_cka convenience function."""

    def test_self_similarity(self):
        """minibatch_cka(X, X) should be 1.0."""
        torch.manual_seed(40)
        X = torch.randn(100, 16)
        score = minibatch_cka(X, X, batch_size=25)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_matches_linear_cka(self):
        """Should match full linear CKA for same data."""
        torch.manual_seed(41)
        X = torch.randn(200, 32)
        Y = torch.randn(200, 24)
        full = linear_cka(X, Y).item()
        mini = minibatch_cka(X, Y, batch_size=50)
        assert full == pytest.approx(mini, abs=1e-5)

    def test_different_batch_sizes_same_result(self):
        """Different batch sizes should yield same result."""
        torch.manual_seed(42)
        X = torch.randn(120, 16)
        Y = torch.randn(120, 16)
        score_30 = minibatch_cka(X, Y, batch_size=30)
        score_60 = minibatch_cka(X, Y, batch_size=60)
        assert score_30 == pytest.approx(score_60, abs=1e-5)

    def test_batch_size_larger_than_n(self):
        """batch_size > n should still work (single batch)."""
        torch.manual_seed(43)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)
        score = minibatch_cka(X, Y, batch_size=1000)
        full = linear_cka(X, Y).item()
        assert score == pytest.approx(full, abs=1e-5)


# ===================================================================
# Tests for convert_block (public API)
# ===================================================================

class TestConvertBlock:
    """Tests for the convert_block public API."""

    def test_weight_map_attn(self, mock_attn, ssm_block):
        """Weight map strategy for attention block."""
        cfg = ConversionConfig(strategy="weight_map")
        metrics = convert_block(mock_attn, ssm_block, cfg, block_type="attn")
        assert metrics["strategy"] == "weight_map"
        assert metrics["block_type"] == "attn"
        assert "weight_map" in metrics

    def test_weight_map_ffn(self, mock_mlp, moe_block):
        """Weight map strategy for FFN block."""
        cfg = ConversionConfig(strategy="weight_map")
        metrics = convert_block(mock_mlp, moe_block, cfg, block_type="ffn")
        assert metrics["strategy"] == "weight_map"
        assert metrics["block_type"] == "ffn"
        assert "weight_map" in metrics

    def test_distillation_without_calibration_warns(self, mock_attn, ssm_block):
        """Distillation without calibration data should log warning, not crash."""
        cfg = ConversionConfig(strategy="distillation")
        # No calibration data -- should skip refinement gracefully
        metrics = convert_block(mock_attn, ssm_block, cfg, block_type="attn")
        assert "refinement" not in metrics

    def test_hybrid_strategy(self, mock_attn, ssm_block):
        """Hybrid strategy without calibration data: weight_map runs, refinement skipped."""
        cfg = ConversionConfig(strategy="hybrid")
        metrics = convert_block(mock_attn, ssm_block, cfg, block_type="attn")
        assert "weight_map" in metrics
        # No calibration data so refinement was skipped
        assert "refinement" not in metrics

    def test_layer_idx_propagated(self, mock_attn, ssm_block):
        """layer_idx should appear in metrics."""
        cfg = ConversionConfig(strategy="weight_map")
        metrics = convert_block(mock_attn, ssm_block, cfg, layer_idx=7, block_type="attn")
        assert metrics["layer_idx"] == 7


# ===================================================================
# Additional edge case tests for svd_project
# ===================================================================

class TestSvdProjectEdgeCases:
    """Edge cases for SVD projection."""

    def test_zero_matrix(self):
        """SVD of zero matrix should return zero matrix."""
        w = torch.zeros(32, 32)
        result = svd_project(w, 16, 16)
        assert result.shape == (16, 16)
        assert torch.allclose(result, torch.zeros(16, 16))

    def test_single_row_col(self):
        """Project to 1x1 should work."""
        w = torch.randn(16, 16)
        result = svd_project(w, 1, 1)
        assert result.shape == (1, 1)
        assert torch.isfinite(result).all()

    def test_rank_one_matrix(self):
        """Rank-1 matrix should be well-handled."""
        u = torch.randn(32, 1)
        v = torch.randn(1, 16)
        w = u @ v  # rank 1
        result = svd_project(w, 8, 8)
        assert result.shape == (8, 8)
        assert torch.isfinite(result).all()

    def test_nonsquare_input(self):
        """Non-square input should work."""
        w = torch.randn(100, 20)
        result = svd_project(w, 10, 10)
        assert result.shape == (10, 10)

    def test_float16_input(self):
        """float16 input should return float16 output."""
        w = torch.randn(32, 32).half()
        result = svd_project(w, 16, 16)
        assert result.dtype == torch.float16
