"""Pytest fixtures for aya-distill multilingual testing."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn as nn

# Path setup for local packages
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "aetheris"))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    """Ensure reproducible tests across all test files."""
    torch.manual_seed(42)
    yield


# ---------------------------------------------------------------------------
# Mock modules for converter/hooks tests
# ---------------------------------------------------------------------------

class MockAttentionLayer(nn.Module):
    """Minimal attention layer mock for converter tests."""

    def __init__(self, hidden_size: int = 64, kv_dim: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MockMLPLayer(nn.Module):
    """Minimal MLP/FFN mock for converter tests."""

    def __init__(self, hidden_size: int = 64, intermediate_size: int = 256):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.silu(self.gate_proj(x)) * self.up_proj(x))


class MockTransformerLayer(nn.Module):
    """Minimal transformer layer with attn + mlp."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.self_attn = MockAttentionLayer(hidden_size)
        self.mlp = MockMLPLayer(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class MockTransformerModel(nn.Module):
    """Minimal HuggingFace-style model with model.layers."""

    def __init__(self, n_layers: int = 4, hidden_size: int = 64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockTransformerLayer(hidden_size) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


@pytest.fixture
def mock_teacher() -> MockTransformerModel:
    """4-layer mock transformer teacher."""
    return MockTransformerModel(n_layers=4, hidden_size=64)


@pytest.fixture
def dummy_model() -> Callable[[str], str]:
    """A dummy model that echoes back the prompt (for testing the test harness)."""

    def model_fn(prompt: str) -> str:
        return f"This is a response to: {prompt[:50]}..."

    return model_fn


@pytest.fixture
def multilingual_suite(
    dummy_model: Callable[[str], str],
):
    """Pre-configured multilingual test suite with dummy model."""
    from aya_distill.testing.multilingual import MultilingualTestSuite

    return MultilingualTestSuite(
        model_fn=dummy_model,
        languages=["en", "es", "hi", "zh"],
    )


@pytest.fixture
def tool_test(dummy_model: Callable[[str], str]):
    """Pre-configured tool calling test with dummy model."""
    from aya_distill.testing.tool_testing import ToolCallingTest
    from aya_distill.testing.fixtures import TOOL_SCHEMAS

    return ToolCallingTest(
        model_fn=dummy_model,
        tool_schemas=TOOL_SCHEMAS,
    )


@pytest.fixture
def json_model() -> Callable[[str], str]:
    """A dummy model that always returns a valid tool-call JSON."""

    def model_fn(prompt: str) -> str:
        # Extract a city name heuristic: last quoted string in the prompt
        import re

        cities = re.findall(r'"([^"]+)"', prompt)
        # Pick the last quoted string that looks like a city
        city = "Unknown"
        for candidate in reversed(cities):
            if "?" not in candidate and len(candidate) < 30:
                city = candidate
                break
        return (
            '{"function": "get_weather", '
            f'"arguments": {{"location": "{city}"}}}}'
        )

    return model_fn


@pytest.fixture
def tool_test_json(json_model: Callable[[str], str]):
    """Tool calling test using the JSON-producing dummy model."""
    from aya_distill.testing.tool_testing import ToolCallingTest
    from aya_distill.testing.fixtures import TOOL_SCHEMAS

    return ToolCallingTest(
        model_fn=json_model,
        tool_schemas=TOOL_SCHEMAS,
    )
