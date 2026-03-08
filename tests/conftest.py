"""Pytest fixtures for aya-distill multilingual testing."""
from __future__ import annotations

from typing import Callable

import pytest


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
