"""Smoke tests for aya-distill package structure."""
from __future__ import annotations


def test_version():
    from aya_distill import __version__
    assert __version__ == "0.1.0"


def test_languages():
    from aya_distill.languages import LANGUAGES, LANGUAGE_FAMILIES, LANGUAGE_CODES
    assert len(LANGUAGES) == 10
    assert len(LANGUAGE_FAMILIES) == 8
    assert "en" in LANGUAGE_CODES
    assert LANGUAGES["te"].script == "Telugu"


def test_cka_import():
    from aya_distill.distill.cka import linear_cka, rbf_cka
    assert callable(linear_cka)
    assert callable(rbf_cka)


def test_eval_metrics_import():
    from aya_distill.eval.metrics import (
        bootstrap_confidence_interval,
        degradation_equity_score,
    )
    assert callable(bootstrap_confidence_interval)
    assert callable(degradation_equity_score)


def test_cka_linear():
    import torch
    from aya_distill.distill.cka import linear_cka

    X = torch.randn(50, 32)
    score = linear_cka(X, X)
    assert abs(score - 1.0) < 1e-5, "CKA of identical matrices should be 1.0"


def test_multilingual_suite():
    from aya_distill.testing import MultilingualTestSuite

    def echo(prompt: str) -> str:
        return f"Response: {prompt[:30]}"

    suite = MultilingualTestSuite(model_fn=echo, languages=["en", "es"])
    result = suite.run()
    assert "en" in result.language_results
    assert "es" in result.language_results
    assert result.mean_coherence > 0


def test_tool_calling():
    from aya_distill.testing import ToolCallingTest

    def json_model(prompt: str) -> str:
        return '{"function": "get_weather", "arguments": {"location": "Tokyo"}}'

    test = ToolCallingTest(model_fn=json_model)
    result = test.test_single(
        prompt="What is the weather?",
        expected_function="get_weather",
        expected_args={"location": "Tokyo"},
    )
    assert result.json_valid
    assert result.function_correct
    assert result.args_correct


def test_fixtures():
    from aya_distill.testing.fixtures import (
        MULTILINGUAL_PROMPTS,
        TOOL_SCHEMAS,
        TOOL_SCENARIOS,
    )
    assert len(MULTILINGUAL_PROMPTS["general_knowledge"]) == 10
    assert len(TOOL_SCHEMAS) == 3
    assert len(TOOL_SCENARIOS) == 10
