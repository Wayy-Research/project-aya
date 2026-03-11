"""Tests for aya-distill evaluation and benchmark modules.

Covers: benchmarks config, mGSM prompt building & answer extraction,
XCOPA prompt building & choice extraction, throughput config & measurement,
and metrics edge cases (empty inputs, single elements, extreme values,
degradation equity score).

No GPU or real model/dataset downloads required -- uses mocks throughout.

Wayy Research -- Project Aya
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from aya_distill.eval.metrics import (
    ConfidenceInterval,
    FamilyResult,
    LanguageResult,
    PairedBootstrapResult,
    accuracy_from_predictions,
    aggregate_by_family,
    bootstrap_confidence_interval,
    degradation_equity_score,
    paired_bootstrap_test,
)
from aya_distill.eval.prompts.mgsm import (
    MGSM_EXEMPLARS,
    MGSM_INSTRUCTIONS,
    build_mgsm_prompt,
    extract_mgsm_answer,
)
from aya_distill.eval.prompts.xcopa import (
    XCOPA_CAUSE_INSTRUCTIONS,
    XCOPA_EFFECT_INSTRUCTIONS,
    XCOPA_EXEMPLARS,
    build_xcopa_prompt,
    extract_xcopa_answer,
)
from aya_distill.eval.benchmarks import (
    BenchmarkConfig,
    SUPPORTED_LANGUAGES,
    save_results,
)
from aya_distill.eval.throughput import (
    LanguageThroughput,
    ThroughputConfig,
    THROUGHPUT_PROMPTS,
)


# ===========================================================================
# 1. BenchmarkConfig
# ===========================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self):
        cfg = BenchmarkConfig(model_name="test-model")
        assert cfg.model_name == "test-model"
        assert cfg.dtype == "float16"
        assert cfg.device_map == "auto"
        assert cfg.n_shot == 8
        assert cfg.max_new_tokens == 256
        assert cfg.n_bootstrap == 1000
        assert cfg.seed == 42
        assert cfg.output_dir == "results/baseline"

    def test_default_languages(self):
        cfg = BenchmarkConfig(model_name="test-model")
        assert cfg.languages == list(SUPPORTED_LANGUAGES)
        assert "en" in cfg.languages
        assert "zh" in cfg.languages

    def test_custom_languages(self):
        cfg = BenchmarkConfig(model_name="m", languages=["en", "es"])
        assert cfg.languages == ["en", "es"]

    def test_frozen_config(self):
        cfg = BenchmarkConfig(model_name="m")
        with pytest.raises(AttributeError):
            cfg.model_name = "other"  # type: ignore[misc]

    def test_custom_n_shot(self):
        cfg = BenchmarkConfig(model_name="m", n_shot=4)
        assert cfg.n_shot == 4


# ===========================================================================
# 2. mGSM Prompt Building
# ===========================================================================


class TestMGSMPromptBuilding:
    """Tests for mGSM prompt construction."""

    def test_prompt_contains_instruction(self):
        prompt = build_mgsm_prompt("What is 2+2?", language="en")
        assert "####" in prompt
        assert MGSM_INSTRUCTIONS["en"] in prompt

    def test_prompt_contains_question(self):
        prompt = build_mgsm_prompt("What is 2+2?", language="en")
        assert "Q: What is 2+2?" in prompt
        assert prompt.endswith("A:")

    def test_prompt_includes_exemplars(self):
        prompt = build_mgsm_prompt("Test question", language="en", n_shot=2)
        # Should have 2 exemplars from the static set
        assert prompt.count("Q:") == 3  # 2 exemplars + 1 test question
        assert prompt.count("A:") == 3  # 2 exemplar answers + 1 trailing A:

    def test_prompt_n_shot_zero(self):
        prompt = build_mgsm_prompt("What is 2+2?", language="en", n_shot=0)
        assert prompt.count("Q:") == 1
        # Only the trailing "A:" for the test question
        assert prompt.count("A:") == 1

    def test_prompt_custom_exemplars(self):
        exemplars = [
            {"question": "Custom Q1", "answer": "Custom A1 #### 10"},
            {"question": "Custom Q2", "answer": "Custom A2 #### 20"},
        ]
        prompt = build_mgsm_prompt(
            "Test?", language="en", exemplars=exemplars, n_shot=2
        )
        assert "Custom Q1" in prompt
        assert "Custom Q2" in prompt

    def test_prompt_non_english_instruction(self):
        prompt = build_mgsm_prompt("Pregunta?", language="es")
        assert MGSM_INSTRUCTIONS["es"] in prompt

    def test_prompt_unknown_language_falls_back_to_english(self):
        prompt = build_mgsm_prompt("Question?", language="xx")
        assert MGSM_INSTRUCTIONS["en"] in prompt


# ===========================================================================
# 3. mGSM Answer Extraction
# ===========================================================================


class TestMGSMAnswerExtraction:
    """Tests for extracting numeric answers from mGSM model output."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("The total is 9 * 2 = 18.\n#### 18", "18"),
            ("#### 42", "42"),
            ("Step 1: 5+5=10\nStep 2: 10*3=30\n#### 30", "30"),
            ("Some reasoning.\n#### 1000", "1000"),
            ("#### 3.14", "3.14"),
            ("Blah blah\n#### $1,200", "1200"),
            ("#### 70000", "70000"),
            ("#### 70,000", "70000"),
        ],
        ids=[
            "standard-with-reasoning",
            "bare-delimiter",
            "multi-step",
            "large-number",
            "decimal",
            "dollar-comma",
            "plain-large",
            "comma-separated",
        ],
    )
    def test_extraction_with_delimiter(self, text: str, expected: str):
        assert extract_mgsm_answer(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "The answer is 42",
            "42",
            "Answer: 42.0",
            "No delimiter here",
            "",
        ],
        ids=[
            "no-delimiter-sentence",
            "bare-number",
            "answer-colon",
            "no-number-at-all",
            "empty-string",
        ],
    )
    def test_extraction_without_delimiter_returns_empty(self, text: str):
        result = extract_mgsm_answer(text)
        assert result == ""

    def test_extraction_multiple_delimiters_takes_last(self):
        text = "#### 10\nMore work...\n#### 20"
        assert extract_mgsm_answer(text) == "20"

    def test_extraction_delimiter_with_whitespace(self):
        text = "####   55  "
        assert extract_mgsm_answer(text) == "55"


# ===========================================================================
# 4. XCOPA Prompt Building
# ===========================================================================


class TestXCOPAPromptBuilding:
    """Tests for XCOPA prompt construction."""

    def test_cause_prompt_contains_instruction(self):
        prompt = build_xcopa_prompt(
            premise="The man turned on the faucet.",
            choice1="The toilet flushed.",
            choice2="Water flowed.",
            question_type="cause",
            language="en",
        )
        assert XCOPA_CAUSE_INSTRUCTIONS["en"] in prompt

    def test_effect_prompt_contains_instruction(self):
        prompt = build_xcopa_prompt(
            premise="The man turned on the faucet.",
            choice1="The toilet flushed.",
            choice2="Water flowed.",
            question_type="effect",
            language="en",
        )
        assert XCOPA_EFFECT_INSTRUCTIONS["en"] in prompt

    def test_prompt_contains_test_question(self):
        prompt = build_xcopa_prompt(
            premise="It rained.",
            choice1="The ground got wet.",
            choice2="The sun came out.",
            question_type="effect",
            language="en",
        )
        assert "Premise: It rained." in prompt
        assert "1: The ground got wet." in prompt
        assert "2: The sun came out." in prompt
        assert prompt.endswith("Answer:")

    def test_prompt_n_shot_controls_exemplar_count(self):
        prompt = build_xcopa_prompt(
            premise="Test premise.",
            choice1="A",
            choice2="B",
            question_type="effect",
            language="en",
            n_shot=2,
        )
        # Exemplars have "Answer:" lines, plus the test question's "Answer:"
        assert prompt.count("Answer:") >= 2

    def test_prompt_non_english_language(self):
        prompt = build_xcopa_prompt(
            premise="Premise.",
            choice1="Opcion 1.",
            choice2="Opcion 2.",
            question_type="cause",
            language="es",
        )
        assert XCOPA_CAUSE_INSTRUCTIONS["es"] in prompt

    def test_prompt_unknown_language_falls_back_to_english(self):
        prompt = build_xcopa_prompt(
            premise="P",
            choice1="C1",
            choice2="C2",
            question_type="effect",
            language="xx",
        )
        assert XCOPA_EFFECT_INSTRUCTIONS["en"] in prompt

    def test_prompt_custom_exemplars(self):
        exemplars = [
            {
                "premise": "Custom premise",
                "choice1": "Custom C1",
                "choice2": "Custom C2",
                "question": "cause",
                "label": "1",
            }
        ]
        prompt = build_xcopa_prompt(
            premise="Test",
            choice1="A",
            choice2="B",
            question_type="cause",
            language="en",
            exemplars=exemplars,
            n_shot=1,
        )
        assert "Custom premise" in prompt


# ===========================================================================
# 5. XCOPA Answer Extraction
# ===========================================================================


class TestXCOPAAnswerExtraction:
    """Tests for extracting choice (1 or 2) from XCOPA model output."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("1", "1"),
            ("2", "2"),
            ("The answer is 1.", "1"),
            ("The answer is 2.", "2"),
            ("Option 1 is correct", "1"),
            ("  2  ", "2"),
            ("I think it's 1 because...", "1"),
            ("Choice: 2\nExplanation...", "2"),
        ],
        ids=[
            "bare-1",
            "bare-2",
            "sentence-1",
            "sentence-2",
            "option-1",
            "padded-2",
            "reasoning-1",
            "multiline-2",
        ],
    )
    def test_extraction_valid(self, text: str, expected: str):
        assert extract_xcopa_answer(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "No numbers here",
            "Three and four",
            "   ",
        ],
        ids=["empty", "no-digits", "words-only", "whitespace"],
    )
    def test_extraction_returns_empty(self, text: str):
        assert extract_xcopa_answer(text) == ""

    def test_extraction_takes_first_valid_digit(self):
        """If both 1 and 2 appear, takes the first one."""
        assert extract_xcopa_answer("Option 1 is better than 2") == "1"
        assert extract_xcopa_answer("Between 2 and 1, I pick 2") == "2"

    def test_extraction_ignores_other_digits(self):
        """Digits 3-9 and 0 are ignored until a 1 or 2 appears."""
        assert extract_xcopa_answer("Option 5 or maybe 1") == "1"
        assert extract_xcopa_answer("90 percent sure it's 2") == "2"


# ===========================================================================
# 6. ThroughputConfig
# ===========================================================================


class TestThroughputConfig:
    """Tests for ThroughputConfig dataclass."""

    def test_default_values(self):
        cfg = ThroughputConfig()
        assert cfg.n_tokens == 512
        assert cfg.n_runs == 10
        assert cfg.n_warmup == 2
        assert cfg.seed == 42

    def test_default_languages_match_prompts(self):
        cfg = ThroughputConfig()
        assert set(cfg.languages) == set(THROUGHPUT_PROMPTS.keys())

    def test_custom_values(self):
        cfg = ThroughputConfig(n_tokens=128, n_runs=5, n_warmup=1)
        assert cfg.n_tokens == 128
        assert cfg.n_runs == 5
        assert cfg.n_warmup == 1

    def test_frozen_config(self):
        cfg = ThroughputConfig()
        with pytest.raises(AttributeError):
            cfg.n_tokens = 256  # type: ignore[misc]


# ===========================================================================
# 7. LanguageThroughput
# ===========================================================================


class TestLanguageThroughput:
    """Tests for LanguageThroughput data container."""

    def test_mean_and_std_properties(self):
        lt = LanguageThroughput(
            language="en",
            prompt_tokens=10,
            generated_tokens_per_run=[100, 100, 100],
            tokens_per_second=[50.0, 60.0, 70.0],
            time_to_first_token_ms=[10.0, 12.0, 14.0],
            total_time_sec=[2.0, 1.67, 1.43],
            peak_memory_mb=512.0,
        )
        assert lt.mean_tps == pytest.approx(60.0)
        assert lt.std_tps == pytest.approx(np.std([50.0, 60.0, 70.0]))
        assert lt.mean_ttft_ms == pytest.approx(12.0)
        assert lt.std_ttft_ms == pytest.approx(np.std([10.0, 12.0, 14.0]))

    def test_to_dict_structure(self):
        lt = LanguageThroughput(
            language="es",
            prompt_tokens=15,
            generated_tokens_per_run=[50],
            tokens_per_second=[25.0],
            time_to_first_token_ms=[8.0],
            total_time_sec=[2.0],
            peak_memory_mb=256.0,
        )
        d = lt.to_dict()
        assert d["language"] == "es"
        assert d["prompt_tokens"] == 15
        assert "mean_tokens_per_second" in d
        assert "std_tokens_per_second" in d
        assert "mean_ttft_ms" in d
        assert "std_ttft_ms" in d
        assert "peak_memory_mb" in d
        assert d["n_runs"] == 1
        assert d["raw_tps"] == [25.0]

    def test_single_run(self):
        lt = LanguageThroughput(
            language="zh",
            prompt_tokens=5,
            generated_tokens_per_run=[200],
            tokens_per_second=[100.0],
            time_to_first_token_ms=[5.0],
            total_time_sec=[2.0],
            peak_memory_mb=128.0,
        )
        assert lt.mean_tps == 100.0
        assert lt.std_tps == 0.0


# ===========================================================================
# 8. Metrics Edge Cases
# ===========================================================================


class TestBootstrapEdgeCases:
    """Edge cases for bootstrap_confidence_interval."""

    def test_empty_input(self):
        ci = bootstrap_confidence_interval(np.array([], dtype=np.int_))
        assert ci.mean == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0
        assert ci.std == 0.0

    def test_single_element_all_correct(self):
        ci = bootstrap_confidence_interval(np.array([1], dtype=np.int_))
        assert ci.mean == 1.0
        assert ci.lower == 1.0
        assert ci.upper == 1.0

    def test_single_element_all_wrong(self):
        ci = bootstrap_confidence_interval(np.array([0], dtype=np.int_))
        assert ci.mean == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_all_correct(self):
        correct = np.ones(100, dtype=np.int_)
        ci = bootstrap_confidence_interval(correct, n_bootstrap=500, seed=42)
        assert ci.mean == 1.0
        assert ci.lower == 1.0
        assert ci.upper == 1.0
        assert ci.std == 0.0

    def test_all_wrong(self):
        correct = np.zeros(100, dtype=np.int_)
        ci = bootstrap_confidence_interval(correct, n_bootstrap=500, seed=42)
        assert ci.mean == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_ci_bounds_are_ordered(self):
        rng = np.random.RandomState(123)
        correct = rng.randint(0, 2, size=200).astype(np.int_)
        ci = bootstrap_confidence_interval(correct, n_bootstrap=1000, seed=42)
        assert ci.lower <= ci.mean <= ci.upper

    def test_reproducibility_with_same_seed(self):
        data = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int_)
        ci1 = bootstrap_confidence_interval(data, seed=99)
        ci2 = bootstrap_confidence_interval(data, seed=99)
        assert ci1.mean == ci2.mean
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_different_seeds_may_differ(self):
        data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 10, dtype=np.int_)
        ci1 = bootstrap_confidence_interval(data, seed=1)
        ci2 = bootstrap_confidence_interval(data, seed=2)
        # Mean is deterministic regardless of seed
        assert ci1.mean == ci2.mean
        # But CI bounds will generally differ with different seeds


class TestAccuracyFromPredictions:
    """Tests for accuracy_from_predictions utility."""

    def test_all_correct(self):
        preds = ["a", "b", "c"]
        refs = ["a", "b", "c"]
        result = accuracy_from_predictions(preds, refs)
        assert np.array_equal(result, np.array([1, 1, 1]))

    def test_all_wrong(self):
        preds = ["x", "y", "z"]
        refs = ["a", "b", "c"]
        result = accuracy_from_predictions(preds, refs)
        assert np.array_equal(result, np.array([0, 0, 0]))

    def test_mixed(self):
        preds = ["a", "x", "c"]
        refs = ["a", "b", "c"]
        result = accuracy_from_predictions(preds, refs)
        assert np.array_equal(result, np.array([1, 0, 1]))

    def test_whitespace_stripping(self):
        preds = [" a ", "b "]
        refs = ["a", " b"]
        result = accuracy_from_predictions(preds, refs)
        assert np.array_equal(result, np.array([1, 1]))


class TestPairedBootstrapEdgeCases:
    """Edge cases for paired_bootstrap_test."""

    def test_identical_arrays(self):
        a = np.array([1, 0, 1, 1, 0], dtype=np.int_)
        result = paired_bootstrap_test(a, a.copy(), n_bootstrap=500)
        assert result.mean_diff == 0.0
        # p-value should be high (not significant)
        assert result.p_value >= 0.05

    def test_perfect_vs_zero(self):
        a = np.ones(50, dtype=np.int_)
        b = np.zeros(50, dtype=np.int_)
        result = paired_bootstrap_test(a, b, n_bootstrap=500)
        assert result.mean_diff == 1.0
        assert result.significant is True

    def test_length_mismatch_raises(self):
        a = np.array([1, 0], dtype=np.int_)
        b = np.array([1, 0, 1], dtype=np.int_)
        with pytest.raises(AssertionError, match="same length"):
            paired_bootstrap_test(a, b)


class TestAggregateByFamily:
    """Tests for language family aggregation."""

    def test_empty_input(self):
        result = aggregate_by_family({})
        assert result == []

    def test_single_language(self):
        result = aggregate_by_family({"en": 0.8})
        # "en" is Indo-European
        assert len(result) >= 1
        ie_families = [r for r in result if r.family == "Indo-European"]
        assert len(ie_families) == 1
        assert ie_families[0].mean_accuracy == pytest.approx(0.8)

    def test_multiple_families(self):
        scores = {"en": 0.9, "zh": 0.7, "ja": 0.6, "tr": 0.5}
        result = aggregate_by_family(scores)
        families = {r.family for r in result}
        # Should span multiple families
        assert len(families) >= 3


class TestDegradationEquityScore:
    """Tests for degradation_equity_score."""

    def test_identical_scores_zero_des(self):
        scores = {"en": 0.8, "zh": 0.7, "ja": 0.6}
        des = degradation_equity_score(scores, scores)
        assert des == pytest.approx(0.0)

    def test_uniform_drop_low_des(self):
        teacher = {"en": 0.9, "zh": 0.8, "tr": 0.7}
        student = {"en": 0.8, "zh": 0.7, "tr": 0.6}
        des = degradation_equity_score(teacher, student)
        # Uniform 0.1 drop across families => low variance
        # Not exactly 0 because languages are in different families
        assert des >= 0.0

    def test_unequal_drop_higher_des(self):
        teacher = {"en": 0.9, "zh": 0.9, "tr": 0.9}
        # English barely drops, others drop a lot
        student = {"en": 0.85, "zh": 0.3, "tr": 0.3}
        des = degradation_equity_score(teacher, student)
        assert des > 0.0

    def test_no_common_languages(self):
        teacher = {"en": 0.9}
        student = {"xx": 0.5}
        des = degradation_equity_score(teacher, student)
        # No common keys => no drops => 0.0
        assert des == 0.0

    def test_single_family_returns_zero(self):
        # All in Indo-European, so only 1 family => variance of 1 element = 0
        teacher = {"en": 0.9, "fr": 0.8, "de": 0.85}
        student = {"en": 0.7, "fr": 0.6, "de": 0.65}
        des = degradation_equity_score(teacher, student)
        # Only 1 family => len(family_drops) < 2 => returns 0.0
        assert des == 0.0


# ===========================================================================
# 9. save_results utility
# ===========================================================================


class TestSaveResults:
    """Tests for save_results JSON output."""

    def test_save_creates_file(self, tmp_path):
        results = {"benchmark": "test", "accuracy": 0.9}
        path = save_results(results, str(tmp_path), "mgsm")
        assert path.exists()
        assert path.suffix == ".json"
        assert "mgsm" in path.name

    def test_save_json_content(self, tmp_path):
        import json

        results = {"benchmark": "mgsm", "model": "test", "score": 0.85}
        path = save_results(results, str(tmp_path), "mgsm")
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["benchmark"] == "mgsm"
        assert loaded["score"] == 0.85

    def test_save_creates_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        results = {"x": 1}
        path = save_results(results, str(nested), "test")
        assert path.exists()


# ===========================================================================
# 10. ConfidenceInterval and LanguageResult to_dict
# ===========================================================================


class TestDataclassSerialization:
    """Tests for to_dict methods on result dataclasses."""

    def test_confidence_interval_to_dict(self):
        ci = ConfidenceInterval(
            mean=0.75, lower=0.70, upper=0.80,
            std=0.03, n_bootstrap=1000, confidence_level=0.95,
        )
        d = ci.to_dict()
        assert d["mean"] == 0.75
        assert d["lower"] == 0.70
        assert d["upper"] == 0.80
        assert d["n_bootstrap"] == 1000

    def test_language_result_to_dict(self):
        ci = ConfidenceInterval(
            mean=0.85, lower=0.80, upper=0.90,
            std=0.02, n_bootstrap=500, confidence_level=0.95,
        )
        lr = LanguageResult(language="en", accuracy=0.85, ci=ci, n_samples=100)
        d = lr.to_dict()
        assert d["language"] == "en"
        assert d["accuracy"] == 0.85
        assert d["n_samples"] == 100
        assert "ci" in d

    def test_family_result_to_dict(self):
        fr = FamilyResult(
            family="Indo-European",
            languages=["en", "es"],
            mean_accuracy=0.80,
            accuracies={"en": 0.85, "es": 0.75},
        )
        d = fr.to_dict()
        assert d["family"] == "Indo-European"
        assert len(d["languages"]) == 2
        assert d["mean_accuracy"] == 0.80

    def test_paired_bootstrap_result_to_dict(self):
        r = PairedBootstrapResult(
            p_value=0.03, mean_diff=0.1,
            ci_lower=0.02, ci_upper=0.18,
            significant=True, n_bootstrap=1000,
        )
        d = r.to_dict()
        assert d["significant"] is True
        assert d["p_value"] == 0.03
