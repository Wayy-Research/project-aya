"""Comprehensive test suite for aya-distill package.

Covers: language registry, CKA module, eval metrics, checklist validation,
multilingual testing, tool calling tests, fixtures, and integration.

No GPU or model downloads required -- uses mocks and synthetic data throughout.

Wayy Research -- Project Aya
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# 0. Smoke / version
# ---------------------------------------------------------------------------


def test_version():
    from aya_distill import __version__

    assert __version__ == "0.1.0"


# ===========================================================================
# 1. Language Registry
# ===========================================================================


class TestLanguageRegistry:
    """Tests for aya_distill.languages -- single source of truth."""

    def test_total_language_count(self):
        from aya_distill.languages import LANGUAGES

        assert len(LANGUAGES) == 67, (
            f"Expected 67 languages, got {len(LANGUAGES)}"
        )

    def test_language_codes_sorted(self):
        from aya_distill.languages import LANGUAGE_CODES, LANGUAGES

        assert LANGUAGE_CODES == sorted(LANGUAGES.keys())

    @pytest.mark.parametrize(
        "code",
        [
            "en", "es", "fr", "de", "zh", "ja", "ko", "ar", "hi", "sw",
            "te", "tr", "id", "ru", "pt", "pl", "uk", "bg", "sr", "hr",
            "cs", "sk", "sl", "ro", "ca", "gl", "nl", "it", "da", "sv",
            "no", "el", "et", "fi", "hu", "eu", "cy", "ga", "mt", "fa",
            "ur", "he", "mr", "bn", "gu", "pa", "ta", "ne", "tl", "ms",
            "vi", "jv", "km", "th", "lo", "my", "am", "ha", "ig", "mg",
            "sn", "wo", "xh", "yo", "zu", "lv", "lt",
        ],
    )
    def test_expected_language_code_exists(self, code: str):
        from aya_distill.languages import LANGUAGES

        assert code in LANGUAGES, f"Missing language code: {code}"

    def test_language_families_non_empty(self):
        from aya_distill.languages import LANGUAGE_FAMILIES

        assert len(LANGUAGE_FAMILIES) >= 8, (
            f"Expected at least 8 families, got {len(LANGUAGE_FAMILIES)}"
        )
        for family, codes in LANGUAGE_FAMILIES.items():
            assert len(codes) > 0, f"Family '{family}' has no languages"

    def test_language_families_cover_all_codes(self):
        """Every language code should appear in exactly one family."""
        from aya_distill.languages import LANGUAGE_FAMILIES, LANGUAGES

        all_codes_from_families: list[str] = []
        for codes in LANGUAGE_FAMILIES.values():
            all_codes_from_families.extend(codes)
        assert set(all_codes_from_families) == set(LANGUAGES.keys())

    def test_script_groups_cover_all_codes(self):
        from aya_distill.languages import SCRIPT_GROUPS, LANGUAGES

        all_codes: list[str] = []
        for codes in SCRIPT_GROUPS.values():
            all_codes.extend(codes)
        assert set(all_codes) == set(LANGUAGES.keys())

    def test_region_groups_cover_all_codes(self):
        from aya_distill.languages import REGION_GROUPS, LANGUAGES

        all_codes: list[str] = []
        for codes in REGION_GROUPS.values():
            all_codes.extend(codes)
        assert set(all_codes) == set(LANGUAGES.keys())

    def test_resource_groups_are_high_medium_low(self):
        from aya_distill.languages import RESOURCE_GROUPS

        assert set(RESOURCE_GROUPS.keys()) == {"high", "medium", "low"}

    def test_resource_groups_cover_all_codes(self):
        from aya_distill.languages import RESOURCE_GROUPS, LANGUAGES

        all_codes: list[str] = []
        for codes in RESOURCE_GROUPS.values():
            all_codes.extend(codes)
        assert set(all_codes) == set(LANGUAGES.keys())

    def test_distillation_languages_subset(self):
        from aya_distill.languages import DISTILLATION_LANGUAGES, LANGUAGES

        assert len(DISTILLATION_LANGUAGES) == 10
        for code in DISTILLATION_LANGUAGES:
            assert code in LANGUAGES, (
                f"Distillation language '{code}' not in LANGUAGES"
            )

    def test_latin_script_languages(self):
        from aya_distill.languages import LATIN_SCRIPT_LANGUAGES, LANGUAGES

        assert isinstance(LATIN_SCRIPT_LANGUAGES, frozenset)
        for code in LATIN_SCRIPT_LANGUAGES:
            assert LANGUAGES[code].script == "Latin"
        # Spot-check: English uses Latin, Chinese does not
        assert "en" in LATIN_SCRIPT_LANGUAGES
        assert "zh" not in LATIN_SCRIPT_LANGUAGES

    def test_language_dataclass_is_frozen(self):
        from aya_distill.languages import LANGUAGES

        lang = LANGUAGES["en"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            lang.code = "xx"  # type: ignore[misc]

    def test_language_names_dict(self):
        from aya_distill.languages import LANGUAGE_NAMES, LANGUAGES

        assert len(LANGUAGE_NAMES) == len(LANGUAGES)
        assert LANGUAGE_NAMES["en"] == "English"
        assert LANGUAGE_NAMES["ja"] == "Japanese"

    def test_telugu_script(self):
        from aya_distill.languages import LANGUAGES

        assert LANGUAGES["te"].script == "Telugu"
        assert LANGUAGES["te"].family == "Dravidian"


# ===========================================================================
# 2. CKA Module
# ===========================================================================


class TestLinearCKA:
    """Tests for linear CKA similarity."""

    def test_identical_matrices_score_one(self):
        from aya_distill.distill.cka import linear_cka

        X = torch.randn(50, 32)
        score = linear_cka(X, X).item()
        assert abs(score - 1.0) < 1e-5, (
            f"CKA(X, X) should be 1.0, got {score}"
        )

    def test_random_uncorrelated_near_zero(self):
        """Two large random matrices should have low CKA."""
        from aya_distill.distill.cka import linear_cka

        torch.manual_seed(123)
        X = torch.randn(500, 64)
        Y = torch.randn(500, 64)
        score = linear_cka(X, Y).item()
        assert score < 0.15, f"Expected low CKA for random data, got {score}"

    def test_scale_invariance(self):
        """CKA should be invariant to isotropic scaling."""
        from aya_distill.distill.cka import linear_cka

        torch.manual_seed(42)
        X = torch.randn(100, 32)
        Y = torch.randn(100, 32)
        score_orig = linear_cka(X, Y).item()
        score_scaled = linear_cka(X * 100.0, Y * 0.01).item()
        assert abs(score_orig - score_scaled) < 1e-4, (
            f"CKA should be scale-invariant: {score_orig} vs {score_scaled}"
        )

    def test_different_dimensions(self):
        """CKA works with different feature dimensions."""
        from aya_distill.distill.cka import linear_cka

        torch.manual_seed(7)
        X = torch.randn(80, 32)
        Y = torch.randn(80, 64)
        score = linear_cka(X, Y).item()
        assert 0.0 <= score <= 1.0

    def test_sample_mismatch_raises(self):
        from aya_distill.distill.cka import linear_cka

        X = torch.randn(50, 32)
        Y = torch.randn(30, 32)
        with pytest.raises(AssertionError):
            linear_cka(X, Y)

    def test_output_is_scalar_tensor(self):
        from aya_distill.distill.cka import linear_cka

        X = torch.randn(20, 8)
        result = linear_cka(X, X)
        assert result.dim() == 0


class TestRBFCKA:
    """Tests for RBF kernel CKA."""

    def test_identical_matrices_score_one(self):
        from aya_distill.distill.cka import rbf_cka

        X = torch.randn(50, 16)
        score = rbf_cka(X, X).item()
        assert abs(score - 1.0) < 1e-4, (
            f"RBF CKA(X, X) should be ~1.0, got {score}"
        )

    def test_rbf_bounded_zero_one(self):
        from aya_distill.distill.cka import rbf_cka

        torch.manual_seed(99)
        X = torch.randn(60, 16)
        Y = torch.randn(60, 16)
        score = rbf_cka(X, Y).item()
        assert -0.05 <= score <= 1.05, (
            f"RBF CKA should be roughly in [0,1], got {score}"
        )


class TestMinibatchCKA:
    """Tests for MinibatchCKAAccumulator and minibatch_cka."""

    def test_matches_full_batch(self):
        """Minibatch CKA should closely match full-batch linear CKA."""
        from aya_distill.distill.cka import linear_cka, minibatch_cka

        torch.manual_seed(42)
        X = torch.randn(200, 32)
        Y = torch.randn(200, 64)

        full = linear_cka(X, Y).item()
        mini = minibatch_cka(X, Y, batch_size=50)
        assert abs(full - mini) < 1e-4, (
            f"Minibatch CKA ({mini}) should match full-batch ({full})"
        )

    def test_accumulator_no_samples_raises(self):
        from aya_distill.distill.cka import MinibatchCKAAccumulator

        acc = MinibatchCKAAccumulator(d_x=32, d_y=32)
        with pytest.raises(ValueError, match="No samples"):
            acc.compute()

    def test_accumulator_reset(self):
        from aya_distill.distill.cka import MinibatchCKAAccumulator

        acc = MinibatchCKAAccumulator(d_x=16, d_y=16)
        acc.update(torch.randn(10, 16), torch.randn(10, 16))
        assert acc._n == 10
        acc.reset()
        assert acc._n == 0


class TestCKAPermutationTest:
    """Tests for cka_permutation_test."""

    def test_correlated_pair_is_significant(self):
        """Identical representations should be significantly similar."""
        from aya_distill.distill.cka import cka_permutation_test

        torch.manual_seed(42)
        X = torch.randn(100, 32)
        # Y is a noisy copy of X -- should be highly correlated
        Y = X + torch.randn_like(X) * 0.1

        result = cka_permutation_test(X, Y, n_permutations=200, seed=42)
        assert result["p_value"] < 0.05, (
            f"Correlated pair should be significant, p={result['p_value']}"
        )
        assert result["observed_cka"] > 0.5

    def test_random_pair_not_significant(self):
        """Independent random matrices should not be significant."""
        from aya_distill.distill.cka import cka_permutation_test

        torch.manual_seed(0)
        X = torch.randn(100, 32)
        Y = torch.randn(100, 32)

        result = cka_permutation_test(X, Y, n_permutations=200, seed=0)
        # p-value should be relatively large (not significant)
        assert result["p_value"] > 0.01, (
            f"Random pair should not be highly significant, p={result['p_value']}"
        )

    def test_result_keys(self):
        from aya_distill.distill.cka import cka_permutation_test

        X = torch.randn(30, 8)
        Y = torch.randn(30, 8)
        result = cka_permutation_test(X, Y, n_permutations=10)
        assert set(result.keys()) == {
            "observed_cka", "p_value", "null_mean", "null_std",
        }


class TestComputeLayerwiseCKA:
    """Tests for compute_layerwise_cka."""

    def test_correct_shape(self):
        from aya_distill.distill.cka import compute_layerwise_cka

        teacher = {
            "layer.0": torch.randn(50, 32),
            "layer.1": torch.randn(50, 32),
            "layer.2": torch.randn(50, 32),
        }
        student = {
            "block.0": torch.randn(50, 16),
            "block.1": torch.randn(50, 16),
        }
        result = compute_layerwise_cka(teacher, student)
        assert result.scores.shape == (3, 2)
        assert len(result.teacher_layer_names) == 3
        assert len(result.student_layer_names) == 2

    def test_diagonal_identity(self):
        """When teacher == student, diagonal should be 1.0."""
        from aya_distill.distill.cka import compute_layerwise_cka

        torch.manual_seed(42)
        acts = {
            "layer.0": torch.randn(50, 32),
            "layer.1": torch.randn(50, 32),
        }
        result = compute_layerwise_cka(acts, acts)
        for i in range(2):
            assert abs(result.scores[i, i] - 1.0) < 1e-4

    def test_to_dict_serializable(self):
        from aya_distill.distill.cka import compute_layerwise_cka

        acts = {"l0": torch.randn(20, 8)}
        result = compute_layerwise_cka(acts, acts)
        d = result.to_dict()
        # Should be JSON-serializable
        json.dumps(d)


# ===========================================================================
# 3. Eval Metrics
# ===========================================================================


class TestBootstrapCI:
    """Tests for bootstrap_confidence_interval."""

    def test_known_data(self):
        from aya_distill.eval.metrics import bootstrap_confidence_interval

        # All correct
        correct = np.ones(100, dtype=np.int_)
        ci = bootstrap_confidence_interval(correct, seed=42)
        assert ci.mean == 1.0
        assert ci.lower == 1.0
        assert ci.upper == 1.0

    def test_half_correct(self):
        from aya_distill.eval.metrics import bootstrap_confidence_interval

        correct = np.array([1, 0] * 50, dtype=np.int_)
        ci = bootstrap_confidence_interval(correct, seed=42)
        assert ci.mean == pytest.approx(0.5)
        assert ci.lower < ci.mean
        assert ci.upper > ci.mean

    def test_empty_input(self):
        from aya_distill.eval.metrics import bootstrap_confidence_interval

        correct = np.array([], dtype=np.int_)
        ci = bootstrap_confidence_interval(correct)
        assert ci.mean == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_ci_bounds_ordered(self):
        from aya_distill.eval.metrics import bootstrap_confidence_interval

        correct = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=np.int_)
        ci = bootstrap_confidence_interval(correct, seed=0)
        assert ci.lower <= ci.mean <= ci.upper

    def test_confidence_level_stored(self):
        from aya_distill.eval.metrics import bootstrap_confidence_interval

        correct = np.ones(10, dtype=np.int_)
        ci = bootstrap_confidence_interval(correct, confidence_level=0.99)
        assert ci.confidence_level == 0.99

    def test_to_dict(self):
        from aya_distill.eval.metrics import bootstrap_confidence_interval

        ci = bootstrap_confidence_interval(np.ones(10, dtype=np.int_))
        d = ci.to_dict()
        assert "mean" in d
        assert "lower" in d
        assert "upper" in d


class TestDegradationEquityScore:
    """Tests for degradation_equity_score."""

    def test_equal_degradation_is_zero(self):
        """If all languages degrade equally, DES should be 0."""
        from aya_distill.eval.metrics import degradation_equity_score

        teacher = {"en": 0.9, "es": 0.85, "hi": 0.80, "zh": 0.88, "sw": 0.75}
        # All drop by exactly 0.1
        student = {k: v - 0.1 for k, v in teacher.items()}
        des = degradation_equity_score(teacher, student)
        assert des == pytest.approx(0.0, abs=1e-10)

    def test_unequal_degradation_positive(self):
        """If some families degrade more, DES should be > 0."""
        from aya_distill.eval.metrics import degradation_equity_score

        teacher = {"en": 0.9, "es": 0.9, "zh": 0.9, "ja": 0.9, "sw": 0.9}
        # Indo-European drops 0.1, Sino-Tibetan drops 0.3, Niger-Congo drops 0.5
        student = {"en": 0.8, "es": 0.8, "zh": 0.6, "ja": 0.6, "sw": 0.4}
        des = degradation_equity_score(teacher, student)
        assert des > 0.0

    def test_single_family_returns_zero(self):
        """With only one family, variance is 0."""
        from aya_distill.eval.metrics import degradation_equity_score

        teacher = {"en": 0.9, "es": 0.85}
        student = {"en": 0.8, "es": 0.75}
        des = degradation_equity_score(teacher, student)
        assert des == 0.0


class TestAggregateByFamily:
    """Tests for aggregate_by_family."""

    def test_correct_families(self):
        from aya_distill.eval.metrics import aggregate_by_family

        per_lang = {"en": 0.9, "es": 0.85, "zh": 0.88, "ja": 0.82, "sw": 0.70}
        results = aggregate_by_family(per_lang)
        families = {r.family for r in results}
        assert "Indo-European" in families
        assert "Sino-Tibetan" in families

    def test_empty_input(self):
        from aya_distill.eval.metrics import aggregate_by_family

        results = aggregate_by_family({})
        assert results == []

    def test_mean_accuracy_correct(self):
        from aya_distill.eval.metrics import aggregate_by_family

        per_lang = {"en": 0.8, "fr": 0.6}
        results = aggregate_by_family(per_lang)
        ie_result = [r for r in results if r.family == "Indo-European"][0]
        assert ie_result.mean_accuracy == pytest.approx(0.7)

    def test_to_dict(self):
        from aya_distill.eval.metrics import aggregate_by_family

        results = aggregate_by_family({"en": 0.9})
        assert len(results) == 1
        d = results[0].to_dict()
        assert "family" in d
        assert "mean_accuracy" in d


class TestPairedBootstrapTest:
    """Tests for paired_bootstrap_test."""

    def test_identical_systems_not_significant(self):
        from aya_distill.eval.metrics import paired_bootstrap_test

        a = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10, dtype=np.int_)
        result = paired_bootstrap_test(a, a, seed=42)
        # Identical systems: diff is 0, should not be significant
        assert result.mean_diff == pytest.approx(0.0)

    def test_clearly_different_systems(self):
        from aya_distill.eval.metrics import paired_bootstrap_test

        a = np.ones(100, dtype=np.int_)
        b = np.zeros(100, dtype=np.int_)
        result = paired_bootstrap_test(a, b, seed=42)
        assert result.mean_diff == pytest.approx(1.0)
        assert result.significant is True

    def test_length_mismatch_raises(self):
        from aya_distill.eval.metrics import paired_bootstrap_test

        a = np.ones(10, dtype=np.int_)
        b = np.ones(20, dtype=np.int_)
        with pytest.raises(AssertionError):
            paired_bootstrap_test(a, b)

    def test_to_dict(self):
        from aya_distill.eval.metrics import paired_bootstrap_test

        a = np.array([1, 0, 1, 0], dtype=np.int_)
        result = paired_bootstrap_test(a, a)
        d = result.to_dict()
        assert "p_value" in d
        assert "significant" in d


class TestAccuracyFromPredictions:
    """Tests for accuracy_from_predictions."""

    def test_exact_match(self):
        from aya_distill.eval.metrics import accuracy_from_predictions

        preds = ["cat", "dog", "fish"]
        refs = ["cat", "dog", "fish"]
        correct = accuracy_from_predictions(preds, refs)
        assert correct.sum() == 3

    def test_no_match(self):
        from aya_distill.eval.metrics import accuracy_from_predictions

        preds = ["cat", "dog"]
        refs = ["bird", "fish"]
        correct = accuracy_from_predictions(preds, refs)
        assert correct.sum() == 0

    def test_strips_whitespace(self):
        from aya_distill.eval.metrics import accuracy_from_predictions

        preds = ["  cat  "]
        refs = ["cat"]
        correct = accuracy_from_predictions(preds, refs)
        assert correct[0] == 1


# ===========================================================================
# 4. Checklist
# ===========================================================================


class TestChecklistItems:
    """Tests for CHECKLIST_ITEMS structure."""

    def test_total_count_is_30(self):
        from aya_distill.eval.checklist import CHECKLIST_ITEMS

        assert len(CHECKLIST_ITEMS) == 30

    def test_eight_categories(self):
        from aya_distill.eval.checklist import CATEGORY_NAMES

        assert len(CATEGORY_NAMES) == 8
        expected = {
            "Evaluation Prompts",
            "Choice of Metrics",
            "Statistical Testing",
            "Aggregating Results",
            "Qualitative Insights",
            "Reproducibility",
            "Meta-Evaluation",
            "Reasoning Evaluation",
        }
        assert set(CATEGORY_NAMES) == expected

    def test_all_items_have_checker(self):
        from aya_distill.eval.checklist import CHECKLIST_INDEX

        assert len(CHECKLIST_INDEX) == 30

    def test_severity_values(self):
        from aya_distill.eval.checklist import CHECKLIST_ITEMS

        for item in CHECKLIST_ITEMS:
            assert item.severity in ("required", "recommended")


class TestValidateResults:
    """Tests for validate_results."""

    def _compliant_results(self) -> dict[str, Any]:
        """Build a fully compliant results dict (all 30 items)."""
        per_language: dict[str, Any] = {}
        for code in ("en", "es", "hi", "zh"):
            per_language[code] = {
                "accuracy": 0.85,
                "f1": 0.83,
                "ci": {"lower": 0.80, "upper": 0.90, "mean": 0.85},
                "n_samples": 500,
            }
        return {
            "model": "aya-expanse-8b",
            "seed": 42,
            "timestamp": "2026-03-07T12:00:00Z",
            "primary_metric": "accuracy",
            "metrics": ["accuracy", "f1"],
            "prompt_template": {
                "type": "chat",
                "instruction_language": "native",
                "n_shot": 5,
                "version": "1.0",
                "exemplars": {"en": "...", "es": "...", "hi": "...", "zh": "..."},
            },
            "per_language": per_language,
            "family_aggregation": {"Indo-European": 0.85, "Sino-Tibetan": 0.85},
            "variance": 0.001,
            "temperature": 0.0,
            # Reasoning evaluation data (RE-1 through RE-5)
            "mean_cot_quality": 0.75,
            "reasoning": {
                "en": {
                    "mathematical": {"cot_quality": 0.8, "n_steps_found": 3},
                    "causal": {"cot_quality": 0.7, "n_steps_found": 2},
                },
                "es": {
                    "mathematical": {"cot_quality": 0.7, "n_steps_found": 3},
                    "causal": {"cot_quality": 0.65, "n_steps_found": 2},
                },
            },
            "reasoning_categories": ["mathematical", "causal"],
            "reasoning_family_scores": {"Indo-European": 0.72},
        }

    def test_compliant_results_pass(self):
        from aya_distill.eval.checklist import validate_results

        report = validate_results(self._compliant_results())
        assert report.compliant, (
            f"Expected compliant, got {report.n_required_failed} required failures"
        )

    def test_minimal_results_fail(self):
        from aya_distill.eval.checklist import validate_results

        report = validate_results({})
        assert not report.compliant
        assert report.n_required_failed > 0

    def test_report_properties(self):
        from aya_distill.eval.checklist import validate_results

        report = validate_results(self._compliant_results())
        assert report.n_total == 30
        assert report.n_passed + report.n_failed + report.n_manual == 30
        assert 0.0 <= report.score <= 1.0

    def test_n_passed_property(self):
        from aya_distill.eval.checklist import validate_results

        report = validate_results(self._compliant_results())
        assert report.n_passed > 0

    def test_skip_items(self):
        from aya_distill.eval.checklist import validate_results

        report = validate_results({}, skip_items={"EP-1", "EP-2", "EP-3"})
        # Those 3 should now be PASS
        skipped = [
            r for r in report.results if r.item.id in {"EP-1", "EP-2", "EP-3"}
        ]
        assert all(r.passed for r in skipped)

    def test_category_summaries(self):
        from aya_distill.eval.checklist import validate_results

        report = validate_results(self._compliant_results())
        summaries = report.category_summaries
        assert len(summaries) == 8
        for s in summaries:
            assert s.n_total > 0

    def test_to_dict_serializable(self):
        from aya_distill.eval.checklist import validate_results

        report = validate_results(self._compliant_results())
        d = report.to_dict()
        json.dumps(d)  # Must be JSON-serializable

    def test_check_status_truthiness(self):
        from aya_distill.eval.checklist import CheckStatus

        assert bool(CheckStatus.PASS) is True
        assert bool(CheckStatus.FAIL) is False
        assert bool(CheckStatus.MANUAL_CHECK_NEEDED) is True


# ===========================================================================
# 5. Multilingual Testing
# ===========================================================================


class TestMultilingualTestSuite:
    """Tests for MultilingualTestSuite with an echo model."""

    @staticmethod
    def _echo_model(prompt: str) -> str:
        return f"Response to your question about: {prompt[:50]}"

    def test_basic_run(self):
        from aya_distill.testing import MultilingualTestSuite

        suite = MultilingualTestSuite(
            model_fn=self._echo_model, languages=["en", "es"]
        )
        result = suite.run()
        assert "en" in result.language_results
        assert "es" in result.language_results

    def test_per_language_results_populated(self):
        from aya_distill.testing import MultilingualTestSuite

        suite = MultilingualTestSuite(
            model_fn=self._echo_model, languages=["en", "es", "hi"]
        )
        result = suite.run()
        for code in ("en", "es", "hi"):
            lr = result.language_results[code]
            assert lr.lang_code == code
            assert lr.coherence_score > 0
            assert lr.latency_seconds >= 0

    def test_mean_coherence_positive(self):
        from aya_distill.testing import MultilingualTestSuite

        suite = MultilingualTestSuite(
            model_fn=self._echo_model, languages=["en", "es"]
        )
        result = suite.run()
        assert result.mean_coherence > 0

    def test_equity_score_computed(self):
        from aya_distill.testing import MultilingualTestSuite

        suite = MultilingualTestSuite(
            model_fn=self._echo_model, languages=["en", "es"]
        )
        result = suite.run()
        # Equity score is variance of coherence -- should be a valid float
        assert isinstance(result.equity_score, float)
        assert result.equity_score >= 0.0

    def test_cross_lingual_consistency(self):
        from aya_distill.testing import MultilingualTestSuite

        suite = MultilingualTestSuite(
            model_fn=self._echo_model, languages=["en", "es", "fr"]
        )
        result = suite.run()
        # consistency = 1 - std(coherence), should be in (0, 1]
        assert 0.0 < result.cross_lingual_consistency <= 1.0

    def test_subset_of_languages(self):
        """Only requested languages are tested, not all 67."""
        from aya_distill.testing import MultilingualTestSuite

        suite = MultilingualTestSuite(
            model_fn=self._echo_model, languages=["ja", "ko"]
        )
        result = suite.run()
        assert set(result.language_results.keys()) == {"ja", "ko"}

    def test_timestamp_set(self):
        from aya_distill.testing import MultilingualTestSuite

        suite = MultilingualTestSuite(
            model_fn=self._echo_model, languages=["en"]
        )
        result = suite.run()
        assert len(result.timestamp) > 0

    def test_to_dict_serializable(self):
        from aya_distill.testing import MultilingualTestSuite

        suite = MultilingualTestSuite(
            model_fn=self._echo_model, languages=["en"]
        )
        result = suite.run()
        d = result.to_dict()
        json.dumps(d)

    def test_summary_string(self):
        from aya_distill.testing import MultilingualTestSuite

        suite = MultilingualTestSuite(
            model_fn=self._echo_model, languages=["en"]
        )
        result = suite.run()
        s = result.summary()
        assert "Multilingual Test Results" in s

    def test_model_failure_handled(self):
        """If model raises, the language is marked as failed."""
        from aya_distill.testing import MultilingualTestSuite

        def failing_model(prompt: str) -> str:
            raise RuntimeError("GPU on fire")

        suite = MultilingualTestSuite(
            model_fn=failing_model, languages=["en"]
        )
        result = suite.run()
        assert "en" in result.language_results
        assert result.language_results["en"].passed is False

    def test_default_languages_is_all(self):
        """When no languages specified, defaults to all from LANGUAGE_NAMES."""
        from aya_distill.testing import MultilingualTestSuite
        from aya_distill.languages import LANGUAGE_NAMES

        suite = MultilingualTestSuite(model_fn=self._echo_model)
        assert suite.languages == list(LANGUAGE_NAMES.keys())


# ===========================================================================
# 6. Tool Testing
# ===========================================================================


class TestToolCallingTest:
    """Tests for ToolCallingTest."""

    def test_json_producing_model(self):
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
        assert result.score == pytest.approx(1.0)

    def test_invalid_json_model(self):
        from aya_distill.testing import ToolCallingTest

        def bad_model(prompt: str) -> str:
            return "I don't know how to call tools"

        test = ToolCallingTest(model_fn=bad_model)
        result = test.test_single(
            prompt="Call get_weather",
            expected_function="get_weather",
        )
        assert not result.json_valid
        assert not result.function_correct
        assert result.score == 0.0

    def test_wrong_function(self):
        from aya_distill.testing import ToolCallingTest

        def wrong_fn_model(prompt: str) -> str:
            return '{"function": "search_web", "arguments": {"query": "weather"}}'

        test = ToolCallingTest(model_fn=wrong_fn_model)
        result = test.test_single(
            prompt="Weather?",
            expected_function="get_weather",
        )
        assert result.json_valid
        assert not result.function_correct

    def test_partial_args(self):
        from aya_distill.testing import ToolCallingTest

        def partial_model(prompt: str) -> str:
            return '{"function": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}'

        test = ToolCallingTest(model_fn=partial_model)
        result = test.test_single(
            prompt="Weather?",
            expected_function="get_weather",
            expected_args={"location": "Paris", "unit": "fahrenheit"},
        )
        assert result.function_correct
        assert not result.args_correct  # unit mismatch
        assert 0.0 < result.score < 1.0

    def test_model_exception_handled(self):
        from aya_distill.testing import ToolCallingTest

        def exploding(prompt: str) -> str:
            raise ValueError("boom")

        test = ToolCallingTest(model_fn=exploding)
        result = test.test_single(prompt="test", expected_function="f")
        assert not result.json_valid
        assert result.score == 0.0


class TestJSONExtraction:
    """Tests for ToolCallingTest._extract_json static method."""

    def test_raw_json(self):
        from aya_distill.testing.tool_testing import ToolCallingTest

        text = '{"function": "foo", "arguments": {}}'
        result = ToolCallingTest._extract_json(text)
        assert result is not None
        assert result["function"] == "foo"

    def test_markdown_fenced_json(self):
        from aya_distill.testing.tool_testing import ToolCallingTest

        text = 'Here is the result:\n```json\n{"function": "bar"}\n```'
        result = ToolCallingTest._extract_json(text)
        assert result is not None
        assert result["function"] == "bar"

    def test_embedded_json(self):
        from aya_distill.testing.tool_testing import ToolCallingTest

        text = 'Sure! Here is the call: {"function": "baz", "args": {}} hope that helps.'
        result = ToolCallingTest._extract_json(text)
        assert result is not None
        assert result["function"] == "baz"

    def test_no_json(self):
        from aya_distill.testing.tool_testing import ToolCallingTest

        text = "There is no JSON here at all."
        result = ToolCallingTest._extract_json(text)
        assert result is None

    def test_empty_string(self):
        from aya_distill.testing.tool_testing import ToolCallingTest

        result = ToolCallingTest._extract_json("")
        assert result is None

    def test_nested_braces(self):
        from aya_distill.testing.tool_testing import ToolCallingTest

        text = '{"function": "f", "arguments": {"nested": {"key": "val"}}}'
        result = ToolCallingTest._extract_json(text)
        assert result is not None
        assert result["arguments"]["nested"]["key"] == "val"


class TestToolCallingMultilingual:
    """Tests for run_multilingual."""

    def test_run_multilingual_with_subset(self):
        from aya_distill.testing.tool_testing import ToolCallingTest

        def json_model(prompt: str) -> str:
            return '{"function": "get_weather", "arguments": {"location": "City"}}'

        test = ToolCallingTest(model_fn=json_model)
        scenarios = {
            "en": {
                "prompt": "Weather in London?",
                "expected_function": "get_weather",
                "expected_args": {"location": "City"},
            },
            "es": {
                "prompt": "Clima en Madrid?",
                "expected_function": "get_weather",
                "expected_args": {"location": "City"},
            },
        }
        results = test.run_multilingual(scenarios=scenarios)
        assert "en" in results
        assert "es" in results
        assert results["en"].json_valid
        assert results["en"].args_correct


# ===========================================================================
# 7. Fixtures
# ===========================================================================


class TestFixtures:
    """Tests for test fixture data completeness."""

    def test_multilingual_prompts_has_67_languages(self):
        from aya_distill.testing.fixtures import MULTILINGUAL_PROMPTS

        gk = MULTILINGUAL_PROMPTS["general_knowledge"]
        assert len(gk) == 67, f"Expected 67 languages, got {len(gk)}"

    def test_tool_scenarios_has_67_languages(self):
        from aya_distill.testing.fixtures import TOOL_SCENARIOS

        assert len(TOOL_SCENARIOS) == 67, (
            f"Expected 67 tool scenarios, got {len(TOOL_SCENARIOS)}"
        )

    def test_all_prompt_codes_in_languages(self):
        from aya_distill.testing.fixtures import MULTILINGUAL_PROMPTS
        from aya_distill.languages import LANGUAGES

        for code in MULTILINGUAL_PROMPTS["general_knowledge"]:
            assert code in LANGUAGES, (
                f"Prompt code '{code}' not in LANGUAGES registry"
            )

    def test_all_scenario_codes_in_languages(self):
        from aya_distill.testing.fixtures import TOOL_SCENARIOS
        from aya_distill.languages import LANGUAGES

        for code in TOOL_SCENARIOS:
            assert code in LANGUAGES, (
                f"Scenario code '{code}' not in LANGUAGES registry"
            )

    def test_tool_schemas_structure(self):
        from aya_distill.testing.fixtures import TOOL_SCHEMAS

        assert len(TOOL_SCHEMAS) == 3
        for schema in TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert schema["parameters"]["type"] == "object"
            assert "properties" in schema["parameters"]

    def test_tool_schemas_names(self):
        from aya_distill.testing.fixtures import TOOL_SCHEMAS

        names = {s["name"] for s in TOOL_SCHEMAS}
        assert names == {"get_weather", "search_web", "translate"}

    def test_all_scenarios_have_required_keys(self):
        from aya_distill.testing.fixtures import TOOL_SCENARIOS

        for code, scenario in TOOL_SCENARIOS.items():
            assert "prompt" in scenario, f"{code} missing 'prompt'"
            assert "expected_function" in scenario, (
                f"{code} missing 'expected_function'"
            )
            assert "expected_args" in scenario, (
                f"{code} missing 'expected_args'"
            )

    def test_prompts_and_scenarios_same_codes(self):
        """Fixtures should cover the same set of language codes."""
        from aya_distill.testing.fixtures import (
            MULTILINGUAL_PROMPTS,
            TOOL_SCENARIOS,
        )

        prompt_codes = set(MULTILINGUAL_PROMPTS["general_knowledge"].keys())
        scenario_codes = set(TOOL_SCENARIOS.keys())
        assert prompt_codes == scenario_codes


# ===========================================================================
# 8. Integration
# ===========================================================================


class TestIntegration:
    """End-to-end pipeline tests: suite -> run -> checklist."""

    def test_multilingual_results_feed_into_checklist(self):
        """Run MultilingualTestSuite, convert to results dict, validate."""
        from aya_distill.testing import MultilingualTestSuite
        from aya_distill.eval.checklist import validate_results

        def echo(prompt: str) -> str:
            return f"This is a detailed response about: {prompt[:80]}. It is quite thorough."

        suite = MultilingualTestSuite(
            model_fn=echo, languages=["en", "es", "hi", "zh"]
        )
        run_result = suite.run()

        # Convert to checklist-compatible dict
        per_language: dict[str, Any] = {}
        for code, lr in run_result.language_results.items():
            per_language[code] = {
                "accuracy": lr.coherence_score,
                "ci": {"lower": lr.coherence_score - 0.05, "upper": lr.coherence_score + 0.05},
                "n_samples": lr.token_count,
            }

        results_dict: dict[str, Any] = {
            "model": "test-echo-model",
            "seed": 42,
            "timestamp": run_result.timestamp,
            "primary_metric": "accuracy",
            "prompt_template": {
                "type": "direct",
                "instruction_language": "native",
                "n_shot": 0,
                "version": "1.0",
            },
            "per_language": per_language,
            "family_aggregation": {"Indo-European": 0.8, "Sino-Tibetan": 0.8},
            "variance": run_result.equity_score,
        }

        report = validate_results(results_dict)
        # Should pass most required items
        assert report.n_total == 30
        assert report.score > 0.5

    def test_tool_test_round_trip(self):
        """Run ToolCallingTest with scenarios, verify structure."""
        from aya_distill.testing.tool_testing import ToolCallingTest

        def json_model(prompt: str) -> str:
            return '{"function": "get_weather", "arguments": {"location": "Test"}}'

        test = ToolCallingTest(model_fn=json_model)
        scenarios = {
            "en": {
                "prompt": "Weather?",
                "expected_function": "get_weather",
                "expected_args": {"location": "Test"},
            },
        }
        results = test.run_multilingual(scenarios=scenarios)

        # Verify structure
        assert "en" in results
        r = results["en"]
        assert r.json_valid
        assert r.function_correct
        assert r.args_correct

        # Verify serialization
        d = r.to_dict()
        json.dumps(d)

    def test_compare_teacher_student(self):
        """Test MultilingualTestSuite.compare method."""
        from aya_distill.testing import MultilingualTestSuite

        def teacher(prompt: str) -> str:
            return f"Teacher gives a very detailed and thorough answer about: {prompt[:80]}"

        def student(prompt: str) -> str:
            return f"Short answer: {prompt[:20]}"

        suite = MultilingualTestSuite(
            model_fn=student, languages=["en", "es"]
        )
        degradation = suite.compare(teacher_fn=teacher)

        assert "en" in degradation
        assert "es" in degradation
        assert "coherence_delta" in degradation["en"]
        assert "relevance_delta" in degradation["en"]


# ===========================================================================
# 9. Reasoning Tests
# ===========================================================================


class TestCoTScoring:
    """Tests for chain-of-thought quality scoring."""

    def test_good_cot_response(self):
        from aya_distill.testing.reasoning import _score_cot_quality

        text = (
            "Step 1: First, we calculate 4 * 2 = 8.\n"
            "Step 2: Then, we calculate 5 * 3 = 15.\n"
            "Step 3: Therefore, the total is 8 + 15 = 23.\n"
            "#### 23"
        )
        score = _score_cot_quality(text)
        assert score > 0.6, f"Good CoT should score > 0.6, got {score}"

    def test_no_cot_response(self):
        from aya_distill.testing.reasoning import _score_cot_quality

        score = _score_cot_quality("23")
        assert score < 0.4, f"Bare answer should score low, got {score}"

    def test_empty_response(self):
        from aya_distill.testing.reasoning import _score_cot_quality

        assert _score_cot_quality("") == 0.0
        assert _score_cot_quality("   ") == 0.0

    def test_medium_cot(self):
        from aya_distill.testing.reasoning import _score_cot_quality

        text = "We need to add the costs. 4 apples cost 8 and 5 oranges cost 15. So the answer is 23."
        score = _score_cot_quality(text)
        assert 0.2 < score < 0.9


class TestLogicalConsistency:
    """Tests for logical consistency scoring."""

    def test_progressive_reasoning(self):
        from aya_distill.testing.reasoning import _score_logical_consistency

        text = (
            "First, all birds can fly. Second, penguins are birds. "
            "Therefore, penguins can fly."
        )
        score = _score_logical_consistency(text)
        assert score > 0.6

    def test_contradictory_response(self):
        from aya_distill.testing.reasoning import _score_logical_consistency

        text = "Yes, penguins can fly. No, penguins cannot fly."
        score = _score_logical_consistency(text)
        # Should be penalized for contradiction
        assert score < 0.7

    def test_neutral_response(self):
        from aya_distill.testing.reasoning import _score_logical_consistency

        text = "The answer is A."
        score = _score_logical_consistency(text)
        assert 0.3 <= score <= 0.7


class TestNumericExtraction:
    """Tests for numeric answer extraction."""

    def test_hash_delimiter(self):
        from aya_distill.testing.reasoning import _extract_numeric_answer

        assert _extract_numeric_answer("some work...\n#### 23") == "23"

    def test_equals_sign(self):
        from aya_distill.testing.reasoning import _extract_numeric_answer

        assert _extract_numeric_answer("total = 42") == "42"

    def test_answer_prefix(self):
        from aya_distill.testing.reasoning import _extract_numeric_answer

        assert _extract_numeric_answer("The answer: 100") == "100"

    def test_fallback_last_number(self):
        from aya_distill.testing.reasoning import _extract_numeric_answer

        result = _extract_numeric_answer("4 times 2 is 8 plus 15 is 23")
        assert result == "23"

    def test_no_numbers(self):
        from aya_distill.testing.reasoning import _extract_numeric_answer

        assert _extract_numeric_answer("no numbers here") is None


class TestChoiceExtraction:
    """Tests for choice answer extraction."""

    def test_answer_prefix(self):
        from aya_distill.testing.reasoning import _extract_choice_answer

        assert _extract_choice_answer("The answer: A") == "A"

    def test_therefore_prefix(self):
        from aya_distill.testing.reasoning import _extract_choice_answer

        assert _extract_choice_answer("therefore: B") == "B"

    def test_standalone_letter(self):
        from aya_distill.testing.reasoning import _extract_choice_answer

        result = _extract_choice_answer("The correct option is C because...")
        assert result == "C"

    def test_no_choice(self):
        from aya_distill.testing.reasoning import _extract_choice_answer

        assert _extract_choice_answer("I'm not sure about the answer.") is None


class TestReasoningStepCounting:
    """Tests for counting reasoning steps."""

    def test_numbered_steps(self):
        from aya_distill.testing.reasoning import _count_reasoning_steps

        text = "1. Calculate 4*2=8\n2. Calculate 5*3=15\n3. Add 8+15=23"
        assert _count_reasoning_steps(text) >= 3

    def test_connective_steps(self):
        from aya_distill.testing.reasoning import _count_reasoning_steps

        text = "First we add. Then we multiply. Finally we get the answer."
        assert _count_reasoning_steps(text) >= 2

    def test_minimal_text(self):
        from aya_distill.testing.reasoning import _count_reasoning_steps

        assert _count_reasoning_steps("23") >= 0


class TestReasoningTestSuite:
    """Tests for ReasoningTestSuite with mock models."""

    @staticmethod
    def _cot_model(prompt: str) -> str:
        return (
            "Step 1: First, calculate 4 * 2 = 8 for the apples.\n"
            "Step 2: Then, calculate 5 * 3 = 15 for the oranges.\n"
            "Step 3: Therefore, the total is 8 + 15 = 23.\n"
            "#### 23"
        )

    @staticmethod
    def _bare_model(prompt: str) -> str:
        return "23"

    def test_basic_run(self):
        from aya_distill.testing import ReasoningTestSuite

        suite = ReasoningTestSuite(
            model_fn=self._cot_model,
            languages=["en", "es"],
            categories=["mathematical"],
        )
        result = suite.run()
        assert "en" in result.results
        assert "mathematical" in result.results["en"]

    def test_cot_model_scores_higher(self):
        from aya_distill.testing import ReasoningTestSuite

        cot_suite = ReasoningTestSuite(
            model_fn=self._cot_model,
            languages=["en"],
            categories=["mathematical"],
        )
        bare_suite = ReasoningTestSuite(
            model_fn=self._bare_model,
            languages=["en"],
            categories=["mathematical"],
        )

        cot_result = cot_suite.run()
        bare_result = bare_suite.run()

        cot_score = cot_result.results["en"]["mathematical"].score
        bare_score = bare_result.results["en"]["mathematical"].score
        assert cot_score > bare_score, (
            f"CoT model ({cot_score}) should score higher than bare ({bare_score})"
        )

    def test_all_four_categories(self):
        from aya_distill.testing import ReasoningTestSuite

        suite = ReasoningTestSuite(
            model_fn=self._cot_model,
            languages=["en"],
        )
        result = suite.run()
        categories = set(result.results["en"].keys())
        assert categories == {"mathematical", "causal", "analogical", "logical"}

    def test_aggregate_metrics(self):
        from aya_distill.testing import ReasoningTestSuite

        suite = ReasoningTestSuite(
            model_fn=self._cot_model,
            languages=["en", "es"],
            categories=["mathematical"],
        )
        result = suite.run()
        assert result.mean_score > 0
        assert result.mean_cot_quality > 0
        assert result.mean_logical_consistency > 0
        assert isinstance(result.cot_equity_score, float)

    def test_family_scores(self):
        from aya_distill.testing import ReasoningTestSuite

        suite = ReasoningTestSuite(
            model_fn=self._cot_model,
            languages=["en", "hi", "tr", "sw"],
            categories=["mathematical"],
        )
        result = suite.run()
        assert len(result.family_scores) > 0

    def test_summary_string(self):
        from aya_distill.testing import ReasoningTestSuite

        suite = ReasoningTestSuite(
            model_fn=self._cot_model,
            languages=["en"],
            categories=["mathematical"],
        )
        result = suite.run()
        s = result.summary()
        assert "Multilingual Reasoning Results" in s
        assert "mathematical" in s

    def test_to_dict_serializable(self):
        from aya_distill.testing import ReasoningTestSuite

        suite = ReasoningTestSuite(
            model_fn=self._cot_model,
            languages=["en"],
            categories=["mathematical"],
        )
        result = suite.run()
        d = result.to_dict()
        json.dumps(d)

    def test_model_failure_handled(self):
        from aya_distill.testing import ReasoningTestSuite

        def fail(prompt: str) -> str:
            raise RuntimeError("OOM")

        suite = ReasoningTestSuite(
            model_fn=fail,
            languages=["en"],
            categories=["mathematical"],
        )
        result = suite.run()
        assert "en" in result.results
        r = result.results["en"]["mathematical"]
        assert r.score == 0.0

    def test_compare_teacher_student(self):
        from aya_distill.testing import ReasoningTestSuite

        suite = ReasoningTestSuite(
            model_fn=self._bare_model,
            languages=["en"],
            categories=["mathematical"],
        )
        degradation = suite.compare(teacher_fn=self._cot_model)
        assert "en" in degradation
        assert "mathematical_delta" in degradation["en"]

    def test_causal_reasoning(self):
        from aya_distill.testing import ReasoningTestSuite

        def causal_model(prompt: str) -> str:
            return (
                "The road is wet. Since rain causes roads to become wet, "
                "the most likely cause is rain. Therefore, the answer: A"
            )

        suite = ReasoningTestSuite(
            model_fn=causal_model,
            languages=["en"],
            categories=["causal"],
        )
        result = suite.run()
        r = result.results["en"]["causal"]
        assert r.final_answer_correct
        assert r.cot_quality > 0

    def test_logical_reasoning(self):
        from aya_distill.testing import ReasoningTestSuite

        def logical_model(prompt: str) -> str:
            return (
                "Given that all birds can fly, and penguins are birds, "
                "therefore penguins can fly. The answer: A"
            )

        suite = ReasoningTestSuite(
            model_fn=logical_model,
            languages=["en"],
            categories=["logical"],
        )
        result = suite.run()
        r = result.results["en"]["logical"]
        assert r.final_answer_correct
        assert r.logical_consistency > 0


class TestReasoningFixtures:
    """Tests for reasoning fixture data."""

    def test_reasoning_scenarios_has_four_categories(self):
        from aya_distill.testing.fixtures import REASONING_SCENARIOS

        assert set(REASONING_SCENARIOS.keys()) == {
            "mathematical", "causal", "analogical", "logical"
        }

    def test_each_category_has_languages(self):
        from aya_distill.testing.fixtures import REASONING_SCENARIOS

        for cat, scenarios in REASONING_SCENARIOS.items():
            assert len(scenarios) >= 20, (
                f"Category {cat} has only {len(scenarios)} languages, expected >= 20"
            )

    def test_scenarios_have_required_keys(self):
        from aya_distill.testing.fixtures import REASONING_SCENARIOS

        for cat, scenarios in REASONING_SCENARIOS.items():
            for lang, scenario in scenarios.items():
                assert "prompt" in scenario, f"{cat}/{lang} missing 'prompt'"
                assert "expected_answer" in scenario, f"{cat}/{lang} missing 'expected_answer'"
                assert "answer_type" in scenario, f"{cat}/{lang} missing 'answer_type'"

    def test_math_answers_are_23(self):
        from aya_distill.testing.fixtures import REASONING_SCENARIOS

        for lang, scenario in REASONING_SCENARIOS["mathematical"].items():
            assert scenario["expected_answer"] == "23", (
                f"Math answer for {lang} should be 23"
            )
            assert scenario["answer_type"] == "numeric"

    def test_causal_answers_are_A(self):
        from aya_distill.testing.fixtures import REASONING_SCENARIOS

        for lang, scenario in REASONING_SCENARIOS["causal"].items():
            assert scenario["expected_answer"] == "A"
            assert scenario["answer_type"] == "choice"

    def test_cot_instructions_multilingual(self):
        from aya_distill.testing.fixtures import _COT_INSTRUCTION

        # Should cover at least 30 languages
        assert len(_COT_INSTRUCTION) >= 30

    def test_all_categories_same_languages(self):
        from aya_distill.testing.fixtures import REASONING_SCENARIOS

        langs_per_cat = [
            set(scenarios.keys())
            for scenarios in REASONING_SCENARIOS.values()
        ]
        # All categories should cover the same language set
        for i in range(1, len(langs_per_cat)):
            assert langs_per_cat[0] == langs_per_cat[i]


class TestReasoningChecklist:
    """Tests for reasoning-specific checklist items."""

    def test_checklist_has_30_items(self):
        from aya_distill.eval.checklist import CHECKLIST_ITEMS

        assert len(CHECKLIST_ITEMS) == 30

    def test_reasoning_category_exists(self):
        from aya_distill.eval.checklist import CATEGORY_NAMES

        assert "Reasoning Evaluation" in CATEGORY_NAMES

    def test_reasoning_items_count(self):
        from aya_distill.eval.checklist import CHECKLIST_ITEMS

        reasoning = [i for i in CHECKLIST_ITEMS if i.category == "Reasoning Evaluation"]
        assert len(reasoning) == 5

    def test_reasoning_required_items(self):
        from aya_distill.eval.checklist import CHECKLIST_ITEMS

        required = [
            i for i in CHECKLIST_ITEMS
            if i.category == "Reasoning Evaluation" and i.required
        ]
        assert len(required) == 3  # RE-1, RE-2, RE-3

    def test_reasoning_checklist_ids(self):
        from aya_distill.eval.checklist import CHECKLIST_INDEX

        for i in range(1, 6):
            assert f"RE-{i}" in CHECKLIST_INDEX

    def test_validate_with_reasoning_data(self):
        """Full reasoning results should pass RE-1 through RE-3."""
        from aya_distill.eval.checklist import validate_results

        results = {
            "model": "test-model",
            "seed": 42,
            "timestamp": "2026-03-07T00:00:00Z",
            "primary_metric": "reasoning_score",
            "prompt_template": {
                "type": "cot",
                "instruction_language": "native",
                "n_shot": 0,
                "version": "1.0",
            },
            "per_language": {
                "en": {"accuracy": 0.8, "ci": {"lower": 0.7, "upper": 0.9}, "n_samples": 100},
                "es": {"accuracy": 0.75, "ci": {"lower": 0.65, "upper": 0.85}, "n_samples": 100},
            },
            "family_aggregation": {"Indo-European": 0.77},
            "variance": 0.001,
            "mean_cot_quality": 0.75,
            "reasoning": {
                "en": {
                    "mathematical": {"cot_quality": 0.8, "n_steps_found": 3},
                    "causal": {"cot_quality": 0.7, "n_steps_found": 2},
                },
                "es": {
                    "mathematical": {"cot_quality": 0.7, "n_steps_found": 3},
                    "causal": {"cot_quality": 0.65, "n_steps_found": 2},
                },
            },
            "reasoning_categories": ["mathematical", "causal"],
            "reasoning_family_scores": {"Indo-European": 0.72},
            "cot_equity_score": 0.002,
        }

        report = validate_results(results)
        # Check reasoning items specifically
        reasoning_results = {
            r.item.id: r for r in report.results
            if r.item.id.startswith("RE-")
        }
        assert reasoning_results["RE-1"].status.value == "pass"
        assert reasoning_results["RE-2"].status.value == "pass"
        assert reasoning_results["RE-3"].status.value == "pass"

    def test_validate_without_reasoning_fails(self):
        """Missing reasoning data should fail RE-1, RE-2, RE-3."""
        from aya_distill.eval.checklist import validate_results

        results = {
            "model": "test-model",
            "seed": 42,
            "timestamp": "2026-03-07T00:00:00Z",
            "primary_metric": "accuracy",
            "per_language": {"en": {"accuracy": 0.8}},
        }
        report = validate_results(results)
        reasoning_fails = [
            r for r in report.results
            if r.item.id.startswith("RE-") and r.item.required and not r.passed
        ]
        assert len(reasoning_fails) == 3  # RE-1, RE-2, RE-3


class TestReasoningIntegration:
    """End-to-end: reasoning suite -> checklist validation."""

    def test_reasoning_results_feed_into_checklist(self):
        from aya_distill.testing import ReasoningTestSuite
        from aya_distill.eval.checklist import validate_results

        def cot_model(prompt: str) -> str:
            return (
                "Step 1: First, 4 * 2 = 8.\n"
                "Step 2: Then, 5 * 3 = 15.\n"
                "Step 3: Therefore, 8 + 15 = 23.\n"
                "#### 23"
            )

        suite = ReasoningTestSuite(
            model_fn=cot_model,
            languages=["en", "es", "hi"],
            categories=["mathematical", "causal"],
        )
        run_result = suite.run()

        # Build checklist-compatible results dict
        results_dict: dict[str, Any] = {
            "model": "test-cot-model",
            "seed": 42,
            "timestamp": run_result.timestamp,
            "primary_metric": "reasoning_score",
            "prompt_template": {
                "type": "cot",
                "instruction_language": "native",
                "n_shot": 0,
                "version": "1.0",
            },
            "per_language": {
                lang: {
                    "accuracy": np.mean([r.score for r in cats.values()]),
                    "ci": {"lower": 0.5, "upper": 0.9},
                    "n_samples": 4,
                }
                for lang, cats in run_result.results.items()
            },
            "family_aggregation": run_result.family_scores,
            "variance": run_result.cot_equity_score,
            "mean_cot_quality": run_result.mean_cot_quality,
            "reasoning": {
                lang: {cat: r.to_dict() for cat, r in cats.items()}
                for lang, cats in run_result.results.items()
            },
            "reasoning_categories": ["mathematical", "causal"],
            "reasoning_family_scores": run_result.family_scores,
            "cot_equity_score": run_result.cot_equity_score,
        }

        report = validate_results(results_dict)
        # All 3 required reasoning items should pass
        re_results = {
            r.item.id: r for r in report.results if r.item.id.startswith("RE-")
        }
        assert re_results["RE-1"].passed
        assert re_results["RE-2"].passed
        assert re_results["RE-3"].passed
