"""
Shared metrics utilities for multilingual evaluation.

Provides bootstrap confidence intervals, degradation equity scoring,
language-family aggregation, and paired statistical testing. These
are the building blocks for every benchmark in the Aya evaluation
pipeline -- never hide language variation behind a single average.

Wayy Research -- Project Aya
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from aya_distill.languages import LANGUAGE_FAMILIES, LANGUAGES


# ---------------------------------------------------------------------------
# Language family taxonomy (imported from aya_distill.languages)
# ---------------------------------------------------------------------------

LANG_TO_FAMILY: dict[str, str] = {
    lang: family
    for family, langs in LANGUAGE_FAMILIES.items()
    for lang in langs
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceInterval:
    """Bootstrap confidence interval for a single metric."""

    mean: float
    lower: float
    upper: float
    std: float
    n_bootstrap: int
    confidence_level: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "std": self.std,
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Result of paired bootstrap significance test."""

    p_value: float
    mean_diff: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_bootstrap: int

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "p_value": self.p_value,
            "mean_diff": self.mean_diff,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "significant": self.significant,
            "n_bootstrap": self.n_bootstrap,
        }


@dataclass
class LanguageResult:
    """Per-language accuracy + confidence interval."""

    language: str
    accuracy: float
    ci: ConfidenceInterval
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "accuracy": self.accuracy,
            "ci": self.ci.to_dict(),
            "n_samples": self.n_samples,
        }


@dataclass
class FamilyResult:
    """Aggregated result for a language family."""

    family: str
    languages: list[str]
    mean_accuracy: float
    accuracies: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "languages": self.languages,
            "mean_accuracy": self.mean_accuracy,
            "accuracies": self.accuracies,
        }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_confidence_interval(
    correct: NDArray[np.int_],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval for accuracy.

    Parameters
    ----------
    correct : array of 0/1 indicating per-sample correctness
    n_bootstrap : number of bootstrap resamples
    confidence_level : e.g. 0.95 for 95 % CI
    seed : for reproducibility -- always set the seed

    Returns
    -------
    ConfidenceInterval with mean, lower, upper, std
    """
    rng = np.random.RandomState(seed)
    n = len(correct)
    if n == 0:
        return ConfidenceInterval(
            mean=0.0,
            lower=0.0,
            upper=0.0,
            std=0.0,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )

    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        boot_means[i] = correct[indices].mean()

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(
        mean=float(correct.mean()),
        lower=lower,
        upper=upper,
        std=float(boot_means.std()),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


# ---------------------------------------------------------------------------
# Paired bootstrap significance test
# ---------------------------------------------------------------------------


def paired_bootstrap_test(
    correct_a: NDArray[np.int_],
    correct_b: NDArray[np.int_],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Two-sided paired bootstrap test for difference in accuracy.

    Tests H0: acc(A) == acc(B) against H1: acc(A) != acc(B).

    Parameters
    ----------
    correct_a, correct_b : per-sample correctness arrays, same length
    n_bootstrap : number of bootstrap resamples
    confidence_level : CI level for the difference
    seed : reproducibility

    Returns
    -------
    PairedBootstrapResult with p-value, mean difference, CI, significance
    """
    assert len(correct_a) == len(correct_b), (
        f"Arrays must have same length: {len(correct_a)} vs {len(correct_b)}"
    )
    rng = np.random.RandomState(seed)
    n = len(correct_a)

    observed_diff = float(correct_a.mean() - correct_b.mean())

    boot_diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        boot_diffs[i] = correct_a[indices].mean() - correct_b[indices].mean()

    # Two-sided p-value: proportion of bootstrap diffs on the other side of 0
    # relative to observed diff
    if observed_diff >= 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    else:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    return PairedBootstrapResult(
        p_value=p_value,
        mean_diff=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=p_value < (1.0 - confidence_level),
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Language-family aggregation
# ---------------------------------------------------------------------------


def aggregate_by_family(
    per_language: dict[str, float],
) -> list[FamilyResult]:
    """Group per-language accuracies into language families.

    Returns one FamilyResult per family that has at least one language
    present in *per_language*.
    """
    results: list[FamilyResult] = []
    for family, langs in LANGUAGE_FAMILIES.items():
        present = {lg: per_language[lg] for lg in langs if lg in per_language}
        if not present:
            continue
        results.append(
            FamilyResult(
                family=family,
                languages=list(present.keys()),
                mean_accuracy=float(np.mean(list(present.values()))),
                accuracies=present,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Degradation Equity Score (DES)
# ---------------------------------------------------------------------------


def degradation_equity_score(
    teacher_scores: dict[str, float],
    student_scores: dict[str, float],
) -> float:
    """Compute Degradation Equity Score.

    DES = variance of per-family accuracy *drop* after distillation.
    A lower DES means the distillation hurt all language families
    roughly equally (equitable degradation). A high DES means some
    families were disproportionately harmed.

    Parameters
    ----------
    teacher_scores : lang -> accuracy for teacher
    student_scores : lang -> accuracy for student (same keys)

    Returns
    -------
    float : variance of per-family mean accuracy drops
    """
    common_langs = set(teacher_scores) & set(student_scores)
    drops = {lg: teacher_scores[lg] - student_scores[lg] for lg in common_langs}

    family_drops: list[float] = []
    for family, langs in LANGUAGE_FAMILIES.items():
        present = [drops[lg] for lg in langs if lg in drops]
        if present:
            family_drops.append(float(np.mean(present)))

    if len(family_drops) < 2:
        return 0.0

    return float(np.var(family_drops))


# ---------------------------------------------------------------------------
# Utility: compute accuracy from predictions
# ---------------------------------------------------------------------------


def accuracy_from_predictions(
    predictions: Sequence[str],
    references: Sequence[str],
) -> NDArray[np.int_]:
    """Return binary array: 1 where prediction matches reference."""
    assert len(predictions) == len(references)
    correct = np.array(
        [int(p.strip() == r.strip()) for p, r in zip(predictions, references)],
        dtype=np.int_,
    )
    return correct
