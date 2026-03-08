"""Multilingual reasoning tests for distilled models.

Evaluates chain-of-thought (CoT) quality, step correctness, logical
consistency, and cross-lingual reasoning transfer across all target
languages.  Wraps any ``model_fn: (str) -> str`` callable.

Reasoning categories tested:
  - Mathematical reasoning (multi-step arithmetic with CoT)
  - Causal reasoning (cause-effect chains)
  - Analogical reasoning (A:B :: C:?)
  - Logical reasoning (if-then deduction)

Each category produces per-language scores that feed into the Deja Vu
checklist reasoning extension items (RE-1 through RE-5).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from aya_distill.languages import LANGUAGE_NAMES, LANGUAGE_FAMILIES

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────

@dataclass
class ReasoningStepResult:
    """Result of evaluating a single reasoning step."""

    step_text: str
    correct: bool
    expected: str = ""


@dataclass
class ReasoningResult:
    """Result of a single reasoning test for one language."""

    language: str
    lang_code: str
    category: str  # mathematical, causal, analogical, logical
    has_reasoning_steps: bool = False
    n_steps_found: int = 0
    n_steps_correct: int = 0
    final_answer_correct: bool = False
    cot_quality: float = 0.0  # 0-1, how well-structured the CoT is
    logical_consistency: float = 0.0  # 0-1, are conclusions consistent with premises
    score: float = 0.0  # overall score 0-1
    latency_seconds: float = 0.0
    raw_output: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "lang_code": self.lang_code,
            "category": self.category,
            "has_reasoning_steps": self.has_reasoning_steps,
            "n_steps_found": self.n_steps_found,
            "n_steps_correct": self.n_steps_correct,
            "final_answer_correct": self.final_answer_correct,
            "cot_quality": round(self.cot_quality, 4),
            "logical_consistency": round(self.logical_consistency, 4),
            "score": round(self.score, 4),
            "latency_seconds": round(self.latency_seconds, 4),
            "raw_output": self.raw_output[:500],
        }


@dataclass
class MultilingualReasoningResult:
    """Aggregated reasoning results across languages and categories."""

    results: dict[str, dict[str, ReasoningResult]] = field(default_factory=dict)
    # results[lang_code][category] = ReasoningResult
    mean_cot_quality: float = 0.0
    mean_logical_consistency: float = 0.0
    mean_score: float = 0.0
    cot_equity_score: float = 0.0  # variance across languages (lower = fairer)
    family_scores: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def summary(self) -> str:
        lines = ["Multilingual Reasoning Results", "=" * 60]
        for lang, cats in self.results.items():
            for cat, r in cats.items():
                cot = "CoT" if r.has_reasoning_steps else "---"
                ans = "OK" if r.final_answer_correct else "X"
                lines.append(
                    f"  {lang} ({cat:>14}): score={r.score:.2f} "
                    f"cot={r.cot_quality:.2f} logic={r.logical_consistency:.2f} "
                    f"[{cot}] [{ans}]"
                )
        lines.append("")
        lines.append(f"  Mean CoT quality:         {self.mean_cot_quality:.3f}")
        lines.append(f"  Mean logical consistency: {self.mean_logical_consistency:.3f}")
        lines.append(f"  Mean score:               {self.mean_score:.3f}")
        lines.append(f"  CoT equity (lower=better):{self.cot_equity_score:.4f}")
        if self.family_scores:
            lines.append("")
            lines.append("  By language family:")
            for fam, sc in sorted(self.family_scores.items()):
                lines.append(f"    {fam:<25} {sc:.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mean_cot_quality": self.mean_cot_quality,
            "mean_logical_consistency": self.mean_logical_consistency,
            "mean_score": self.mean_score,
            "cot_equity_score": self.cot_equity_score,
            "family_scores": self.family_scores,
            "languages": {
                lang: {cat: r.to_dict() for cat, r in cats.items()}
                for lang, cats in self.results.items()
            },
        }


# ── CoT extraction and scoring ───────────────────────────────────────────

# Patterns that indicate reasoning steps in model output
_STEP_PATTERNS = [
    re.compile(r"(?:step|步骤|ステップ|단계|خطوة|चरण|adım|hatua)\s*\d+", re.IGNORECASE),
    re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE),
    re.compile(r"(?:first|second|third|then|next|finally|therefore|so|因此|したがって|따라서|لذلك|इसलिए|sonuç|kwa hiyo)", re.IGNORECASE),
    re.compile(r"(?:because|since|if.*then|dado que|parce que|weil|因为|なぜなら|때문에|لأن|क्योंकि|çünkü|kwa sababu)", re.IGNORECASE),
]

_NUMERIC_ANSWER_PATTERN = re.compile(r"(?:####|answer[:\s]|=)\s*(-?\d[\d,\.]*)")
_CHOICE_ANSWER_PATTERN = re.compile(r"(?:answer[:\s]|therefore[:\s])\s*([A-Da-d1-4])")


def _count_reasoning_steps(text: str) -> int:
    """Count how many reasoning steps appear in model output."""
    steps = 0
    for pattern in _STEP_PATTERNS:
        steps += len(pattern.findall(text))
    # Also count sentence-level indicators
    sentences = [s.strip() for s in re.split(r'[.!?。！？\n]', text) if s.strip()]
    return max(steps, len(sentences) - 1)  # at least n-1 intermediate steps


def _extract_numeric_answer(text: str) -> str | None:
    """Extract a numeric answer from model output."""
    match = _NUMERIC_ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: find the last number in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def _extract_choice_answer(text: str) -> str | None:
    """Extract a letter/number choice answer."""
    match = _CHOICE_ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    # Fallback: find first A/B/C/D or 1/2/3/4
    for pattern in [r'\b([ABCD])\b', r'\b([1-4])\b']:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _score_cot_quality(text: str, expected_answer: str | None = None) -> float:
    """Score the quality of chain-of-thought reasoning.

    Evaluates:
    - Presence of reasoning steps (0.3)
    - Step count / structure (0.2)
    - Use of logical connectives (0.2)
    - Response length adequacy (0.15)
    - Final answer presence (0.15)
    """
    score = 0.0
    text_stripped = text.strip()

    if not text_stripped:
        return 0.0

    # Presence of reasoning steps
    n_steps = _count_reasoning_steps(text_stripped)
    if n_steps >= 3:
        score += 0.3
    elif n_steps >= 1:
        score += 0.15

    # Step count / structure
    has_numbered = bool(re.search(r'^\s*\d+[\.\)]', text_stripped, re.MULTILINE))
    has_markers = bool(re.search(r'(?:first|then|next|finally|therefore|step)', text_stripped, re.IGNORECASE))
    if has_numbered:
        score += 0.2
    elif has_markers:
        score += 0.1

    # Logical connectives
    connectives = re.findall(
        r'(?:because|since|therefore|so|thus|hence|if|then|given|'
        r'因此|所以|なぜなら|したがって|따라서|그러므로|لذلك|بسبب|इसलिए|क्योंकि|'
        r'çünkü|dolayısıyla|kwa hiyo|kwa sababu)',
        text_stripped, re.IGNORECASE,
    )
    if len(connectives) >= 2:
        score += 0.2
    elif len(connectives) >= 1:
        score += 0.1

    # Response length
    word_count = len(text_stripped.split())
    if 20 <= word_count <= 500:
        score += 0.15
    elif word_count > 10:
        score += 0.08

    # Final answer presence
    has_answer = (
        _extract_numeric_answer(text_stripped) is not None
        or _extract_choice_answer(text_stripped) is not None
        or bool(re.search(r'(?:answer|result|therefore|conclusion|so|=)', text_stripped, re.IGNORECASE))
    )
    if has_answer:
        score += 0.15

    return min(score, 1.0)


def _score_logical_consistency(text: str) -> float:
    """Score logical consistency of reasoning.

    Checks for:
    - Contradictions (negative signal)
    - Progressive reasoning (positive signal)
    - Conclusion follows from premises (positive signal)
    """
    score = 0.5  # neutral starting point
    text_lower = text.lower()

    # Check for contradictions
    contradiction_pairs = [
        (r'\byes\b.*\bno\b', -0.15),
        (r'\btrue\b.*\bfalse\b', -0.15),
        (r'\bincrease\b.*\bdecrease\b', -0.1),
    ]
    for pattern, penalty in contradiction_pairs:
        if re.search(pattern, text_lower):
            score += penalty

    # Progressive reasoning
    progressive_markers = re.findall(
        r'(?:first|second|third|next|then|finally|therefore|'
        r'step\s*\d|thus|hence|consequently)',
        text_lower,
    )
    score += min(len(progressive_markers) * 0.1, 0.3)

    # Conclusion marker
    if re.search(r'(?:therefore|thus|hence|so|in conclusion|the answer)', text_lower):
        score += 0.15

    return max(0.0, min(score, 1.0))


# ── Main test suite ─────────────────────────────────────────────────────

class ReasoningTestSuite:
    """Run reasoning quality tests across multiple languages.

    Tests chain-of-thought quality, step correctness, logical consistency,
    and cross-lingual reasoning equity for any model callable.

    Args:
        model_fn: Callable ``(prompt: str) -> str``.
        languages: ISO 639-1 codes to test.  Defaults to all target
            languages with reasoning fixtures.
        categories: Reasoning categories to test.  Defaults to all four:
            mathematical, causal, analogical, logical.
    """

    CATEGORIES = ("mathematical", "causal", "analogical", "logical")

    def __init__(
        self,
        model_fn: Callable[[str], str],
        languages: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> None:
        self.model_fn = model_fn
        self.categories = categories or list(self.CATEGORIES)
        # Languages determined at run time from available fixtures
        self._requested_languages = languages

    def run(
        self,
        scenarios: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> MultilingualReasoningResult:
        """Run reasoning tests across languages and categories.

        Args:
            scenarios: Nested dict ``category -> lang_code -> scenario``.
                Each scenario has keys: ``prompt``, ``expected_answer``,
                ``answer_type`` ("numeric" | "choice" | "open").
                If None, loads from fixtures.

        Returns:
            MultilingualReasoningResult with per-language, per-category
            scores and aggregate metrics.
        """
        from .fixtures import REASONING_SCENARIOS

        if scenarios is None:
            scenarios = REASONING_SCENARIOS

        languages = self._requested_languages
        result = MultilingualReasoningResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        all_scores: list[float] = []
        all_cot: list[float] = []
        all_logic: list[float] = []
        per_lang_scores: dict[str, list[float]] = {}

        for category in self.categories:
            cat_scenarios = scenarios.get(category, {})
            if not cat_scenarios:
                logger.warning("No scenarios for category %s", category)
                continue

            test_langs = languages or list(cat_scenarios.keys())

            for lang in test_langs:
                if lang not in cat_scenarios:
                    continue

                scenario = cat_scenarios[lang]
                lang_name = LANGUAGE_NAMES.get(lang, lang)

                t0 = time.perf_counter()
                try:
                    output = self.model_fn(scenario["prompt"])
                except Exception as exc:
                    logger.error("Model failed for %s/%s: %s", lang, category, exc)
                    r = ReasoningResult(
                        language=lang_name,
                        lang_code=lang,
                        category=category,
                        details={"error": str(exc)},
                    )
                    result.results.setdefault(lang, {})[category] = r
                    continue
                latency = time.perf_counter() - t0

                # Score the response
                r = self._score_response(
                    output=output,
                    scenario=scenario,
                    lang_code=lang,
                    lang_name=lang_name,
                    category=category,
                    latency=latency,
                )
                result.results.setdefault(lang, {})[category] = r

                all_scores.append(r.score)
                all_cot.append(r.cot_quality)
                all_logic.append(r.logical_consistency)
                per_lang_scores.setdefault(lang, []).append(r.score)

                logger.info(
                    "%s/%s: score=%.2f cot=%.2f logic=%.2f [%s]",
                    lang, category, r.score, r.cot_quality,
                    r.logical_consistency,
                    "CoT" if r.has_reasoning_steps else "no-CoT",
                )

        # Aggregate metrics
        if all_scores:
            result.mean_score = float(np.mean(all_scores))
            result.mean_cot_quality = float(np.mean(all_cot))
            result.mean_logical_consistency = float(np.mean(all_logic))

            # Per-language mean scores for equity calculation
            lang_means = [float(np.mean(v)) for v in per_lang_scores.values()]
            if len(lang_means) >= 2:
                result.cot_equity_score = float(np.var(lang_means))

            # Family-level aggregation
            result.family_scores = self._aggregate_by_family(per_lang_scores)

        return result

    def _score_response(
        self,
        output: str,
        scenario: dict[str, Any],
        lang_code: str,
        lang_name: str,
        category: str,
        latency: float,
    ) -> ReasoningResult:
        """Score a single reasoning response."""
        expected = scenario.get("expected_answer")
        answer_type = scenario.get("answer_type", "open")

        n_steps = _count_reasoning_steps(output)
        cot_quality = _score_cot_quality(output, expected)
        logic = _score_logical_consistency(output)

        # Check final answer
        final_correct = False
        if expected is not None:
            if answer_type == "numeric":
                extracted = _extract_numeric_answer(output)
                if extracted is not None:
                    final_correct = extracted == str(expected).replace(",", "")
            elif answer_type == "choice":
                extracted = _extract_choice_answer(output)
                if extracted is not None:
                    final_correct = extracted.upper() == str(expected).upper()
            else:
                # Open-ended: check if expected answer appears in output
                final_correct = str(expected).lower() in output.lower()

        # Composite score
        score = 0.0
        score += 0.35 * cot_quality
        score += 0.25 * logic
        score += 0.25 * (1.0 if final_correct else 0.0)
        score += 0.15 * (1.0 if n_steps >= 2 else 0.5 if n_steps >= 1 else 0.0)

        return ReasoningResult(
            language=lang_name,
            lang_code=lang_code,
            category=category,
            has_reasoning_steps=n_steps >= 1,
            n_steps_found=n_steps,
            final_answer_correct=final_correct,
            cot_quality=cot_quality,
            logical_consistency=logic,
            score=score,
            latency_seconds=latency,
            raw_output=output,
        )

    @staticmethod
    def _aggregate_by_family(
        per_lang_scores: dict[str, list[float]],
    ) -> dict[str, float]:
        """Compute mean reasoning score per language family."""
        from aya_distill.languages import LANGUAGES

        family_scores: dict[str, list[float]] = {}
        for lang, scores in per_lang_scores.items():
            if lang in LANGUAGES:
                fam = LANGUAGES[lang].family
                family_scores.setdefault(fam, []).extend(scores)

        return {
            fam: float(np.mean(vals))
            for fam, vals in family_scores.items()
        }

    def compare(
        self,
        teacher_fn: Callable[[str], str],
        scenarios: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compare student (self.model_fn) against teacher on reasoning.

        Returns per-language degradation = teacher_score - student_score.
        Positive means student is worse.
        """
        teacher_suite = ReasoningTestSuite(
            model_fn=teacher_fn,
            languages=self._requested_languages,
            categories=self.categories,
        )

        student_result = self.run(scenarios=scenarios)
        teacher_result = teacher_suite.run(scenarios=scenarios)

        degradation: dict[str, dict[str, float]] = {}
        for lang in student_result.results:
            s_cats = student_result.results.get(lang, {})
            t_cats = teacher_result.results.get(lang, {})
            for cat in self.categories:
                s = s_cats.get(cat)
                t = t_cats.get(cat)
                if s is None or t is None:
                    continue
                degradation.setdefault(lang, {})[f"{cat}_delta"] = round(
                    t.score - s.score, 4
                )
                degradation.setdefault(lang, {})[f"{cat}_cot_delta"] = round(
                    t.cot_quality - s.cot_quality, 4
                )

        return degradation
