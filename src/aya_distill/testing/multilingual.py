"""Multilingual semantic testing for distilled models.

Wraps any model callable (teacher or student) and evaluates quality
across languages, measuring cross-lingual consistency, per-language
coherence/relevance, and equity of performance.

Integrates with testLLM's ``SemanticTest`` and ``LocalAgent`` when
available; falls back to heuristic scoring otherwise.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Target languages (single source of truth) ──────────────────────────────
from aya_distill.languages import LANGUAGE_NAMES

# Scripts that use Latin characters
_LATIN_LANGS: frozenset[str] = frozenset({"en", "es", "sw", "tr", "id"})


# ── Result dataclasses ──────────────────────────────────────────────────────
@dataclass
class LanguageTestResult:
    """Test results for a single language."""

    language: str
    lang_code: str
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    tool_accuracy: float = 0.0
    latency_seconds: float = 0.0
    token_count: int = 0
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultilingualTestResult:
    """Aggregated results across all languages."""

    language_results: dict[str, LanguageTestResult] = field(default_factory=dict)
    cross_lingual_consistency: float = 0.0
    equity_score: float = 0.0  # Lower is better (variance across languages)
    mean_coherence: float = 0.0
    mean_relevance: float = 0.0
    timestamp: str = ""

    def summary(self) -> str:
        """Human-readable summary table."""
        lines = ["Multilingual Test Results", "=" * 50]
        for code, r in self.language_results.items():
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  {code} ({r.language:>10}): coherence={r.coherence_score:.2f} "
                f"relevance={r.relevance_score:.2f} latency={r.latency_seconds:.3f}s "
                f"[{status}]"
            )
        lines.append("")
        lines.append(f"  Mean coherence:              {self.mean_coherence:.3f}")
        lines.append(f"  Mean relevance:              {self.mean_relevance:.3f}")
        lines.append(
            f"  Cross-lingual consistency:   {self.cross_lingual_consistency:.3f}"
        )
        lines.append(
            f"  Equity score (lower=better): {self.equity_score:.4f}"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for JSON export."""
        return {
            "timestamp": self.timestamp,
            "mean_coherence": self.mean_coherence,
            "mean_relevance": self.mean_relevance,
            "cross_lingual_consistency": self.cross_lingual_consistency,
            "equity_score": self.equity_score,
            "languages": {
                code: {
                    "language": r.language,
                    "coherence": r.coherence_score,
                    "relevance": r.relevance_score,
                    "tool_accuracy": r.tool_accuracy,
                    "latency_seconds": r.latency_seconds,
                    "token_count": r.token_count,
                    "passed": r.passed,
                }
                for code, r in self.language_results.items()
            },
        }


# ── testLLM integration helpers ────────────────────────────────────────────
def _try_testllm_semantic_score(
    prompt: str,
    response: str,
    lang: str,
    judge_model: str | None,
) -> dict[str, float] | None:
    """Attempt to score with testLLM SemanticTest. Returns None on failure."""
    if judge_model is None:
        return None
    try:
        from testllm import SemanticTest, LocalAgent

        agent = LocalAgent(model=lambda _p: response)
        test = SemanticTest(
            test_id=f"multilingual_{lang}",
            description=f"Quality check for {lang}",
            evaluator_models=[judge_model],
            consensus_threshold=0.5,
        )
        test.add_scenario(
            user_input=prompt,
            criteria=[
                "The response is coherent and well-structured",
                "The response is relevant to the question asked",
                f"The response is in the correct language ({LANGUAGE_NAMES.get(lang, lang)})",
            ],
        )
        import asyncio

        results = asyncio.run(test.execute(agent))
        if results and results[0].criterion_results:
            cr = results[0].criterion_results
            coherence = cr[0].get("consensus_score", 0.5) if len(cr) > 0 else 0.5
            relevance = cr[1].get("consensus_score", 0.5) if len(cr) > 1 else 0.5
            return {"coherence": float(coherence), "relevance": float(relevance)}
    except Exception as exc:
        logger.debug("testLLM scoring failed for %s: %s", lang, exc)
    return None


# ── Heuristic scorer ────────────────────────────────────────────────────────
def _heuristic_score(prompt: str, response: str, lang: str) -> dict[str, float]:
    """Basic heuristic scoring when no LLM judge is available.

    Checks response length, script consistency, and basic structure.
    """
    scores: dict[str, float] = {}
    resp_len = len(response.strip())

    # Coherence: reward reasonable-length, structured responses
    if resp_len < 10:
        scores["coherence"] = 0.1
    elif resp_len < 50:
        scores["coherence"] = 0.4
    elif resp_len < 2000:
        scores["coherence"] = 0.8
    else:
        scores["coherence"] = 0.7  # Very long responses lose some signal

    # Relevance: check script consistency between prompt and response
    prompt_has_nonlatin = any(ord(c) > 127 for c in prompt)
    resp_has_nonlatin = any(ord(c) > 127 for c in response)

    if lang in _LATIN_LANGS:
        scores["relevance"] = 0.7 if resp_len > 20 else 0.3
    elif prompt_has_nonlatin and resp_has_nonlatin:
        scores["relevance"] = 0.7
    elif prompt_has_nonlatin and not resp_has_nonlatin:
        # Model responded in Latin script for a non-Latin prompt
        scores["relevance"] = 0.3
    else:
        scores["relevance"] = 0.5

    return scores


# ── Main test suite ─────────────────────────────────────────────────────────
class MultilingualTestSuite:
    """Run semantic quality tests across multiple languages.

    Wraps any ``model_fn: (str) -> str`` callable and measures coherence,
    relevance, cross-lingual consistency, and equity across target languages.

    Args:
        model_fn: Callable that takes ``(prompt: str)`` and returns generated
            text.  Works with both teacher and student models.
        languages: List of ISO 639-1 codes to test.  Defaults to all 10
            target languages.
        judge_model: Optional LLM model name for testLLM semantic scoring.
            If ``None`` (default), uses heuristic scoring.
        thresholds: Dict of ``{metric: minimum_score}`` used to decide
            per-language pass/fail.
    """

    def __init__(
        self,
        model_fn: Callable[[str], str],
        languages: list[str] | None = None,
        judge_model: str | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.model_fn = model_fn
        self.languages = languages or list(LANGUAGE_NAMES.keys())
        self.judge_model = judge_model
        self.thresholds = thresholds or {
            "coherence": 0.5,
            "relevance": 0.5,
        }

    # ── Public API ──────────────────────────────────────────────────────
    def run(
        self,
        prompts: dict[str, str] | None = None,
        prompt_category: str = "general_knowledge",
    ) -> MultilingualTestResult:
        """Run the full test suite.

        Args:
            prompts: Dict of ``lang_code -> prompt``.  If ``None``, uses
                the fixtures for *prompt_category*.
            prompt_category: Key in ``MULTILINGUAL_PROMPTS`` to use when
                *prompts* is not provided.

        Returns:
            ``MultilingualTestResult`` with per-language scores and
            aggregate metrics.
        """
        from .fixtures import MULTILINGUAL_PROMPTS

        if prompts is None:
            prompts = MULTILINGUAL_PROMPTS.get(prompt_category, {})

        result = MultilingualTestResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        responses: dict[str, str] = {}

        for lang in self.languages:
            if lang not in prompts:
                logger.warning("No prompt for language %s, skipping", lang)
                continue

            prompt = prompts[lang]
            lang_name = LANGUAGE_NAMES.get(lang, lang)

            # Generate response and measure latency
            t0 = time.perf_counter()
            try:
                response = self.model_fn(prompt)
            except Exception as exc:
                logger.error("Model failed for %s: %s", lang, exc)
                result.language_results[lang] = LanguageTestResult(
                    language=lang_name,
                    lang_code=lang,
                    passed=False,
                    details={"error": str(exc)},
                )
                continue
            latency = time.perf_counter() - t0
            responses[lang] = response

            # Score the response -- try testLLM judge first, fall back to heuristic
            scores = _try_testllm_semantic_score(
                prompt, response, lang, self.judge_model,
            )
            if scores is None:
                scores = _heuristic_score(prompt, response, lang)

            passed = all(
                scores.get(k, 0.0) >= v for k, v in self.thresholds.items()
            )

            result.language_results[lang] = LanguageTestResult(
                language=lang_name,
                lang_code=lang,
                coherence_score=scores.get("coherence", 0.0),
                relevance_score=scores.get("relevance", 0.0),
                latency_seconds=round(latency, 4),
                token_count=len(response.split()),
                passed=passed,
                details={"response_preview": response[:200]},
            )

            logger.info(
                "%s (%s): coherence=%.2f relevance=%.2f latency=%.3fs [%s]",
                lang,
                lang_name,
                scores["coherence"],
                scores["relevance"],
                latency,
                "PASS" if passed else "FAIL",
            )

        # ── Aggregate metrics ───────────────────────────────────────────
        coherence_vals = [
            r.coherence_score for r in result.language_results.values()
        ]
        relevance_vals = [
            r.relevance_score for r in result.language_results.values()
        ]

        if coherence_vals:
            result.mean_coherence = float(np.mean(coherence_vals))
            result.mean_relevance = float(np.mean(relevance_vals))
            # Consistency: 1 - stdev (ranges 0..1 where 1 = perfectly uniform)
            result.cross_lingual_consistency = 1.0 - float(
                np.std(coherence_vals)
            )
            # Equity: variance of coherence across languages (lower = fairer)
            result.equity_score = float(np.var(coherence_vals))

        return result

    def compare(
        self,
        teacher_fn: Callable[[str], str],
        prompts: dict[str, str] | None = None,
        prompt_category: str = "general_knowledge",
    ) -> dict[str, dict[str, float]]:
        """Compare student (``self.model_fn``) against a teacher.

        Returns per-language degradation = teacher_score - student_score.
        Positive values mean the student is worse.
        """
        from .fixtures import MULTILINGUAL_PROMPTS

        if prompts is None:
            prompts = MULTILINGUAL_PROMPTS.get(prompt_category, {})

        teacher_suite = MultilingualTestSuite(
            model_fn=teacher_fn,
            languages=self.languages,
            judge_model=self.judge_model,
            thresholds=self.thresholds,
        )

        student_result = self.run(prompts=prompts)
        teacher_result = teacher_suite.run(prompts=prompts)

        degradation: dict[str, dict[str, float]] = {}
        for lang in self.languages:
            s = student_result.language_results.get(lang)
            t = teacher_result.language_results.get(lang)
            if s is None or t is None:
                continue
            degradation[lang] = {
                "coherence_delta": round(t.coherence_score - s.coherence_score, 4),
                "relevance_delta": round(t.relevance_score - s.relevance_score, 4),
                "latency_delta": round(t.latency_seconds - s.latency_seconds, 4),
            }

        return degradation
