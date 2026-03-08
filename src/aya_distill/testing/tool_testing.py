"""Tool calling tests for multilingual models.

Verifies that distilled models can produce valid structured tool calls
from prompts in any of the 10 target languages.  Integrates with
testLLM's ``ToolExpectations`` DSL when available.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolCallingResult:
    """Result of a single tool calling test."""

    language: str
    lang_code: str
    json_valid: bool = False
    schema_match: bool = False
    function_correct: bool = False
    args_correct: bool = False
    score: float = 0.0
    raw_output: str = ""
    parsed_call: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON reporting."""
        return {
            "language": self.language,
            "lang_code": self.lang_code,
            "json_valid": self.json_valid,
            "schema_match": self.schema_match,
            "function_correct": self.function_correct,
            "args_correct": self.args_correct,
            "score": self.score,
            "raw_output": self.raw_output[:300],
            "parsed_call": self.parsed_call,
        }


class ToolCallingTest:
    """Test structured tool calling across languages.

    Verifies that the model can:

    1. Generate valid JSON
    2. Select the correct function
    3. Extract correct arguments from multilingual prompts

    Optionally integrates with testLLM's ``expect_tools`` DSL for richer
    validation when the package is installed.

    Args:
        model_fn: Callable ``(prompt: str) -> str``.
        tool_schemas: List of function schemas the model should choose from.
            Used for documentation / future schema-aware validation.
    """

    def __init__(
        self,
        model_fn: Callable[[str], str],
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model_fn = model_fn
        self.tool_schemas = tool_schemas

    # ── JSON extraction ─────────────────────────────────────────────────
    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Try to extract a JSON object from model output.

        Handles:
        - Raw JSON
        - JSON inside markdown code fences
        - JSON embedded in surrounding prose
        """
        text = text.strip()

        # Direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Markdown code fence
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Greedy: find outermost braces
        # Walk forward to find balanced braces
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    # ── Single test ─────────────────────────────────────────────────────
    def test_single(
        self,
        prompt: str,
        expected_function: str,
        expected_args: dict[str, Any] | None = None,
        lang_code: str = "en",
        language: str = "English",
    ) -> ToolCallingResult:
        """Run a single tool calling test.

        Args:
            prompt: Full prompt including tool instructions.
            expected_function: Name of the function the model should call.
            expected_args: Expected argument key/value pairs (case-insensitive
                value comparison).
            lang_code: ISO 639-1 code for reporting.
            language: Human-readable language name.
        """
        result = ToolCallingResult(language=language, lang_code=lang_code)

        try:
            output = self.model_fn(prompt)
        except Exception as exc:
            logger.error("Model failed for %s: %s", lang_code, exc)
            result.raw_output = str(exc)
            return result

        result.raw_output = output
        parsed = self._extract_json(output)

        if parsed is None:
            logger.warning("%s: No valid JSON found in output", lang_code)
            return result

        result.json_valid = True
        result.parsed_call = parsed

        # Check function name (support multiple field names)
        fn_name = (
            parsed.get("function")
            or parsed.get("name")
            or parsed.get("tool")
            or parsed.get("tool_name")
        )
        if fn_name == expected_function:
            result.function_correct = True
            result.score += 0.5

        # Check arguments
        if expected_args is not None:
            call_args = (
                parsed.get("arguments")
                or parsed.get("args")
                or parsed.get("parameters")
                or {}
            )
            if isinstance(call_args, str):
                try:
                    call_args = json.loads(call_args)
                except json.JSONDecodeError:
                    call_args = {}

            total = max(len(expected_args), 1)
            matches = sum(
                1
                for k, v in expected_args.items()
                if str(call_args.get(k, "")).lower() == str(v).lower()
            )
            if matches == len(expected_args):
                result.args_correct = True
                result.score += 0.5
            else:
                result.score += 0.25 * (matches / total)

        result.schema_match = result.json_valid and result.function_correct
        return result

    # ── testLLM integration ─────────────────────────────────────────────
    def _try_testllm_verification(
        self,
        output: str,
        expected_function: str,
        expected_args: dict[str, Any] | None,
    ) -> bool | None:
        """Optionally verify via testLLM expect_tools DSL."""
        try:
            from testllm.tool_testing import expect_tools, AutoAdapter, ToolCall

            parsed = self._extract_json(output)
            if parsed is None:
                return False

            expectations = expect_tools().expect_call(expected_function)
            if expected_args:
                expectations = expectations.with_arguments_containing(
                    **expected_args,
                )

            tool_call = ToolCall(
                tool_name=expected_function,
                arguments=parsed.get("arguments") or parsed.get("args") or {},
            )
            summary = expectations.verify([tool_call])
            return summary.all_passed
        except ImportError:
            return None
        except Exception as exc:
            logger.debug("testLLM tool verification failed: %s", exc)
            return None

    # ── Multilingual batch ──────────────────────────────────────────────
    def run_multilingual(
        self,
        scenarios: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, ToolCallingResult]:
        """Run tool calling tests across all target languages.

        Args:
            scenarios: Dict of ``lang_code -> {prompt, expected_function,
                expected_args}``.  Uses ``TOOL_SCENARIOS`` from fixtures
                if not provided.

        Returns:
            Dict of ``lang_code -> ToolCallingResult``.
        """
        from .fixtures import TOOL_SCENARIOS
        from aya_distill.languages import LANGUAGE_NAMES

        scenarios = scenarios or TOOL_SCENARIOS
        results: dict[str, ToolCallingResult] = {}

        for lang_code, scenario in scenarios.items():
            lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

            result = self.test_single(
                prompt=scenario["prompt"],
                expected_function=scenario["expected_function"],
                expected_args=scenario.get("expected_args"),
                lang_code=lang_code,
                language=lang_name,
            )
            results[lang_code] = result

            logger.info(
                "%s: json=%s fn=%s args=%s score=%.2f",
                lang_code,
                result.json_valid,
                result.function_correct,
                result.args_correct,
                result.score,
            )

        return results

    def summary(
        self,
        results: dict[str, ToolCallingResult],
    ) -> str:
        """Generate a human-readable summary of tool calling results."""
        lines = ["Tool Calling Results", "=" * 50]
        total_score = 0.0
        n = 0
        for code, r in results.items():
            status_parts = []
            if r.json_valid:
                status_parts.append("JSON")
            if r.function_correct:
                status_parts.append("FN")
            if r.args_correct:
                status_parts.append("ARGS")
            status = "+".join(status_parts) if status_parts else "FAIL"
            lines.append(
                f"  {code} ({r.language:>10}): score={r.score:.2f} [{status}]"
            )
            total_score += r.score
            n += 1

        if n > 0:
            lines.append(f"\n  Mean score: {total_score / n:.3f}")
        return "\n".join(lines)
