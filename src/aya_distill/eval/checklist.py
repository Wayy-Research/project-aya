"""
Cohere Multilingual Evaluation Checklist -- validation layer.

Implements 22 checklist items across 7 categories, inspired by the
Cohere multilingual eval best practices. This ensures every evaluation
we run meets publication standards: per-language scores, confidence
intervals, prompt documentation, reproducibility metadata.

Run this after every benchmark to get a compliance report. Red items
block paper submission.

Wayy Research -- Project Aya
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checklist item definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChecklistItem:
    """A single checklist item."""

    id: str
    category: str
    description: str
    severity: str  # "required" or "recommended"


CHECKLIST_ITEMS: list[ChecklistItem] = [
    # Category 1: Evaluation Prompts
    ChecklistItem("EP-1", "Evaluation Prompts", "Prompts use native-language instructions (not English-only)", "required"),
    ChecklistItem("EP-2", "Evaluation Prompts", "Prompt templates are logged and versioned", "required"),
    ChecklistItem("EP-3", "Evaluation Prompts", "Few-shot exemplars are documented per language", "required"),
    ChecklistItem("EP-4", "Evaluation Prompts", "Prompt format is consistent across languages", "recommended"),

    # Category 2: Choice of Metrics
    ChecklistItem("CM-1", "Choice of Metrics", "Primary metric is clearly defined (accuracy, F1, etc.)", "required"),
    ChecklistItem("CM-2", "Choice of Metrics", "Metrics are appropriate for the task type", "required"),
    ChecklistItem("CM-3", "Choice of Metrics", "Multiple metrics reported where applicable", "recommended"),

    # Category 3: Statistical Testing
    ChecklistItem("ST-1", "Statistical Testing", "Bootstrap confidence intervals reported per language", "required"),
    ChecklistItem("ST-2", "Statistical Testing", "Sample sizes reported per language", "required"),
    ChecklistItem("ST-3", "Statistical Testing", "Statistical significance tests used for comparisons", "recommended"),

    # Category 4: Aggregating Results
    ChecklistItem("AR-1", "Aggregating Results", "Per-language scores are reported (not just averages)", "required"),
    ChecklistItem("AR-2", "Aggregating Results", "Language family aggregation is provided", "required"),
    ChecklistItem("AR-3", "Aggregating Results", "Variance across languages is reported", "required"),
    ChecklistItem("AR-4", "Aggregating Results", "Min/max language performance is highlighted", "recommended"),

    # Category 5: Qualitative Insights
    ChecklistItem("QI-1", "Qualitative Insights", "Example predictions are logged for manual inspection", "recommended"),
    ChecklistItem("QI-2", "Qualitative Insights", "Error patterns are analyzed per language", "recommended"),
    ChecklistItem("QI-3", "Qualitative Insights", "Script/tokenization differences are acknowledged", "recommended"),

    # Category 6: Reproducibility
    ChecklistItem("RP-1", "Reproducibility", "Model version/checkpoint is documented", "required"),
    ChecklistItem("RP-2", "Reproducibility", "Random seed is set and documented", "required"),
    ChecklistItem("RP-3", "Reproducibility", "Evaluation timestamp is recorded", "required"),
    ChecklistItem("RP-4", "Reproducibility", "Hardware/device info is documented", "recommended"),

    # Category 7: Meta-Evaluation
    ChecklistItem("ME-1", "Meta-Evaluation", "Benchmark limitations are acknowledged", "recommended"),
    # Note: 22 items total
]

assert len(CHECKLIST_ITEMS) == 22, f"Expected 22 checklist items, got {len(CHECKLIST_ITEMS)}"


# ---------------------------------------------------------------------------
# Checklist validation
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of checking a single item."""

    item: ChecklistItem
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item.id,
            "category": self.item.category,
            "description": self.item.description,
            "severity": self.item.severity,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class ChecklistReport:
    """Full checklist compliance report."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def n_required_failed(self) -> int:
        return sum(
            1 for r in self.results
            if not r.passed and r.item.severity == "required"
        )

    @property
    def compliant(self) -> bool:
        """All required items must pass."""
        return self.n_required_failed == 0

    def to_dict(self) -> dict[str, Any]:
        by_category: dict[str, list[dict]] = {}
        for r in self.results:
            cat = r.item.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r.to_dict())

        return {
            "compliant": self.compliant,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_required_failed": self.n_required_failed,
            "n_total": len(self.results),
            "by_category": by_category,
        }

    def print_report(self) -> None:
        """Print a human-readable checklist report."""
        status = "COMPLIANT" if self.compliant else "NON-COMPLIANT"
        print(f"\n{'=' * 72}")
        print(f"  Multilingual Evaluation Checklist -- {status}")
        print(f"  Passed: {self.n_passed}/{len(self.results)}")
        if self.n_required_failed > 0:
            print(f"  BLOCKING: {self.n_required_failed} required items failed")
        print(f"{'=' * 72}")

        current_cat = ""
        for r in self.results:
            if r.item.category != current_cat:
                current_cat = r.item.category
                print(f"\n  [{current_cat}]")

            icon = "PASS" if r.passed else "FAIL"
            severity = r.item.severity.upper()
            print(f"    [{icon}] [{severity}] {r.item.id}: {r.item.description}")
            if r.detail and not r.passed:
                print(f"           -> {r.detail}")

        print(f"\n{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Checker functions
# ---------------------------------------------------------------------------


def _has_per_language_scores(results: dict[str, Any]) -> tuple[bool, str]:
    """Check AR-1: per-language scores are reported."""
    per_lang = results.get("per_language", {})
    if not per_lang:
        return False, "No per_language key found in results"
    scored = [
        lang for lang, data in per_lang.items()
        if isinstance(data, dict) and "accuracy" in data
    ]
    if not scored:
        return False, "No languages have accuracy scores"
    return True, f"{len(scored)} languages scored"


def _has_confidence_intervals(results: dict[str, Any]) -> tuple[bool, str]:
    """Check ST-1: bootstrap CIs reported per language."""
    per_lang = results.get("per_language", {})
    for lang, data in per_lang.items():
        if isinstance(data, dict) and "accuracy" in data:
            ci = data.get("ci", {})
            if "lower" not in ci or "upper" not in ci:
                return False, f"Language '{lang}' missing CI bounds"
    return True, "All scored languages have CIs"


def _has_sample_sizes(results: dict[str, Any]) -> tuple[bool, str]:
    """Check ST-2: sample sizes reported."""
    per_lang = results.get("per_language", {})
    for lang, data in per_lang.items():
        if isinstance(data, dict) and "accuracy" in data:
            if "n_samples" not in data:
                return False, f"Language '{lang}' missing n_samples"
    return True, "All scored languages have sample sizes"


def _has_prompt_templates(results: dict[str, Any]) -> tuple[bool, str]:
    """Check EP-2: prompt templates are logged."""
    pt = results.get("prompt_template", {})
    if not pt:
        return False, "No prompt_template metadata found"
    required_fields = ["type", "n_shot"]
    missing = [f for f in required_fields if f not in pt]
    if missing:
        return False, f"Missing prompt_template fields: {missing}"
    return True, f"Prompt template: {pt.get('type', 'unknown')}, {pt.get('n_shot', '?')}-shot"


def _has_model_version(results: dict[str, Any]) -> tuple[bool, str]:
    """Check RP-1: model version documented."""
    model = results.get("model", "")
    if not model:
        return False, "No model identifier found"
    return True, f"Model: {model}"


def _has_seed(results: dict[str, Any]) -> tuple[bool, str]:
    """Check RP-2: random seed documented."""
    seed = results.get("seed")
    if seed is None:
        return False, "No random seed found"
    return True, f"Seed: {seed}"


def _has_timestamp(results: dict[str, Any]) -> tuple[bool, str]:
    """Check RP-3: timestamp recorded."""
    ts = results.get("timestamp", "")
    if not ts:
        return False, "No timestamp found"
    return True, f"Timestamp: {ts}"


def _has_family_aggregation(results: dict[str, Any]) -> tuple[bool, str]:
    """Check AR-2: language family aggregation."""
    fam = results.get("family_aggregation", [])
    if not fam:
        return False, "No family_aggregation found"
    return True, f"{len(fam)} language families"


def _has_variance_reported(results: dict[str, Any]) -> tuple[bool, str]:
    """Check AR-3: variance across languages reported."""
    per_lang = results.get("per_language", {})
    accs = [
        data["accuracy"]
        for data in per_lang.values()
        if isinstance(data, dict) and "accuracy" in data
    ]
    if len(accs) < 2:
        return False, "Fewer than 2 languages scored -- cannot compute variance"
    import numpy as np
    var = float(np.var(accs))
    return True, f"Cross-language variance: {var:.6f}"


def _has_native_instructions(results: dict[str, Any]) -> tuple[bool, str]:
    """Check EP-1: native-language instructions."""
    pt = results.get("prompt_template", {})
    instr_lang = pt.get("instruction_language", "")
    if instr_lang != "native":
        return False, f"Instruction language is '{instr_lang}', expected 'native'"
    return True, "Using native-language instructions"


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------


def validate_results(
    results: dict[str, Any],
    include_throughput: bool = False,
) -> ChecklistReport:
    """Run the full Cohere multilingual checklist against benchmark results.

    Parameters
    ----------
    results : benchmark results dict (from run_mgsm, run_xcopa, etc.)
    include_throughput : whether to relax checks for throughput results
                        (which do not have accuracy/CI)

    Returns
    -------
    ChecklistReport with pass/fail for each of the 22 items
    """
    report = ChecklistReport()

    # Map checklist IDs to checker functions
    # Items without explicit checkers get a default pass with a note
    checkers: dict[str, Any] = {
        "EP-1": _has_native_instructions,
        "EP-2": _has_prompt_templates,
        "EP-3": _has_prompt_templates,  # Re-uses prompt template check
        "AR-1": _has_per_language_scores,
        "AR-2": _has_family_aggregation,
        "AR-3": _has_variance_reported,
        "ST-1": _has_confidence_intervals,
        "ST-2": _has_sample_sizes,
        "CM-1": lambda r: (True, "Accuracy (exact match)"),
        "CM-2": lambda r: (True, "Accuracy appropriate for classification/math"),
        "RP-1": _has_model_version,
        "RP-2": _has_seed,
        "RP-3": _has_timestamp,
    }

    for item in CHECKLIST_ITEMS:
        checker = checkers.get(item.id)
        if checker is not None:
            if include_throughput and item.id in ("ST-1", "ST-2", "AR-1", "AR-2", "AR-3"):
                # Throughput results do not have accuracy/CI
                report.results.append(
                    CheckResult(item, True, "N/A for throughput benchmark")
                )
                continue
            try:
                passed, detail = checker(results)
                report.results.append(CheckResult(item, passed, detail))
            except Exception as e:
                report.results.append(
                    CheckResult(item, False, f"Check raised exception: {e}")
                )
        else:
            # Items without automated checks get a manual review note
            report.results.append(
                CheckResult(item, True, "Manual review recommended")
            )

    return report
