"""
Cohere Multilingual Evaluation Checklist -- validation layer.

Implements the 25-item checklist across 7 categories from the "Deja Vu"
paper (arXiv:2504.11829). This ensures every multilingual LLM evaluation
meets publication standards: per-language scores, confidence intervals,
prompt documentation, reproducibility metadata.

Run this after every benchmark to get a compliance report. Required items
that fail block paper submission.

Usage::

    from aya_distill.eval.checklist import validate_results, CHECKLIST_ITEMS

    report = validate_results(my_eval_results)
    report.print_report()
    assert report.compliant, f"Failed: {report.n_required_failed} required items"

Reference
---------
Singh et al., "Deja Vu: Multilingual LLM Evaluation Checklist",
arXiv:2504.11829, 2025.

Wayy Research -- Project Aya
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from aya_distill.languages import LANGUAGES, LANGUAGE_FAMILIES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class CheckStatus(str, Enum):
    """Outcome of a single checklist item validation."""

    PASS = "pass"
    FAIL = "fail"
    MANUAL_CHECK_NEEDED = "manual_check_needed"

    def __bool__(self) -> bool:
        """PASS is truthy; FAIL is falsy; MANUAL_CHECK_NEEDED is truthy.

        Manual-check items do not block compliance -- they surface as
        warnings in the report so a human reviewer picks them up.
        """
        return self is not CheckStatus.FAIL


# ---------------------------------------------------------------------------
# Checklist item definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChecklistItem:
    """A single checklist item from the Deja Vu paper."""

    id: str
    category: str
    description: str
    severity: str  # "required" or "recommended"

    @property
    def required(self) -> bool:
        return self.severity == "required"


# All 25 items across 7 categories, exactly matching the paper.
CHECKLIST_ITEMS: list[ChecklistItem] = [
    # -- Category 1: Evaluation Prompts (4 items) --
    ChecklistItem(
        "EP-1", "Evaluation Prompts",
        "Native-language evaluation prompts (not just translations)",
        "required",
    ),
    ChecklistItem(
        "EP-2", "Evaluation Prompts",
        "Prompt templates logged and versioned",
        "required",
    ),
    ChecklistItem(
        "EP-3", "Evaluation Prompts",
        "Few-shot exemplars provided per language",
        "required",
    ),
    ChecklistItem(
        "EP-4", "Evaluation Prompts",
        "Format consistency across languages",
        "recommended",
    ),

    # -- Category 2: Choice of Metrics (3 items) --
    ChecklistItem(
        "CM-1", "Choice of Metrics",
        "Primary metric clearly defined",
        "required",
    ),
    ChecklistItem(
        "CM-2", "Choice of Metrics",
        "Metrics appropriate for all evaluated languages",
        "required",
    ),
    ChecklistItem(
        "CM-3", "Choice of Metrics",
        "Multiple metrics reported (both automatic and human if possible)",
        "recommended",
    ),

    # -- Category 3: Statistical Testing (4 items) --
    ChecklistItem(
        "ST-1", "Statistical Testing",
        "Bootstrap confidence intervals per language (95% CI, n>=1000)",
        "required",
    ),
    ChecklistItem(
        "ST-2", "Statistical Testing",
        "Adequate sample sizes per language documented",
        "required",
    ),
    ChecklistItem(
        "ST-3", "Statistical Testing",
        "Statistical significance tests between systems",
        "recommended",
    ),
    ChecklistItem(
        "ST-4", "Statistical Testing",
        "Power analysis or minimum detectable effect reported",
        "recommended",
    ),

    # -- Category 4: Aggregating Results (4 items) --
    ChecklistItem(
        "AR-1", "Aggregating Results",
        "Per-language scores always reported (never hidden behind averages)",
        "required",
    ),
    ChecklistItem(
        "AR-2", "Aggregating Results",
        "Language family aggregation provided",
        "required",
    ),
    ChecklistItem(
        "AR-3", "Aggregating Results",
        "Variance/spread across languages documented",
        "required",
    ),
    ChecklistItem(
        "AR-4", "Aggregating Results",
        "Results grouped by resourcedness (high/medium/low)",
        "recommended",
    ),

    # -- Category 5: Qualitative Insights (3 items) --
    ChecklistItem(
        "QI-1", "Qualitative Insights",
        "Example predictions logged per language",
        "recommended",
    ),
    ChecklistItem(
        "QI-2", "Qualitative Insights",
        "Error pattern analysis per language",
        "recommended",
    ),
    ChecklistItem(
        "QI-3", "Qualitative Insights",
        "Script/tokenization efficiency differences documented",
        "recommended",
    ),

    # -- Category 6: Reproducibility (5 items) --
    ChecklistItem(
        "RP-1", "Reproducibility",
        "Model version/commit hash documented",
        "required",
    ),
    ChecklistItem(
        "RP-2", "Reproducibility",
        "Random seed set and documented",
        "required",
    ),
    ChecklistItem(
        "RP-3", "Reproducibility",
        "Evaluation timestamp recorded",
        "required",
    ),
    ChecklistItem(
        "RP-4", "Reproducibility",
        "Hardware and device info recorded",
        "recommended",
    ),
    ChecklistItem(
        "RP-5", "Reproducibility",
        "Decoding parameters documented (temperature, top_p, etc.)",
        "recommended",
    ),

    # -- Category 7: Meta-Evaluation (2 items) --
    ChecklistItem(
        "ME-1", "Meta-Evaluation",
        "Benchmark limitations acknowledged",
        "recommended",
    ),
    ChecklistItem(
        "ME-2", "Meta-Evaluation",
        "Metric correlation with human judgment documented",
        "recommended",
    ),
]

# Quick-access index by item ID.
CHECKLIST_INDEX: dict[str, ChecklistItem] = {item.id: item for item in CHECKLIST_ITEMS}

# Ordered unique category names, preserving definition order.
CATEGORY_NAMES: list[str] = list(dict.fromkeys(item.category for item in CHECKLIST_ITEMS))

assert len(CHECKLIST_ITEMS) == 25, (
    f"Expected 25 checklist items (Deja Vu paper), got {len(CHECKLIST_ITEMS)}"
)


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of validating a single checklist item."""

    item: ChecklistItem
    status: CheckStatus
    detail: str = ""

    @property
    def passed(self) -> bool:
        """True when status is PASS or MANUAL_CHECK_NEEDED."""
        return bool(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item.id,
            "category": self.item.category,
            "description": self.item.description,
            "severity": self.item.severity,
            "status": self.status.value,
            "passed": self.passed,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Category summary
# ---------------------------------------------------------------------------

@dataclass
class CategorySummary:
    """Aggregated pass/fail counts for a single category."""

    name: str
    n_total: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_manual: int = 0
    n_required_failed: int = 0

    @property
    def complete(self) -> bool:
        return self.n_required_failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_total": self.n_total,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_manual": self.n_manual,
            "n_required_failed": self.n_required_failed,
            "complete": self.complete,
        }


# ---------------------------------------------------------------------------
# Checklist report
# ---------------------------------------------------------------------------

@dataclass
class ChecklistReport:
    """Full checklist compliance report with category-level summaries."""

    results: list[CheckResult] = field(default_factory=list)

    # -- Counts --

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.status is CheckStatus.PASS)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if r.status is CheckStatus.FAIL)

    @property
    def n_manual(self) -> int:
        return sum(1 for r in self.results if r.status is CheckStatus.MANUAL_CHECK_NEEDED)

    @property
    def n_required_failed(self) -> int:
        return sum(
            1 for r in self.results
            if r.status is CheckStatus.FAIL and r.item.required
        )

    @property
    def compliant(self) -> bool:
        """True when every *required* item passes (or is manual-check)."""
        return self.n_required_failed == 0

    @property
    def score(self) -> float:
        """Fraction of items passing (PASS or MANUAL_CHECK_NEEDED count)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    # -- Category helpers --

    @property
    def category_summaries(self) -> list[CategorySummary]:
        """Build per-category summaries preserving definition order."""
        buckets: dict[str, CategorySummary] = {}
        for r in self.results:
            cat = r.item.category
            if cat not in buckets:
                buckets[cat] = CategorySummary(name=cat)
            s = buckets[cat]
            s.n_total += 1
            if r.status is CheckStatus.PASS:
                s.n_passed += 1
            elif r.status is CheckStatus.FAIL:
                s.n_failed += 1
                if r.item.required:
                    s.n_required_failed += 1
            elif r.status is CheckStatus.MANUAL_CHECK_NEEDED:
                s.n_manual += 1
        # Return in definition order.
        return [buckets[c] for c in CATEGORY_NAMES if c in buckets]

    # -- Serialization --

    def to_dict(self) -> dict[str, Any]:
        by_category: dict[str, list[dict[str, Any]]] = {}
        for r in self.results:
            by_category.setdefault(r.item.category, []).append(r.to_dict())

        return {
            "compliant": self.compliant,
            "score": round(self.score, 4),
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_manual": self.n_manual,
            "n_required_failed": self.n_required_failed,
            "n_total": self.n_total,
            "categories": [s.to_dict() for s in self.category_summaries],
            "items": by_category,
        }

    # -- Pretty printing --

    def print_report(self) -> None:
        """Print a human-readable checklist report to stdout."""
        status_label = "COMPLIANT" if self.compliant else "NON-COMPLIANT"
        pct = self.score * 100

        print(f"\n{'=' * 74}")
        print(f"  Deja Vu Multilingual Evaluation Checklist -- {status_label}")
        print(f"  Score: {self.n_passed} pass / {self.n_manual} manual / "
              f"{self.n_failed} fail  ({pct:.0f}% of {self.n_total} items)")
        if self.n_required_failed > 0:
            print(f"  BLOCKING: {self.n_required_failed} required item(s) failed")
        print(f"{'=' * 74}")

        for summary in self.category_summaries:
            cat_status = "OK" if summary.complete else "BLOCKED"
            print(f"\n  [{summary.name}]  ({cat_status})")

            for r in self.results:
                if r.item.category != summary.name:
                    continue
                icon = _status_icon(r.status)
                sev = r.item.severity.upper()[:3]
                print(f"    [{icon}] [{sev}] {r.item.id}: {r.item.description}")
                if r.detail:
                    print(f"           {r.detail}")

        print(f"\n{'=' * 74}")
        print(f"  REQ = required (blocks compliance)   REC = recommended")
        print(f"{'=' * 74}\n")


def _status_icon(status: CheckStatus) -> str:
    if status is CheckStatus.PASS:
        return "PASS"
    if status is CheckStatus.FAIL:
        return "FAIL"
    return " ?? "


# ---------------------------------------------------------------------------
# Individual checker functions
# ---------------------------------------------------------------------------
# Each returns (CheckStatus, detail_string).
# Convention: the results dict is the only argument.

_CheckerFn = Callable[[dict[str, Any]], tuple[CheckStatus, str]]


def _check_native_instructions(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """EP-1: Native-language evaluation prompts."""
    pt = results.get("prompt_template") or results.get("prompt_templates")
    if not pt:
        return CheckStatus.FAIL, "No prompt_template(s) metadata found"
    if isinstance(pt, dict):
        lang_flag = pt.get("instruction_language", "")
        if lang_flag == "native":
            return CheckStatus.PASS, "Using native-language instructions"
        # Check if per-language templates are provided.
        per_lang = pt.get("per_language", {})
        if per_lang and len(per_lang) > 1:
            return CheckStatus.PASS, f"Per-language templates for {len(per_lang)} languages"
        if lang_flag:
            return CheckStatus.FAIL, f"instruction_language='{lang_flag}', expected 'native'"
    return CheckStatus.FAIL, "Cannot confirm native-language instructions"


def _check_prompt_templates_logged(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """EP-2: Prompt templates logged and versioned."""
    pt = results.get("prompt_template") or results.get("prompt_templates")
    if not pt:
        return CheckStatus.FAIL, "No prompt_template(s) metadata found"
    if isinstance(pt, dict):
        # We expect at minimum a 'type' or 'template' field.
        has_type = "type" in pt or "template" in pt or "name" in pt
        has_version = "version" in pt or "commit" in pt or "hash" in pt
        if has_type:
            version_note = ""
            if has_version:
                v = pt.get("version") or pt.get("commit") or pt.get("hash")
                version_note = f", version={v}"
            return CheckStatus.PASS, (
                f"Template: {pt.get('type', pt.get('name', pt.get('template', '?')))}"
                f", n_shot={pt.get('n_shot', '?')}{version_note}"
            )
    return CheckStatus.FAIL, "prompt_template missing 'type'/'template'/'name' field"


def _check_fewshot_exemplars(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """EP-3: Few-shot exemplars provided per language."""
    pt = results.get("prompt_template") or results.get("prompt_templates")
    if not pt or not isinstance(pt, dict):
        return CheckStatus.FAIL, "No prompt_template metadata found"
    n_shot = pt.get("n_shot")
    if n_shot is not None and int(n_shot) > 0:
        exemplars = pt.get("exemplars") or pt.get("per_language")
        if exemplars and isinstance(exemplars, dict):
            return CheckStatus.PASS, f"{n_shot}-shot with exemplars for {len(exemplars)} languages"
        return CheckStatus.PASS, f"{n_shot}-shot (exemplar details not embedded in results)"
    if n_shot == 0:
        return CheckStatus.PASS, "0-shot evaluation (no exemplars needed)"
    return CheckStatus.FAIL, "n_shot not documented in prompt_template"


def _check_format_consistency(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """EP-4: Format consistency across languages."""
    return CheckStatus.MANUAL_CHECK_NEEDED, "Verify prompt format is consistent across languages"


def _check_primary_metric(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """CM-1: Primary metric clearly defined."""
    metric = results.get("primary_metric") or results.get("metric")
    if metric:
        return CheckStatus.PASS, f"Primary metric: {metric}"
    # Infer from per_language data.
    per_lang = results.get("per_language", {})
    if per_lang:
        sample = next(iter(per_lang.values()), {})
        if isinstance(sample, dict):
            candidates = [k for k in sample if k not in (
                "ci", "n_samples", "examples", "errors", "language",
                "family", "resourcedness",
            )]
            if candidates:
                return CheckStatus.PASS, f"Inferred metric(s): {', '.join(candidates)}"
    return CheckStatus.FAIL, "No primary_metric key and cannot infer from per_language data"


def _check_metrics_appropriate(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """CM-2: Metrics appropriate for all evaluated languages."""
    metric = results.get("primary_metric") or results.get("metric")
    if metric:
        return CheckStatus.PASS, f"Metric '{metric}' applied uniformly"
    return CheckStatus.MANUAL_CHECK_NEEDED, "Verify metric suitability for all languages"


def _check_multiple_metrics(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """CM-3: Multiple metrics reported."""
    metrics = results.get("metrics")
    if isinstance(metrics, (list, tuple)) and len(metrics) > 1:
        return CheckStatus.PASS, f"Metrics: {', '.join(str(m) for m in metrics)}"
    per_lang = results.get("per_language", {})
    if per_lang:
        sample = next(iter(per_lang.values()), {})
        if isinstance(sample, dict):
            metric_keys = [k for k in sample if k not in (
                "ci", "n_samples", "examples", "errors", "language",
                "family", "resourcedness",
            )]
            if len(metric_keys) > 1:
                return CheckStatus.PASS, f"Multiple metrics found: {', '.join(metric_keys)}"
    return CheckStatus.MANUAL_CHECK_NEEDED, "Only one metric detected; consider adding more"


def _check_confidence_intervals(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """ST-1: Bootstrap confidence intervals per language."""
    per_lang = results.get("per_language", {})
    if not per_lang:
        return CheckStatus.FAIL, "No per_language data found"
    missing: list[str] = []
    checked = 0
    for lang, data in per_lang.items():
        if not isinstance(data, dict):
            continue
        checked += 1
        ci = data.get("ci") or data.get("confidence_interval")
        if not ci or not isinstance(ci, dict):
            missing.append(lang)
            continue
        has_bounds = ("lower" in ci and "upper" in ci) or ("low" in ci and "high" in ci)
        if not has_bounds:
            missing.append(lang)
    if checked == 0:
        return CheckStatus.FAIL, "No language results with scores found"
    if missing:
        return CheckStatus.FAIL, f"Missing CIs for: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
    return CheckStatus.PASS, f"All {checked} languages have confidence intervals"


def _check_sample_sizes(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """ST-2: Adequate sample sizes per language documented."""
    per_lang = results.get("per_language", {})
    if not per_lang:
        return CheckStatus.FAIL, "No per_language data found"
    missing: list[str] = []
    sizes: list[int] = []
    for lang, data in per_lang.items():
        if not isinstance(data, dict):
            continue
        n = data.get("n_samples") or data.get("n") or data.get("sample_size")
        if n is None:
            missing.append(lang)
        else:
            sizes.append(int(n))
    if missing:
        return CheckStatus.FAIL, f"Missing sample sizes for: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
    if sizes:
        return CheckStatus.PASS, f"Sample sizes: min={min(sizes)}, max={max(sizes)}, n_langs={len(sizes)}"
    return CheckStatus.FAIL, "No sample sizes found"


def _check_significance_tests(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """ST-3: Statistical significance tests between systems."""
    sig = results.get("significance_tests") or results.get("paired_tests")
    if sig:
        return CheckStatus.PASS, f"Significance tests present ({len(sig)} comparisons)"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No significance_tests key; run paired bootstrap if comparing systems"


def _check_power_analysis(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """ST-4: Power analysis or minimum detectable effect reported."""
    pa = results.get("power_analysis") or results.get("min_detectable_effect")
    if pa:
        return CheckStatus.PASS, f"Power analysis: {pa}"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No power_analysis found; consider reporting MDE"


def _check_per_language_scores(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """AR-1: Per-language scores always reported."""
    per_lang = results.get("per_language", {})
    if not per_lang:
        return CheckStatus.FAIL, "No per_language key found in results"
    scored: list[str] = []
    for lang, data in per_lang.items():
        if isinstance(data, dict) and len(data) > 0:
            scored.append(lang)
        elif isinstance(data, (int, float)):
            scored.append(lang)
    if not scored:
        return CheckStatus.FAIL, "per_language present but no languages have scores"
    return CheckStatus.PASS, f"{len(scored)} languages with individual scores"


def _check_family_aggregation(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """AR-2: Language family aggregation provided."""
    fam = results.get("family_aggregation") or results.get("by_family")
    if fam:
        n = len(fam) if isinstance(fam, (list, dict)) else 0
        return CheckStatus.PASS, f"{n} language families reported"
    # Try to infer: if per_language exists, check if we can derive families.
    per_lang = results.get("per_language", {})
    known_langs = set(per_lang.keys()) & set(LANGUAGES.keys())
    if known_langs:
        families_present = {LANGUAGES[c].family for c in known_langs}
        if len(families_present) > 1:
            return CheckStatus.FAIL, (
                f"Per-language data covers {len(families_present)} families "
                f"but no family_aggregation key found"
            )
    return CheckStatus.FAIL, "No family_aggregation or by_family key found"


def _check_variance_reported(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """AR-3: Variance/spread across languages documented."""
    spread = results.get("variance") or results.get("spread") or results.get("std")
    if spread is not None:
        return CheckStatus.PASS, f"Reported spread: {spread}"
    # Compute from per_language if possible.
    per_lang = results.get("per_language", {})
    values: list[float] = []
    for data in per_lang.values():
        if isinstance(data, dict):
            # Find any numeric metric.
            for key in ("accuracy", "f1", "bleu", "score", "exact_match"):
                if key in data:
                    values.append(float(data[key]))
                    break
        elif isinstance(data, (int, float)):
            values.append(float(data))
    if len(values) >= 2:
        std = statistics.stdev(values)
        mn, mx = min(values), max(values)
        return CheckStatus.PASS, (
            f"Computed from per_language: std={std:.4f}, "
            f"range=[{mn:.4f}, {mx:.4f}]"
        )
    return CheckStatus.FAIL, "Fewer than 2 scored languages; cannot assess variance"


def _check_resourcedness_groups(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """AR-4: Results grouped by resourcedness (high/medium/low)."""
    rg = results.get("by_resourcedness") or results.get("resourcedness_groups")
    if rg:
        return CheckStatus.PASS, f"Resourcedness groups: {list(rg.keys()) if isinstance(rg, dict) else rg}"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No by_resourcedness key; consider grouping high/med/low resource"


def _check_example_predictions(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """QI-1: Example predictions logged per language."""
    per_lang = results.get("per_language", {})
    langs_with_examples: list[str] = []
    for lang, data in per_lang.items():
        if isinstance(data, dict) and ("examples" in data or "predictions" in data):
            langs_with_examples.append(lang)
    examples_top = results.get("examples") or results.get("predictions")
    if langs_with_examples:
        return CheckStatus.PASS, f"Examples logged for {len(langs_with_examples)} languages"
    if examples_top:
        return CheckStatus.PASS, "Top-level examples/predictions found"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No example predictions found in results"


def _check_error_patterns(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """QI-2: Error pattern analysis per language."""
    per_lang = results.get("per_language", {})
    langs_with_errors: list[str] = []
    for lang, data in per_lang.items():
        if isinstance(data, dict) and ("errors" in data or "error_analysis" in data):
            langs_with_errors.append(lang)
    ea = results.get("error_analysis")
    if langs_with_errors:
        return CheckStatus.PASS, f"Error analysis for {len(langs_with_errors)} languages"
    if ea:
        return CheckStatus.PASS, "Top-level error_analysis found"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No error analysis found in results"


def _check_script_tokenization(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """QI-3: Script/tokenization efficiency differences documented."""
    tok = results.get("tokenization") or results.get("token_stats")
    if tok:
        return CheckStatus.PASS, f"Tokenization info present"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No tokenization/token_stats key; consider documenting"


def _check_model_version(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """RP-1: Model version/commit hash documented."""
    model = results.get("model") or results.get("model_name") or results.get("model_id")
    commit = results.get("commit") or results.get("model_commit")
    if model:
        detail = f"Model: {model}"
        if commit:
            detail += f" (commit: {commit})"
        return CheckStatus.PASS, detail
    return CheckStatus.FAIL, "No model/model_name/model_id found in results"


def _check_random_seed(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """RP-2: Random seed set and documented."""
    seed = results.get("seed") or results.get("random_seed")
    if seed is not None:
        return CheckStatus.PASS, f"Seed: {seed}"
    return CheckStatus.FAIL, "No seed or random_seed found in results"


def _check_timestamp(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """RP-3: Evaluation timestamp recorded."""
    ts = results.get("timestamp") or results.get("eval_timestamp") or results.get("date")
    if ts:
        return CheckStatus.PASS, f"Timestamp: {ts}"
    return CheckStatus.FAIL, "No timestamp/eval_timestamp/date found in results"


def _check_hardware_info(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """RP-4: Hardware and device info recorded."""
    hw = results.get("hardware") or results.get("device") or results.get("gpu")
    if hw:
        return CheckStatus.PASS, f"Hardware: {hw}"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No hardware/device info found; consider documenting"


def _check_decoding_params(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """RP-5: Decoding parameters documented."""
    dp = results.get("decoding") or results.get("generation_config") or results.get("decoding_params")
    if dp and isinstance(dp, dict):
        keys = list(dp.keys())[:5]
        return CheckStatus.PASS, f"Decoding params: {', '.join(keys)}"
    # Check for individual keys.
    for key in ("temperature", "top_p", "top_k", "max_tokens", "max_new_tokens"):
        if key in results:
            return CheckStatus.PASS, f"Decoding param found at top level (e.g. {key}={results[key]})"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No decoding/generation_config found"


def _check_benchmark_limitations(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """ME-1: Benchmark limitations acknowledged."""
    lim = results.get("limitations") or results.get("caveats")
    if lim:
        return CheckStatus.PASS, "Limitations documented"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No limitations/caveats key; document known limitations"


def _check_metric_human_correlation(results: dict[str, Any]) -> tuple[CheckStatus, str]:
    """ME-2: Metric correlation with human judgment documented."""
    corr = results.get("human_correlation") or results.get("metric_correlation")
    if corr:
        return CheckStatus.PASS, f"Human correlation: {corr}"
    return CheckStatus.MANUAL_CHECK_NEEDED, "No human_correlation key; document if available"


# ---------------------------------------------------------------------------
# Checker registry -- maps item IDs to validation functions
# ---------------------------------------------------------------------------

_CHECKERS: dict[str, _CheckerFn] = {
    # Evaluation Prompts
    "EP-1": _check_native_instructions,
    "EP-2": _check_prompt_templates_logged,
    "EP-3": _check_fewshot_exemplars,
    "EP-4": _check_format_consistency,
    # Choice of Metrics
    "CM-1": _check_primary_metric,
    "CM-2": _check_metrics_appropriate,
    "CM-3": _check_multiple_metrics,
    # Statistical Testing
    "ST-1": _check_confidence_intervals,
    "ST-2": _check_sample_sizes,
    "ST-3": _check_significance_tests,
    "ST-4": _check_power_analysis,
    # Aggregating Results
    "AR-1": _check_per_language_scores,
    "AR-2": _check_family_aggregation,
    "AR-3": _check_variance_reported,
    "AR-4": _check_resourcedness_groups,
    # Qualitative Insights
    "QI-1": _check_example_predictions,
    "QI-2": _check_error_patterns,
    "QI-3": _check_script_tokenization,
    # Reproducibility
    "RP-1": _check_model_version,
    "RP-2": _check_random_seed,
    "RP-3": _check_timestamp,
    "RP-4": _check_hardware_info,
    "RP-5": _check_decoding_params,
    # Meta-Evaluation
    "ME-1": _check_benchmark_limitations,
    "ME-2": _check_metric_human_correlation,
}

# Sanity: every item has a checker.
assert set(_CHECKERS.keys()) == {item.id for item in CHECKLIST_ITEMS}, (
    "Mismatch between CHECKLIST_ITEMS and _CHECKERS registry"
)


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_results(
    results: dict[str, Any],
    *,
    skip_items: set[str] | None = None,
    include_throughput: bool = False,
) -> ChecklistReport:
    """Run the full Deja Vu multilingual checklist against benchmark results.

    Parameters
    ----------
    results:
        Benchmark results dict (from run_mgsm, run_xcopa, etc.).
        Expected top-level keys vary by benchmark but typically include:
        ``per_language``, ``model``, ``seed``, ``timestamp``,
        ``prompt_template``, ``family_aggregation``, etc.
    skip_items:
        Set of item IDs to skip (e.g. ``{"ST-4", "ME-2"}``).
        Skipped items are recorded as PASS with a note.
    include_throughput:
        When True, accuracy-dependent checks (ST-1, ST-2, AR-1, AR-2, AR-3)
        are auto-passed because throughput benchmarks do not produce scores.

    Returns
    -------
    ChecklistReport
        Report with pass/fail/manual_check for each of the 25 items.
    """
    skip_items = skip_items or set()
    report = ChecklistReport()

    # Items that are N/A for throughput benchmarks.
    throughput_skip = {"ST-1", "ST-2", "AR-1", "AR-2", "AR-3"}

    for item in CHECKLIST_ITEMS:
        # Skip explicitly excluded items.
        if item.id in skip_items:
            report.results.append(
                CheckResult(item, CheckStatus.PASS, "Skipped by caller")
            )
            continue

        # Auto-pass throughput-irrelevant items.
        if include_throughput and item.id in throughput_skip:
            report.results.append(
                CheckResult(item, CheckStatus.PASS, "N/A for throughput benchmark")
            )
            continue

        checker = _CHECKERS.get(item.id)
        if checker is None:
            # Defensive: should not happen given the assert above.
            report.results.append(
                CheckResult(item, CheckStatus.MANUAL_CHECK_NEEDED, "No automated checker")
            )
            continue

        try:
            status, detail = checker(results)
            report.results.append(CheckResult(item, status, detail))
        except Exception as exc:
            logger.warning("Checker %s raised: %s", item.id, exc)
            report.results.append(
                CheckResult(item, CheckStatus.FAIL, f"Checker raised exception: {exc}")
            )

    return report
