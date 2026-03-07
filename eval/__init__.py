"""
Multilingual evaluation pipeline for Project Aya.

Provides benchmark harnesses (mGSM, XCOPA), throughput measurement,
shared metrics utilities, and the Cohere multilingual eval checklist.

Usage::

    from eval.benchmarks import run_mgsm, run_xcopa, BenchmarkConfig
    from eval.throughput import run_throughput_benchmark, ThroughputConfig
    from eval.metrics import bootstrap_confidence_interval, degradation_equity_score
    from eval.checklist import validate_results

Wayy Research -- Project Aya
"""
from eval.benchmarks import (
    BenchmarkConfig,
    load_model_and_tokenizer,
    print_results_table,
    run_mgsm,
    run_xcopa,
    save_results,
)
from eval.checklist import ChecklistReport, validate_results
from eval.metrics import (
    ConfidenceInterval,
    FamilyResult,
    LanguageResult,
    aggregate_by_family,
    bootstrap_confidence_interval,
    degradation_equity_score,
    paired_bootstrap_test,
)
from eval.throughput import (
    ThroughputConfig,
    print_throughput_table,
    run_throughput_benchmark,
)

__all__ = [
    # Benchmarks
    "BenchmarkConfig",
    "load_model_and_tokenizer",
    "run_mgsm",
    "run_xcopa",
    "save_results",
    "print_results_table",
    # Throughput
    "ThroughputConfig",
    "run_throughput_benchmark",
    "print_throughput_table",
    # Metrics
    "ConfidenceInterval",
    "LanguageResult",
    "FamilyResult",
    "bootstrap_confidence_interval",
    "degradation_equity_score",
    "paired_bootstrap_test",
    "aggregate_by_family",
    # Checklist
    "ChecklistReport",
    "validate_results",
]
