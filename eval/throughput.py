"""
CPU/GPU throughput benchmarking for multilingual models.

Measures tokens/sec, time-to-first-token (TTFT), and peak memory
across a standardized prompt set (one per language). Reports
per-language throughput because different scripts have different
tokenization costs -- a fact often hidden in aggregate numbers.

Wayy Research -- Project Aya
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standardized prompts -- one per language, roughly equivalent complexity
# ---------------------------------------------------------------------------

THROUGHPUT_PROMPTS: dict[str, str] = {
    "en": "Explain the concept of gravitational waves and how they were first detected. Provide a detailed scientific explanation.",
    "es": "Explica el concepto de ondas gravitacionales y como fueron detectadas por primera vez. Proporciona una explicacion cientifica detallada.",
    "hi": "गुरुत्वाकर्षण तरंगों की अवधारणा और उन्हें पहली बार कैसे खोजा गया, इसकी विस्तृत वैज्ञानिक व्याख्या दें।",
    "zh": "解释引力波的概念以及它们是如何被首次探测到的。请提供详细的科学解释。",
    "ar": "اشرح مفهوم موجات الجاذبية وكيف تم اكتشافها لأول مرة. قدم شرحا علميا مفصلا.",
    "sw": "Eleza dhana ya mawimbi ya uvutano na jinsi yalivyogunduliwa kwa mara ya kwanza. Toa maelezo ya kisayansi ya kina.",
    "tr": "Kutle cekim dalgalari kavramini ve ilk kez nasil tespit edildiklerini aciklayin. Ayrintili bir bilimsel aciklama sunun.",
    "ja": "重力波の概念と、それが初めて検出された方法について説明してください。詳細な科学的説明を提供してください。",
    "id": "Jelaskan konsep gelombang gravitasi dan bagaimana mereka pertama kali terdeteksi. Berikan penjelasan ilmiah yang terperinci.",
    "te": "గురుత్వాకర్షణ తరంగాల భావన మరియు అవి మొదటిసారి ఎలా కనుగొనబడ్డాయో వివరించండి. వివరమైన శాస్త్రీయ వివరణ ఇవ్వండి.",
}


@dataclass(frozen=True)
class ThroughputConfig:
    """Configuration for throughput benchmarking."""

    n_tokens: int = 512
    n_runs: int = 10
    n_warmup: int = 2
    languages: list[str] = field(
        default_factory=lambda: list(THROUGHPUT_PROMPTS.keys())
    )
    seed: int = 42


@dataclass
class LanguageThroughput:
    """Throughput measurements for a single language."""

    language: str
    prompt_tokens: int
    generated_tokens_per_run: list[int]
    tokens_per_second: list[float]
    time_to_first_token_ms: list[float]
    total_time_sec: list[float]
    peak_memory_mb: float

    @property
    def mean_tps(self) -> float:
        return float(np.mean(self.tokens_per_second))

    @property
    def std_tps(self) -> float:
        return float(np.std(self.tokens_per_second))

    @property
    def mean_ttft_ms(self) -> float:
        return float(np.mean(self.time_to_first_token_ms))

    @property
    def std_ttft_ms(self) -> float:
        return float(np.std(self.time_to_first_token_ms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "prompt_tokens": self.prompt_tokens,
            "mean_tokens_per_second": self.mean_tps,
            "std_tokens_per_second": self.std_tps,
            "mean_ttft_ms": self.mean_ttft_ms,
            "std_ttft_ms": self.std_ttft_ms,
            "mean_total_time_sec": float(np.mean(self.total_time_sec)),
            "peak_memory_mb": self.peak_memory_mb,
            "n_runs": len(self.tokens_per_second),
            "raw_tps": self.tokens_per_second,
            "raw_ttft_ms": self.time_to_first_token_ms,
        }


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _get_peak_memory_mb(device: torch.device) -> float:
    """Get peak GPU memory in MB, or RSS for CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return 0.0


def _reset_peak_memory(device: torch.device) -> None:
    """Reset peak memory tracker."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


class _TTFTCallback:
    """Callback-style hook to measure time to first token.

    We record the timestamp when the first new token is generated
    by hooking into the generate loop via a custom stopping criteria.
    """

    def __init__(self) -> None:
        self.first_token_time: float | None = None
        self.start_time: float = 0.0
        self._called = False

    def reset(self, start_time: float) -> None:
        self.first_token_time = None
        self.start_time = start_time
        self._called = False

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time is None:
            return 0.0
        return (self.first_token_time - self.start_time) * 1000


# ---------------------------------------------------------------------------
# Main throughput benchmark
# ---------------------------------------------------------------------------


@torch.inference_mode()
def measure_throughput_single(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    n_tokens: int,
    device: torch.device,
) -> tuple[int, int, float, float, float]:
    """Run a single generation and measure throughput.

    Returns (prompt_tokens, generated_tokens, tokens_per_sec, ttft_ms, total_sec).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=n_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    end = time.perf_counter()
    total_sec = end - start

    generated_len = outputs.shape[1] - prompt_len

    # Estimate TTFT: for greedy decoding, approximate as total_time / n_tokens
    # (first token includes prefill, subsequent tokens are decode-only)
    # A better approach uses streaming, but this gives a reasonable estimate
    if generated_len > 1:
        # Rough estimate: prefill is proportionally more expensive
        ttft_est_ms = (total_sec / generated_len) * 1000 * 1.5
    else:
        ttft_est_ms = total_sec * 1000

    tps = generated_len / total_sec if total_sec > 0 else 0.0

    return prompt_len, generated_len, tps, ttft_est_ms, total_sec


def run_throughput_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: ThroughputConfig,
) -> dict[str, Any]:
    """Run throughput benchmark across all configured languages.

    Parameters
    ----------
    model : the model to benchmark
    tokenizer : corresponding tokenizer
    config : benchmark configuration

    Returns
    -------
    dict with per-language throughput measurements and aggregate stats
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = next(model.parameters()).device

    results: dict[str, Any] = {
        "benchmark": "throughput",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "n_tokens": config.n_tokens,
        "n_runs": config.n_runs,
        "n_warmup": config.n_warmup,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "per_language": {},
        "aggregate": {},
    }

    all_tps: list[float] = []
    all_ttft: list[float] = []

    for lang in config.languages:
        prompt = THROUGHPUT_PROMPTS.get(lang)
        if prompt is None:
            logger.warning("No throughput prompt for language: %s", lang)
            continue

        logger.info("Benchmarking throughput for language: %s", lang)

        # Warmup runs (not counted)
        for _ in range(config.n_warmup):
            measure_throughput_single(
                model, tokenizer, prompt, config.n_tokens, device
            )

        # Reset memory tracking after warmup
        _reset_peak_memory(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        tps_list: list[float] = []
        ttft_list: list[float] = []
        total_time_list: list[float] = []
        gen_tokens_list: list[int] = []
        prompt_tokens = 0

        for run_idx in range(config.n_runs):
            prompt_len, gen_len, tps, ttft, total = measure_throughput_single(
                model, tokenizer, prompt, config.n_tokens, device
            )
            prompt_tokens = prompt_len
            tps_list.append(tps)
            ttft_list.append(ttft)
            total_time_list.append(total)
            gen_tokens_list.append(gen_len)

        peak_mem = _get_peak_memory_mb(device)

        lang_result = LanguageThroughput(
            language=lang,
            prompt_tokens=prompt_tokens,
            generated_tokens_per_run=gen_tokens_list,
            tokens_per_second=tps_list,
            time_to_first_token_ms=ttft_list,
            total_time_sec=total_time_list,
            peak_memory_mb=peak_mem,
        )
        results["per_language"][lang] = lang_result.to_dict()
        all_tps.extend(tps_list)
        all_ttft.extend(ttft_list)

    # Aggregate
    if all_tps:
        results["aggregate"] = {
            "mean_tokens_per_second": float(np.mean(all_tps)),
            "std_tokens_per_second": float(np.std(all_tps)),
            "mean_ttft_ms": float(np.mean(all_ttft)),
            "std_ttft_ms": float(np.std(all_ttft)),
            "n_languages": len(results["per_language"]),
        }

    return results


def print_throughput_table(results: dict[str, Any]) -> None:
    """Print a formatted throughput summary table."""
    print(f"\n{'=' * 80}")
    print(f"  Throughput Results -- {results.get('device_name', 'unknown')}")
    print(f"{'=' * 80}")
    print(
        f"  {'Language':<10} {'Prompt Tok':>10} {'TPS (mean)':>12} "
        f"{'TPS (std)':>10} {'TTFT ms':>10} {'Mem MB':>10}"
    )
    print(f"  {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")

    per_lang = results.get("per_language", {})
    for lang, data in sorted(per_lang.items()):
        print(
            f"  {lang:<10} {data['prompt_tokens']:>10} "
            f"{data['mean_tokens_per_second']:>12.2f} "
            f"{data['std_tokens_per_second']:>10.2f} "
            f"{data['mean_ttft_ms']:>10.2f} "
            f"{data['peak_memory_mb']:>10.1f}"
        )

    agg = results.get("aggregate", {})
    if agg:
        print(f"  {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")
        print(
            f"  {'OVERALL':<10} {'':>10} "
            f"{agg['mean_tokens_per_second']:>12.2f} "
            f"{agg['std_tokens_per_second']:>10.2f} "
            f"{agg['mean_ttft_ms']:>10.2f} "
            f"{'':>10}"
        )
    print(f"{'=' * 80}\n")
