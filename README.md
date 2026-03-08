# Project Aya

**Compression, Equity, and the Architecture of Linguistic Inclusion**

Cohere Labs x Wayy Research | Buffalo, NY | Est. 2026 | Open Science / Open Weights

---

## Overview

Most efficient AI models are English-centric. The architecture of language technology is leaving billions behind. Project Aya investigates whether transformer-based multilingual models can be distilled into Mamba architectures while preserving multilingual capability, structured tool use, and cross-lingual reasoning — and whether compression degrades uniformly across language families.

### Core Research Question

> *Can transformer-based multilingual models be distilled into Mamba architectures while preserving multilingual capability, structured tool use, and cross-lingual reasoning — and does the compression degrade uniformly across language families?*

### Teacher Model: tiny-aya-global

[CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) — 3.35B parameters, 70+ languages, 4-layer transformer (3 sliding window attention + 1 global attention), CC-BY-NC license. Loads without quantization on 4GB VRAM.

### Student Architecture: Aetheris (Hybrid Mamba-MoE)

A novel student architecture combining selective state spaces with sparse mixture-of-experts, distilled from tiny-aya-global into Aetheris (~500-800M params, Mamba-MoE).

| Component | Symbol | Description |
|-----------|--------|-------------|
| SSM Blocks | Psi | O(n) selective scan with constant memory. Mamba backbone enables linear-time sequence processing |
| Sparse MoE | Sigma | 4 expert FFNs with top-1 routing and load balancing. Only one expert fires per token |
| Hybrid Design | Delta | SSM blocks on even layers, MoE on odd layers. 24 total layers with weight-tied embeddings and gradient checkpointing |

### 3-Stage MambaInLlama Pipeline

1. **Layer Alignment** — Map transformer attention layers to Mamba SSM blocks with CKA-guided structural correspondence
2. **KL Distillation** — Soft-target training with KL divergence to transfer knowledge from teacher to student
3. **Supervised Fine-Tuning** — Restore multilingual capability and structured tool calling via targeted SFT

---

## Installation

```bash
pip install "aya-distill @ git+https://github.com/Wayy-Research/project-aya.git"
```

That's it. One install gives you everything: evaluation, distillation, CKA analysis, multilingual testing, reasoning evaluation, metrics, and the CLI.

### For development

```bash
git clone https://github.com/Wayy-Research/project-aya.git
cd project-aya
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e "."
```

> **Note**: If you have system-wide PyTorch with CUDA, use
> `uv venv --python 3.10 --system-site-packages` to reuse it.
> For Jupyter notebooks add `uv pip install -e ".[notebooks]"`.
> For 4-bit quantization add `uv pip install -e ".[quantize]"`.

---

## Package: `aya-distill`

### Python API

```python
# Language registry — single source of truth for 67 target languages
from aya_distill.languages import LANGUAGES, LANGUAGE_FAMILIES, RESOURCE_GROUPS

# CKA layer similarity measurement
from aya_distill.distill.cka import linear_cka, rbf_cka, minibatch_cka

# Evaluation metrics and statistical testing
from aya_distill.eval.metrics import degradation_equity_score, bootstrap_confidence_interval

# Deja Vu evaluation checklist (30 items, 8 categories)
from aya_distill.eval.checklist import validate_results, CHECKLIST_ITEMS

# Multilingual quality and tool calling testing
from aya_distill.testing import MultilingualTestSuite, ToolCallingTest

# Reasoning evaluation with CoT scoring
from aya_distill.testing import ReasoningTestSuite
```

### CLI

```bash
aya-distill convert --strategy weight_map --output checkpoints/converted/
aya-distill distill --stage 1 --config configs/distill_stage1.yaml
aya-distill eval --config configs/eval_baseline.yaml
aya-distill profile --quantize 4bit
```

---

## Research Examples

### 1. Multilingual Quality Testing (67 Languages)

Test any model's coherence, relevance, and cross-lingual consistency:

```python
from aya_distill.testing import MultilingualTestSuite
from aya_distill.languages import LANGUAGE_FAMILIES, RESOURCE_GROUPS

# Wrap your model as a callable
def my_model(prompt: str) -> str:
    return model.generate(prompt)

# Run quality tests across all 67 languages
suite = MultilingualTestSuite(model_fn=my_model)
result = suite.run()
print(result.summary())
# Output:
# Multilingual Test Results
# ==================================================
#   en (   English): coherence=0.80 relevance=0.70 latency=0.234s [PASS]
#   es (   Spanish): coherence=0.80 relevance=0.70 latency=0.198s [PASS]
#   ar (    Arabic): coherence=0.80 relevance=0.70 latency=0.312s [PASS]
#   ...
#   Mean coherence:              0.780
#   Mean relevance:              0.690
#   Cross-lingual consistency:   0.952
#   Equity score (lower=better): 0.0012

# Test a subset of languages (e.g. low-resource only)
low_resource_suite = MultilingualTestSuite(
    model_fn=my_model,
    languages=RESOURCE_GROUPS["low"],
)
low_result = low_resource_suite.run()

# Use an LLM judge for higher-quality scoring
suite_with_judge = MultilingualTestSuite(
    model_fn=my_model,
    judge_model="gpt-4",  # or any testLLM-compatible model
    thresholds={"coherence": 0.6, "relevance": 0.6},
)
judged_result = suite_with_judge.run()

# Compare student against teacher — per-language degradation
degradation = suite.compare(teacher_fn=teacher_model)
# Returns: {"en": {"coherence_delta": 0.05, "relevance_delta": 0.02, ...}, ...}
# Positive delta = student is worse than teacher
```

### 2. Multilingual Reasoning Evaluation

Evaluate chain-of-thought quality, logical consistency, and reasoning correctness across 4 categories (mathematical, causal, analogical, logical) in 45+ languages:

```python
from aya_distill.testing import ReasoningTestSuite

def my_model(prompt: str) -> str:
    return model.generate(prompt)

# Run full reasoning suite across all languages with fixtures
reasoning = ReasoningTestSuite(model_fn=my_model)
result = reasoning.run()
print(result.summary())
# Output:
# Multilingual Reasoning Results
# ============================================================
#   en (  mathematical): score=0.85 cot=0.90 logic=0.80 [CoT] [OK]
#   en (        causal): score=0.72 cot=0.70 logic=0.75 [CoT] [OK]
#   es (  mathematical): score=0.82 cot=0.85 logic=0.78 [CoT] [OK]
#   ar (  mathematical): score=0.68 cot=0.65 logic=0.70 [CoT] [X]
#   ...
#
#   Mean CoT quality:         0.742
#   Mean logical consistency: 0.695
#   Mean score:               0.718
#   CoT equity (lower=better):0.0034
#
#   By language family:
#     Indo-European             0.785
#     Afroasiatic               0.672
#     Niger-Congo               0.645
#     Turkic                    0.698

# Test specific reasoning categories
math_only = ReasoningTestSuite(
    model_fn=my_model,
    categories=["mathematical", "logical"],
    languages=["en", "es", "hi", "ar", "zh", "sw"],
)
math_result = math_only.run()

# Compare reasoning between student and teacher
reasoning_degradation = reasoning.compare(teacher_fn=teacher_model)
# Returns per-language deltas:
# {"en": {"mathematical_delta": 0.05, "mathematical_cot_delta": 0.03, ...}, ...}

# Export results for downstream analysis
import json
with open("reasoning_results.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)
```

### 3. Tool Calling Across Languages

Verify structured JSON tool call generation in 67 languages:

```python
from aya_distill.testing import ToolCallingTest

tool_test = ToolCallingTest(model_fn=my_model)

# Run across all 67 languages with built-in fixtures
tool_results = tool_test.run_multilingual()
print(tool_test.summary(tool_results))
# Output:
# Tool Calling Results
# ==================================================
#   en (   English): score=1.00 [JSON+FN+ARGS]
#   es (   Spanish): score=0.75 [JSON+FN]
#   ar (    Arabic): score=0.50 [JSON]
#   ...
#   Mean score: 0.723

# Custom tool calling scenario
result = tool_test.test_single(
    prompt="Busca vuelos de Madrid a Tokio para el 15 de marzo",
    expected_function="search_flights",
    expected_args={"origin": "Madrid", "destination": "Tokyo"},
    lang_code="es",
    language="Spanish",
)
print(f"JSON valid: {result.json_valid}")
print(f"Function correct: {result.function_correct}")
print(f"Arguments correct: {result.args_correct}")
```

### 4. CKA Layer Similarity Analysis

Monitor information preservation during distillation using Centered Kernel Alignment:

```python
import torch
from aya_distill.distill.cka import (
    linear_cka,
    rbf_cka,
    minibatch_cka,
    cka_permutation_test,
    compute_layerwise_cka,
    MinibatchCKAAccumulator,
)

# Compare two layers' activations
teacher_acts = torch.randn(256, 768)   # 256 samples, 768-dim
student_acts = torch.randn(256, 1024)  # different dimension is fine

# Linear CKA (fast, O(n*d^2))
score = linear_cka(teacher_acts, student_acts)
print(f"Linear CKA: {score:.4f}")  # 0.0 = no similarity, 1.0 = identical

# RBF kernel CKA (captures nonlinear relationships)
score = rbf_cka(teacher_acts, student_acts)
print(f"RBF CKA: {score:.4f}")

# Mini-batch CKA for memory efficiency on large activation sets
score = minibatch_cka(teacher_acts, student_acts, batch_size=64)

# Streaming mini-batch CKA for datasets that don't fit in memory
acc = MinibatchCKAAccumulator(d_x=768, d_y=1024)
for batch in dataloader:
    teacher_batch = teacher.get_activations(batch)  # (B, 768)
    student_batch = student.get_activations(batch)  # (B, 1024)
    acc.update(teacher_batch, student_batch)
cka_score = acc.compute()

# Statistical significance: is this CKA score above chance?
test_result = cka_permutation_test(teacher_acts, student_acts, n_permutations=1000)
print(f"Observed CKA: {test_result['observed_cka']:.4f}")
print(f"p-value: {test_result['p_value']:.4f}")
print(f"Null mean: {test_result['null_mean']:.4f} +/- {test_result['null_std']:.4f}")

# Full layer-wise heatmap (teacher x student)
heatmap = compute_layerwise_cka(
    teacher_activations={"layer_0": t0, "layer_1": t1, "layer_2": t2, "layer_3": t3},
    student_activations={"block_0": s0, "block_1": s1, ..., "block_23": s23},
    kernel="linear",
    batch_size=128,  # memory-efficient mode
)
# heatmap.scores is (4, 24) numpy array — visualize with seaborn/matplotlib
import seaborn as sns
sns.heatmap(heatmap.scores, xticklabels=heatmap.student_layer_names,
            yticklabels=heatmap.teacher_layer_names, cmap="viridis")
```

### 5. Evaluation Metrics and Statistical Testing

Rigorous per-language metrics with bootstrap confidence intervals and equity scoring:

```python
import numpy as np
from aya_distill.eval.metrics import (
    bootstrap_confidence_interval,
    paired_bootstrap_test,
    degradation_equity_score,
    aggregate_by_family,
    accuracy_from_predictions,
)

# Per-language accuracy with 95% bootstrap CI
correct = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1])  # per-sample correctness
ci = bootstrap_confidence_interval(correct, n_bootstrap=1000, confidence_level=0.95)
print(f"Accuracy: {ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]")

# Compare two systems with paired bootstrap test
system_a = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1])
system_b = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
test = paired_bootstrap_test(system_a, system_b)
print(f"Mean diff: {test.mean_diff:.4f}, p-value: {test.p_value:.4f}")
print(f"Significant: {test.significant}")

# Degradation Equity Score — does distillation pick winners and losers?
teacher_scores = {"en": 0.85, "es": 0.82, "hi": 0.75, "ar": 0.70, "sw": 0.65}
student_scores = {"en": 0.80, "es": 0.78, "hi": 0.68, "ar": 0.55, "sw": 0.50}
des = degradation_equity_score(teacher_scores, student_scores)
print(f"DES: {des:.6f}")  # Lower = more equitable compression

# Aggregate per-language scores by language family
family_results = aggregate_by_family(teacher_scores)
for fam in family_results:
    print(f"  {fam.family}: {fam.mean_accuracy:.3f} ({', '.join(fam.languages)})")

# Compute accuracy from raw predictions
preds = ["23", "42", "23", "15"]
refs  = ["23", "42", "24", "15"]
correct = accuracy_from_predictions(preds, refs)
# array([1, 1, 0, 1])
```

### 6. Benchmarks: mGSM and XCOPA

Run standardized multilingual benchmarks with full reproducibility metadata:

```python
from aya_distill.eval.benchmarks import (
    run_mgsm,
    run_xcopa,
    BenchmarkConfig,
    load_model_and_tokenizer,
    save_results,
    print_results_table,
)

# Load the teacher model
model, tokenizer = load_model_and_tokenizer(
    "CohereLabs/tiny-aya-global",
    dtype="float16",
    device_map="auto",
)

# Configure the benchmark
config = BenchmarkConfig(
    model_name="CohereLabs/tiny-aya-global",
    n_shot=8,
    languages=["en", "es", "hi", "zh", "ar", "sw", "tr", "ja", "id", "te"],
    max_new_tokens=256,
    n_bootstrap=1000,
    seed=42,
    output_dir="results/baseline",
)

# Run mGSM (multilingual grade school math)
mgsm_results = run_mgsm(model, tokenizer, config)
print_results_table(mgsm_results)
# Per-language accuracy with 95% CI, family aggregation, metadata

# Run XCOPA (cross-lingual commonsense reasoning)
xcopa_results = run_xcopa(model, tokenizer, config)
print_results_table(xcopa_results)

# Save with full reproducibility metadata
save_results(mgsm_results, "results/baseline", "mgsm")
save_results(xcopa_results, "results/baseline", "xcopa")
```

### 7. Deja Vu Evaluation Checklist

Validate any benchmark run against the [Cohere Deja Vu multilingual evaluation checklist](https://github.com/Cohere-Labs/multilingual-llm-evaluation-checklist) — 30 items across 8 categories (25 original + 5 reasoning extension):

```python
from aya_distill.eval.checklist import validate_results, CHECKLIST_ITEMS

# Validate benchmark results for compliance
report = validate_results(mgsm_results)
report.print_report()
# ==========================================================================
#   Deja Vu Multilingual Evaluation Checklist -- COMPLIANT
#   Score: 28 pass / 2 manual / 0 fail  (100% of 30 items)
# ==========================================================================
#
#   [Evaluation Prompts]  (OK)
#     [PASS] [REQ] EP-1: Native-language evaluation prompts
#     [PASS] [REQ] EP-2: Prompt templates logged and versioned
#     ...
#   [Reasoning Evaluation]  (OK)
#     [PASS] [REQ] RE-1: Chain-of-thought quality assessed per language
#     [PASS] [REQ] RE-2: Multiple reasoning categories tested
#     [PASS] [REQ] RE-3: Reasoning equity across language families documented
#     ...

# Check compliance programmatically
assert report.compliant, f"{report.n_required_failed} required items failed"

# Include reasoning data in your results for RE-1 through RE-5
results_with_reasoning = {
    **mgsm_results,
    "reasoning": {
        "en": {
            "mathematical": {"cot_quality": 0.85, "n_steps_found": 4},
            "causal": {"cot_quality": 0.72, "n_steps_found": 3},
        },
        "es": {
            "mathematical": {"cot_quality": 0.80, "n_steps_found": 3},
            "causal": {"cot_quality": 0.70, "n_steps_found": 2},
        },
    },
    "family_scores": {"Indo-European": 0.78, "Afroasiatic": 0.65},
    "cot_equity_score": 0.0034,
}
report = validate_results(results_with_reasoning)

# Skip specific items if not applicable
report = validate_results(results, skip_items={"ST-4", "ME-2"})

# For throughput-only benchmarks (no accuracy scores)
report = validate_results(throughput_results, include_throughput=True)

# Serialize for CI/CD integration
import json
print(json.dumps(report.to_dict(), indent=2))
```

### 8. End-to-End Research Workflow

Combine all components for a complete multilingual evaluation:

```python
import json
from aya_distill.testing import (
    MultilingualTestSuite,
    ReasoningTestSuite,
    ToolCallingTest,
)
from aya_distill.eval.metrics import degradation_equity_score
from aya_distill.eval.checklist import validate_results
from aya_distill.languages import DISTILLATION_LANGUAGES, LANGUAGE_FAMILIES

# Define model callables
def teacher(prompt: str) -> str:
    return teacher_model.generate(prompt)

def student(prompt: str) -> str:
    return student_model.generate(prompt)

# ---- Phase 1: Multilingual quality ----
quality_suite = MultilingualTestSuite(
    model_fn=student,
    languages=DISTILLATION_LANGUAGES,  # 10 core languages
)
quality_result = quality_suite.run()
quality_degradation = quality_suite.compare(teacher_fn=teacher)

# ---- Phase 2: Reasoning ----
reasoning_suite = ReasoningTestSuite(
    model_fn=student,
    languages=DISTILLATION_LANGUAGES,
)
reasoning_result = reasoning_suite.run()
reasoning_degradation = reasoning_suite.compare(teacher_fn=teacher)

# ---- Phase 3: Tool calling ----
tool_test = ToolCallingTest(model_fn=student)
tool_results = tool_test.run_multilingual()

# ---- Phase 4: Equity analysis ----
teacher_quality = MultilingualTestSuite(model_fn=teacher, languages=DISTILLATION_LANGUAGES)
teacher_result = teacher_quality.run()

teacher_scores = {
    lang: r.coherence_score
    for lang, r in teacher_result.language_results.items()
}
student_scores = {
    lang: r.coherence_score
    for lang, r in quality_result.language_results.items()
}
des = degradation_equity_score(teacher_scores, student_scores)
print(f"Degradation Equity Score: {des:.6f}")

# ---- Phase 5: Checklist validation ----
combined_results = {
    "model": "aetheris-mamba-moe-v1",
    "seed": 42,
    "timestamp": reasoning_result.timestamp,
    "primary_metric": "accuracy",
    "per_language": {
        lang: {
            "accuracy": r.coherence_score,
            "ci": {"lower": 0.0, "upper": 1.0},
            "n_samples": 250,
        }
        for lang, r in quality_result.language_results.items()
    },
    "family_aggregation": {
        fam: langs for fam, langs in LANGUAGE_FAMILIES.items()
    },
    "variance": quality_result.equity_score,
    "prompt_template": {
        "type": "few_shot",
        "n_shot": 8,
        "instruction_language": "native",
    },
    "reasoning": reasoning_result.to_dict()["languages"],
    "family_scores": reasoning_result.family_scores,
    "cot_equity_score": reasoning_result.cot_equity_score,
    "reasoning_degradation": reasoning_degradation,
}
report = validate_results(combined_results)
report.print_report()

# ---- Export everything ----
with open("results/full_evaluation.json", "w") as f:
    json.dump({
        "quality": quality_result.to_dict(),
        "reasoning": reasoning_result.to_dict(),
        "tool_calling": {k: v.to_dict() for k, v in tool_results.items()},
        "degradation_equity_score": des,
        "checklist": report.to_dict(),
    }, f, indent=2, default=str)
```

---

## Language Registry

Single source of truth for all 67 target languages:

```python
from aya_distill.languages import (
    LANGUAGES,              # 67 Language dataclasses
    LANGUAGE_FAMILIES,      # 16 families
    SCRIPT_GROUPS,          # 19 scripts
    RESOURCE_GROUPS,        # high / medium / low
    REGION_GROUPS,          # 7 regions
    DISTILLATION_LANGUAGES, # 10 core distillation languages
)

# Each language has rich metadata
lang = LANGUAGES["sw"]
# Language(code='sw', name='Swahili', family='Niger-Congo',
#          script='Latin', typology='Agglutinative',
#          resourcedness='low', region='Africa')

# Filter by family
dravidian = LANGUAGE_FAMILIES["Dravidian"]
# ['ta', 'te']

# Filter by resource level
low_resource = RESOURCE_GROUPS["low"]
# ['am', 'et', 'eu', 'ga', 'gl', 'gu', 'ha', 'ig', 'jv', 'km', ...]

# Core distillation languages (10)
# ['ar', 'en', 'es', 'hi', 'id', 'ja', 'sw', 'te', 'tr', 'zh']
```

### Languages: 67 Languages, 16 Families, 19 Scripts

| Family | Languages |
|--------|-----------|
| **Indo-European** | English, Spanish, French, Portuguese, German, Italian, Dutch, Polish, Romanian, Ukrainian, Czech, Greek, Hindi, Bengali, Marathi, Gujarati, Nepali, Persian, Urdu, Russian, Croatian, Slovak, Slovenian, Serbian, Bulgarian, Catalan, Galician, Danish, Swedish, Norwegian, Latvian, Lithuanian, Welsh, Irish |
| **Afroasiatic** | Arabic, Hebrew, Amharic, Hausa, Maltese |
| **Niger-Congo** | Swahili, Yoruba, Igbo, Zulu, Xhosa, Shona, Wolof |
| **Turkic** | Turkish |
| **Sino-Tibetan** | Chinese, Burmese |
| **Japonic** | Japanese |
| **Koreanic** | Korean |
| **Dravidian** | Telugu, Tamil |
| **Austronesian** | Indonesian, Malay, Tagalog, Malagasy, Javanese |
| **Austroasiatic** | Vietnamese, Khmer |
| **Kra-Dai** | Thai, Lao |
| **Uralic** | Finnish, Hungarian, Estonian |
| **Language isolate** | Basque |

### Resource Levels

- **High** (10): en, es, fr, pt, de, it, nl, zh, ja, ko, ar, hi, ru
- **Medium** (26): ro, uk, cs, pl, el, he, fa, ur, tr, mr, bn, ta, te, ms, id, vi, th, tl, fi, hu, da, sv, no, hr, sk, sl, sr, bg, km, ca
- **Low** (31): am, et, eu, ga, gl, gu, ha, ig, jv, km, lo, lv, lt, mg, mt, my, ne, pa, sn, sw, wo, xh, yo, zu, cy

---

## Evaluation

### Benchmarks

| Benchmark | Description |
|-----------|-------------|
| **mGSM** | Multilingual grade school math across target languages, 250 problems each |
| **XCOPA** | Cross-lingual commonsense reasoning |
| **Reasoning Suite** | 4-category CoT evaluation (mathematical, causal, analogical, logical) in 45+ languages |
| **Tool Calling** | Structured JSON generation and function selection in 67 languages |
| **CPU Throughput** | Tokens/sec, peak memory, time-to-first-token on consumer hardware |

### Degradation Equity Score

Key novel metric: the variance of accuracy drop across language families after distillation. Measures whether compression picks winners and losers. Lower is better — compression should not pick winners.

### Reasoning Evaluation

Chain-of-thought quality is scored on 5 dimensions:
- **Step presence** (0.3): Does the model show reasoning steps?
- **Structure** (0.2): Are steps numbered or marked with connectives?
- **Logical connectives** (0.2): Does the model use "because", "therefore", etc.?
- **Response length** (0.15): Is the response adequately detailed?
- **Final answer** (0.15): Does the model arrive at a clear answer?

Logical consistency scoring detects contradictions (penalty), progressive reasoning (bonus), and conclusion markers (bonus). The composite reasoning score combines CoT quality (35%), logical consistency (25%), answer correctness (25%), and step count (15%).

### Cohere Deja Vu Checklist

All evaluations are validated against the [Cohere multilingual LLM evaluation checklist](https://github.com/Cohere-Labs/multilingual-llm-evaluation-checklist) — extended to 30 items across 8 categories (original 25 + 5 reasoning evaluation items by Wayy Research). Per-language scores are always primary — never hidden behind averages.

Categories: Data & Sampling, Metrics & Aggregation, Cultural & Linguistic Adequacy, Human Evaluation, Reproducibility & Documentation, Bias & Fairness, Robustness & Generalization, Reasoning Evaluation.

---

## Distillation Pipeline

### Stage 1: Layer Alignment

Maps teacher attention layers to student SSM blocks and teacher FFN layers to student MoE blocks. Monitored with **Centered Kernel Alignment (CKA)** — CKA < 0.75 triggers investigation of information loss.

### Stage 2: KL Distillation

Soft-target training with temperature-scaled KL divergence (T=2.0). Combined loss: `0.7 * KL + 0.3 * CE`. Per-language KL tracking to detect if distillation favors some languages over others.

### Stage 3: Supervised Fine-Tuning

Restores multilingual capability and structured tool calling via targeted SFT on chat-format data with language-balanced sampling.

### Data: NVIDIA ClimbMix

400B-token CLIMB-filtered dataset that outperforms FineWeb-Edu, DCLM, and Olmo under equal token budgets. Two tokenizer modes:
- **pretokenized**: GPT-2 token IDs directly (fast)
- **retokenize**: decode to text, re-encode with Aya 256k tokenizer (for multilingual student)

---

## Repository Structure

```
project-aya/
├── README.md
├── CONTRIBUTING.md             # Branching strategy, commit conventions, PR workflow
├── LICENSE                     # MIT
├── pyproject.toml              # aya-distill package config (hatchling, src-layout)
│
├── src/aya_distill/            # pip-installable package
│   ├── __init__.py             # Package root (version)
│   ├── cli.py                  # Unified CLI: aya-distill
│   ├── languages.py            # 67 target languages with metadata
│   │
│   ├── distill/                # 3-stage distillation pipeline
│   │   ├── converter.py        # Core: attention->SSM and FFN->MoE weight mapping
│   │   ├── block_surgery.py    # Full model conversion (all layers + embeddings)
│   │   ├── alignment.py        # Stage 1: Layer alignment with CKA monitoring
│   │   ├── kl_distillation.py  # Stage 2: KL divergence with temperature scaling
│   │   ├── sft.py              # Stage 3: Multilingual SFT with tool calling recovery
│   │   ├── cka.py              # CKA module (linear, RBF, mini-batch, permutation test)
│   │   ├── hooks.py            # Activation extraction for teacher + student models
│   │   ├── data.py             # Language-balanced multilingual data pipeline
│   │   └── climbmix.py         # NVIDIA ClimbMix 400B-token dataset loader
│   │
│   ├── eval/                   # Multilingual evaluation pipeline
│   │   ├── benchmarks.py       # mGSM + XCOPA harness (few-shot, per-language)
│   │   ├── metrics.py          # Bootstrap CIs, DES, family aggregation
│   │   ├── throughput.py       # Tokens/sec, TTFT, peak memory benchmarking
│   │   ├── checklist.py        # Deja Vu eval checklist (30 items, 8 categories)
│   │   └── prompts/            # Native-language few-shot templates
│   │       ├── mgsm.py         # 8-shot mGSM prompts for 67 languages
│   │       └── xcopa.py        # 4-shot XCOPA prompts for 67 languages
│   │
│   └── testing/                # Multilingual test harness (extends testkitLLM)
│       ├── multilingual.py     # MultilingualTestSuite — per-language quality + equity
│       ├── tool_testing.py     # ToolCallingTest — JSON tool call verification
│       ├── reasoning.py        # ReasoningTestSuite — CoT quality + logical consistency
│       └── fixtures.py         # Prompts, tool schemas, reasoning scenarios in 45+ languages
│
├── configs/
│   ├── student.yaml            # Aetheris student model (1024d, 24L, 4 experts)
│   ├── eval_baseline.yaml      # Teacher baseline benchmark config
│   ├── distill_stage1.yaml     # Layer alignment: 10k steps, CKA threshold 0.75
│   ├── distill_stage2.yaml     # KL distillation: 20k steps, T=2.0, alpha=0.7
│   └── distill_climbmix.yaml   # KL distillation with ClimbMix dataset
│
├── scripts/                    # Standalone scripts (can also use CLI)
│   ├── run_conversion.py       # Convert transformer -> Aetheris Mamba-MoE
│   ├── run_baseline.py         # Run teacher eval (mGSM + XCOPA + throughput)
│   ├── run_distill.py          # Run distillation stages 1/2/3
│   ├── run_climbmix_distill.py # Quick-start ClimbMix distillation
│   ├── profile_tiny_aya.py     # Profile Aya resource usage per language
│   └── generate_training_data.py # Generate multilingual SFT data
│
├── notebooks/                  # Research notebooks
│   ├── 01_explore_aya_teacher.ipynb   # Load tiny-aya-global, tokenizer analysis
│   ├── 02_cka_analysis.ipynb          # CKA tutorial, noise sensitivity, heatmaps
│   └── 03_block_conversion_experiments.ipynb  # Attention->SSM conversion
│
├── tests/
│   ├── conftest.py             # Pytest fixtures for model testing
│   └── test_package.py         # 209 tests: languages, CKA, eval, checklist, reasoning
│
├── docs/                       # Project documentation
│   ├── training_data_catalog.md
│   ├── Project Aetheris Research Guide.pdf
│   ├── Project_Aya_Team_Doc.docx.pdf
│   └── Project-Aya-Research-Scope.docx
│
├── results/                    # Output artifacts (gitignored except .json)
└── data/                       # Generated/processed data
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- CUDA GPU recommended (tiny-aya-global loads on 4GB VRAM without quantization)

### Run Tests

```bash
pytest tests/ -v
# 209 tests covering: language registry, CKA, eval metrics, checklist (30 items),
# multilingual testing, tool calling, reasoning evaluation, CoT scoring
```

### Run Research Notebooks

```bash
uv pip install -e ".[notebooks]"
jupyter lab notebooks/
```

### Run Teacher Baseline Evaluation

```bash
aya-distill eval --config configs/eval_baseline.yaml
```

### Run Distillation

```bash
aya-distill distill --stage 1 --config configs/distill_stage1.yaml
aya-distill distill --stage 2 --config configs/distill_stage2.yaml
aya-distill distill --stage 3
```

### Model Conversion

```bash
aya-distill convert --strategy weight_map --output checkpoints/converted/
aya-distill convert --strategy hybrid --refinement-steps 1000
```

### Profile Teacher Model

```bash
aya-distill profile --quantize 4bit --output results/profile.json
```

---

## Roadmap

| Phase | Name | Key Deliverables | Target |
|-------|------|-----------------|--------|
| 01 | Baseline | Load tiny-aya-global, establish CPU tokens/sec across 67 languages. Validate tokenizer compatibility. | Q1 2026 |
| 02 | Distillation | Execute 3-stage MambaInLlama pipeline on RunPod A100. Layer alignment, KL distillation, then multilingual SFT with tool calling recovery. | Q2 2026 |
| 03 | Evaluation | Full benchmark suite across 67 languages. Equity analysis, ablation studies, and paper preparation targeting EMNLP / ACL / NeurIPS. | Q3 2026 |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

**Branching strategy:** Everyone works on personal branches (`<name>/<feature>`), submits PRs to `dev`, and `dev` merges into `main` at milestones.

**Commit style:** [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `eval:`, `train:`, `data:`, `docs:`

---

## References

- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Wang et al., "The Mamba in the Llama" (2024)
- Diao et al., "CLIMB: CLustering-based Iterative Data Mixture Bootstrapping" (2025) — [arXiv:2504.13161](https://arxiv.org/abs/2504.13161)
- Kreutzer et al., "Deja Vu: Multilingual LLM Evaluation" (2025) — [arXiv:2504.11829](https://arxiv.org/abs/2504.11829)
- Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019) — CKA
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) — KL distillation

## License

MIT

---

*Wayy Research — Buffalo, NY — Est. 2024*
*People for research, research for people.*
