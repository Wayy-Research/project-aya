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

### From GitHub (recommended)

```bash
pip install "aya-distill @ git+https://github.com/Wayy-Research/project-aya.git"
```

### For development

```bash
git clone https://github.com/Wayy-Research/project-aya.git
cd project-aya

# Create environment
uv venv --python 3.10
source .venv/bin/activate

# Install (pick one):
uv pip install -e "."              # Core only
uv pip install -e ".[notebooks]"   # + Jupyter, matplotlib, seaborn, plotly
uv pip install -e ".[quantize]"    # + bitsandbytes for 4-bit quantization
uv pip install -e ".[testing]"     # + testkitLLM for semantic testing
uv pip install -e ".[all]"         # Everything
```

> **Note**: If you have system-wide PyTorch with CUDA already installed, use
> `uv venv --python 3.10 --system-site-packages` to reuse it.

---

## Package: `aya-distill`

### Python API

```python
from aya_distill.languages import LANGUAGES, LANGUAGE_FAMILIES, RESOURCE_GROUPS
from aya_distill.distill.cka import linear_cka, rbf_cka, minibatch_cka
from aya_distill.eval.metrics import degradation_equity_score, bootstrap_ci
from aya_distill.eval.checklist import validate_results, CHECKLIST_ITEMS
from aya_distill.testing import MultilingualTestSuite, ToolCallingTest
```

### CLI

```bash
aya-distill convert --strategy weight_map --output checkpoints/converted/
aya-distill distill --stage 1 --config configs/distill_stage1.yaml
aya-distill eval --config configs/eval_baseline.yaml
aya-distill profile --quantize 4bit
```

### Multilingual Testing (67 languages)

Test any model's multilingual quality and tool calling ability:

```python
from aya_distill.testing import MultilingualTestSuite, ToolCallingTest

# Wrap your model as a callable
def my_model(prompt: str) -> str:
    return model.generate(prompt)

# Run quality tests across 67 languages
suite = MultilingualTestSuite(model_fn=my_model)
result = suite.run()
print(result.summary())
# Per-language coherence, relevance, cross-lingual consistency, equity score

# Run tool calling tests across 67 languages
tool_test = ToolCallingTest(model_fn=my_model)
tool_results = tool_test.run_multilingual()
print(tool_test.summary(tool_results))
# JSON generation + function selection in all 67 languages

# Compare student against teacher
degradation = suite.compare(teacher_fn=teacher_model)
# Per-language coherence/relevance deltas
```

### Language Registry

Single source of truth for all 67 target languages:

```python
from aya_distill.languages import (
    LANGUAGES,           # 67 Language dataclasses
    LANGUAGE_FAMILIES,   # 13 families
    SCRIPT_GROUPS,       # 19 scripts
    RESOURCE_GROUPS,     # high / medium / low
    REGION_GROUPS,       # 6 regions
)

# Filter by family
dravidian = LANGUAGE_FAMILIES["Dravidian"]
# ['te', 'ta', 'ml', 'kn']

# Filter by resource level
low_resource = RESOURCE_GROUPS["low"]
# ['am', 'ny', 'sn', 'xh', 'zu', 'yo', 'ig', 'ha', 'so', ...]
```

### Evaluation Checklist

Validates results against the [Cohere Deja Vu multilingual LLM evaluation checklist](https://github.com/Cohere-Labs/multilingual-llm-evaluation-checklist) (25 items, 7 categories):

```python
from aya_distill.eval.checklist import validate_results

report = validate_results(eval_config={...}, results={...})
print(report.summary())
# Category-by-category pass/fail with recommendations
```

---

## Languages: 67 Languages, 13 Families, 19 Scripts

| Family | Languages |
|--------|-----------|
| **Indo-European** | English, Spanish, French, Portuguese, German, Italian, Dutch, Polish, Romanian, Ukrainian, Czech, Greek, Hindi, Bengali, Marathi, Gujarati, Nepali, Sinhala, Persian |
| **Afroasiatic** | Arabic, Hebrew, Amharic, Somali, Hausa |
| **Niger-Congo** | Swahili, Yoruba, Igbo, Zulu, Xhosa, Shona, Chichewa |
| **Turkic** | Turkish, Azerbaijani, Uzbek, Kazakh, Kyrgyz |
| **Sino-Tibetan** | Mandarin, Burmese |
| **Japonic** | Japanese |
| **Koreanic** | Korean |
| **Dravidian** | Telugu, Tamil, Malayalam, Kannada |
| **Austronesian** | Indonesian, Malay, Filipino, Malagasy, Cebuano |
| **Austroasiatic** | Vietnamese, Khmer |
| **Tai-Kadai** | Thai, Lao |
| **Kartvelian** | Georgian |
| **Uralic** | Finnish, Hungarian, Estonian |

### Resource Levels

- **High** (18): en, es, fr, pt, de, it, nl, pl, zh, ja, ko, ar, hi, bn, tr, vi, id, th
- **Medium** (23): ro, uk, cs, el, he, fa, mr, gu, ne, si, sw, ta, ml, kn, ms, tl, fi, hu, et, az, uz, kk, km
- **Low** (26): am, ny, sn, xh, zu, yo, ig, ha, so, my, lo, ka, mg, ceb, ky, te

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
│   ├── cli.py                  # Unified CLI: aya-distill, aya-convert, aya-eval
│   ├── languages.py            # 67 target languages with metadata (family, script, typology, region, resourcedness)
│   │
│   ├── distill/                # 3-stage distillation pipeline
│   │   ├── converter.py        # Core: attention→SSM and FFN→MoE weight mapping
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
│   │   ├── metrics.py          # Bootstrap CIs, Degradation Equity Score, family aggregation
│   │   ├── throughput.py       # Tokens/sec, TTFT, peak memory benchmarking
│   │   ├── checklist.py        # Cohere Deja Vu eval checklist validator (25 items, 7 categories)
│   │   └── prompts/            # Native-language few-shot templates
│   │       ├── mgsm.py         # 8-shot mGSM prompts for 67 languages
│   │       └── xcopa.py        # 4-shot XCOPA prompts for 67 languages
│   │
│   └── testing/                # Multilingual test harness (extends testkitLLM)
│       ├── multilingual.py     # MultilingualTestSuite — per-language quality + equity
│       ├── tool_testing.py     # ToolCallingTest — JSON tool call verification
│       └── fixtures.py         # Prompts, tool schemas, scenarios in 67 languages
│
├── configs/
│   ├── student.yaml            # Aetheris student model (1024d, 24L, 4 experts)
│   ├── eval_baseline.yaml      # Teacher baseline benchmark config
│   ├── distill_stage1.yaml     # Layer alignment: 10k steps, CKA threshold 0.75
│   ├── distill_stage2.yaml     # KL distillation: 20k steps, T=2.0, alpha=0.7
│   └── distill_climbmix.yaml   # KL distillation with ClimbMix dataset
│
├── scripts/                    # Standalone scripts (can also use CLI)
│   ├── run_conversion.py       # Convert transformer → Aetheris Mamba-MoE
│   ├── run_baseline.py         # Run teacher eval (mGSM + XCOPA + throughput)
│   ├── run_distill.py          # Run distillation stages 1/2/3
│   ├── run_climbmix_distill.py # Quick-start ClimbMix distillation
│   ├── profile_tiny_aya.py     # Profile Aya resource usage per language
│   └── generate_training_data.py # Generate multilingual SFT data
│
├── notebooks/                  # Research notebooks
│   ├── 01_explore_aya_teacher.ipynb   # Load tiny-aya-global, tokenizer analysis, inference profiling
│   ├── 02_cka_analysis.ipynb          # CKA tutorial, noise sensitivity, mini-batch, heatmaps
│   └── 03_block_conversion_experiments.ipynb  # Attention→SSM conversion experiments
│
├── tests/
│   ├── conftest.py             # Pytest fixtures for model testing
│   └── test_package.py         # 164 tests: languages, CKA, eval, checklist, testing harness
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

## Evaluation

### Benchmarks

| Benchmark | Description |
|-----------|-------------|
| **mGSM** | Multilingual grade school math across 67 languages, 250 problems each |
| **XCOPA** | Cross-lingual commonsense reasoning |
| **Tool Calling** | Structured JSON generation and function selection in 67 languages |
| **CPU Throughput** | Tokens/sec, peak memory, time-to-first-token on consumer hardware |

### Degradation Equity Score

Key novel metric: the variance of accuracy drop across language families after distillation. Measures whether compression picks winners and losers. Lower is better — compression should not pick winners.

### Cohere Deja Vu Checklist

All evaluations are validated against the [Cohere multilingual LLM evaluation checklist](https://github.com/Cohere-Labs/multilingual-llm-evaluation-checklist) (25 items, 7 categories). Per-language scores are always primary — never hidden behind averages.

Categories: Data & Sampling, Metrics & Aggregation, Cultural & Linguistic Adequacy, Human Evaluation, Reproducibility & Documentation, Bias & Fairness, Robustness & Generalization.

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

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- CUDA GPU recommended (tiny-aya-global loads on 4GB VRAM without quantization)

### Run Tests

```bash
pytest tests/ -v
# 164 tests covering: language registry, CKA, eval metrics, checklist, multilingual testing, tool calling
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
