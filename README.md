# Project Aya

**Compression, Equity, and the Architecture of Linguistic Inclusion**

Cohere Labs x Wayy Research | Buffalo, NY | Est. 2026 | Open Science / Open Weights

---

## Overview

Most efficient AI models are English-centric. The architecture of language technology is leaving billions behind. Project Aya investigates whether transformer-based multilingual models can be distilled into Mamba architectures while preserving multilingual capability, structured tool use, and cross-lingual reasoning — and whether compression degrades uniformly across language families.

### Core Research Question

> *Can transformer-based multilingual models be distilled into Mamba architectures while preserving multilingual capability, structured tool use, and cross-lingual reasoning — and does the compression degrade uniformly across language families?*

### Architecture: Aetheris (Hybrid Mamba-MoE)

A novel student architecture combining selective state spaces with sparse mixture-of-experts, distilled from Cohere's Aya Expanse (8B params, 23 languages) into Aetheris (~500-800M params, Mamba-MoE).

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

## Package: `aya-distill`

Install as a Python package for programmatic access to all distillation, evaluation, and testing tools:

```python
from aya_distill.distill.cka import linear_cka
from aya_distill.eval.metrics import degradation_equity_score
from aya_distill.testing import MultilingualTestSuite, ToolCallingTest
from aya_distill.languages import LANGUAGES
```

CLI entry points:

```bash
aya-distill convert --strategy weight_map --output checkpoints/converted/
aya-distill distill --stage 1 --config configs/distill_stage1.yaml
aya-distill eval --config configs/eval_baseline.yaml
aya-distill profile --quantize 4bit
```

## Repository Structure

```
project-aya/
├── README.md
├── CONTRIBUTING.md             # Branching strategy, commit conventions, PR workflow
├── LICENSE                     # MIT
├── pyproject.toml              # aya-distill package config (hatchling, src-layout)
│
├── src/aya_distill/            # pip-installable package
│   ├── __init__.py             # Package root (version, top-level imports)
│   ├── cli.py                  # Unified CLI: aya-distill, aya-convert, aya-eval
│   ├── languages.py            # 10 target languages with metadata (family, script, typology)
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
│   │   ├── checklist.py        # Cohere multilingual eval checklist validator (22 items)
│   │   └── prompts/            # Native-language few-shot templates
│   │       ├── mgsm.py         # 8-shot mGSM prompts for 10 languages
│   │       └── xcopa.py        # 4-shot XCOPA prompts for 10 languages
│   │
│   └── testing/                # Multilingual test harness (extends testkitLLM)
│       ├── multilingual.py     # MultilingualTestSuite — per-language quality + equity
│       ├── tool_testing.py     # ToolCallingTest — JSON tool call verification
│       ├── fixtures.py         # Prompts, tool schemas, scenarios in 10 languages
│       └── conftest.py         # Pytest fixtures for model testing
│
├── configs/
│   ├── student.yaml            # Aetheris student model (1024d, 24L, 4 experts)
│   ├── eval_baseline.yaml      # Teacher baseline benchmark config
│   ├── distill_stage1.yaml     # Layer alignment: 10k steps, CKA threshold 0.75
│   ├── distill_stage2.yaml     # KL distillation: 20k steps, T=2.0, alpha=0.7
│   └── distill_climbmix.yaml   # KL distillation with ClimbMix dataset
│
├── scripts/                    # Standalone scripts (can also use CLI)
│   ├── run_conversion.py       # Convert Aya transformer → Aetheris Mamba-MoE
│   ├── run_baseline.py         # Run teacher eval (mGSM + XCOPA + throughput)
│   ├── run_distill.py          # Run distillation stages 1/2/3
│   ├── run_climbmix_distill.py # Quick-start ClimbMix distillation
│   ├── profile_tiny_aya.py     # Profile Aya resource usage per language
│   └── generate_training_data.py # Generate multilingual SFT data
│
├── notebooks/                  # Research notebooks (run these first!)
│   ├── 01_explore_aya_teacher.ipynb   # Load Aya, tokenizer analysis, inference profiling
│   ├── 02_cka_analysis.ipynb          # CKA tutorial, noise sensitivity, mini-batch, heatmaps
│   └── 03_block_conversion_experiments.ipynb  # Attention→SSM conversion on real weights
│
├── docs/                       # Project documentation
│   ├── training_data_catalog.md        # All available training data resources
│   ├── Project Aetheris Research Guide.pdf
│   ├── Project_Aya_Team_Doc.docx.pdf
│   └── Project-Aya-Research-Scope.docx
│
├── tests/                      # Test suite
│   └── test_package.py         # Smoke tests for package imports, CKA, testing harness
├── results/                    # Output artifacts (gitignored except .json)
└── data/                       # Generated/processed data
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- CUDA GPU recommended (tested on RTX 3050 Ti 4GB with 4-bit quantization)
- HuggingFace account with access to [CohereForAI/aya-expanse-8b](https://huggingface.co/CohereForAI/aya-expanse-8b) (gated model — accept license first)

### Setup

```bash
git clone https://github.com/Wayy-Research/project-aya.git
cd project-aya

# Create your personal branch from dev
git checkout dev
git checkout -b <your-name>/feature-description

# Set up environment with uv
uv venv --python 3.10
source .venv/bin/activate

# Install aya-distill package (pick one):
uv pip install -e "."              # Core only
uv pip install -e ".[notebooks]"   # + Jupyter, matplotlib, seaborn, plotly
uv pip install -e ".[quantize]"    # + bitsandbytes for 4-bit quantization
uv pip install -e ".[testing]"     # + testkitLLM for multilingual semantic testing
uv pip install -e ".[all]"         # Everything (dev + notebooks + quantize + testing)

# Login to HuggingFace (required for gated model access)
huggingface-cli login
```

> **Note**: If you have system-wide PyTorch with CUDA already installed, use
> `uv venv --python 3.10 --system-site-packages` to reuse it.

### Run Research Notebooks

```bash
# Start Jupyter
jupyter lab notebooks/

# Notebooks:
# 01_explore_aya_teacher.ipynb  — Load Aya, profile tokenizer + inference per language
# 02_cka_analysis.ipynb         — CKA tutorial: noise sensitivity, mini-batch, permutation tests
# 03_block_conversion_experiments.ipynb — Attention→SSM conversion on real weights
```

### Run Teacher Baseline Evaluation

Establish the scores the student model must match:

```bash
python scripts/run_baseline.py --config configs/eval_baseline.yaml
```

Runs mGSM (8-shot math) and XCOPA (4-shot reasoning) across 10 languages with bootstrap confidence intervals, per-language reporting, and Cohere eval checklist validation.

### Run Distillation

```bash
# Stage 1: Layer alignment with CKA monitoring
python scripts/run_distill.py --stage 1 --config configs/distill_stage1.yaml

# Stage 2: KL distillation
python scripts/run_distill.py --stage 2 --config configs/distill_stage2.yaml

# Stage 2 (with ClimbMix dataset):
python scripts/run_climbmix_distill.py --mode retokenize

# Stage 3: Supervised fine-tuning
python scripts/run_distill.py --stage 3
```

### Model Conversion (Weight Map)

Convert Aya transformer blocks directly to Aetheris Mamba-MoE:

```bash
# Direct weight mapping (no data needed, fast)
python scripts/run_conversion.py --strategy weight_map --output checkpoints/converted/

# Hybrid (weight map + refinement distillation)
python scripts/run_conversion.py --strategy hybrid --refinement-steps 1000
```

### Profile Teacher Model

```bash
python scripts/profile_tiny_aya.py --quantize 4bit --output results/profile.json
```

### Generate Multilingual Training Data

```bash
python scripts/generate_training_data.py --quantize 4bit --output data/sft/
```

### Multilingual Testing

Test any model's multilingual quality and tool calling ability:

```python
from aya_distill.testing import MultilingualTestSuite, ToolCallingTest

# Wrap your model as a callable
def my_model(prompt: str) -> str:
    return model.generate(prompt)

# Run quality tests across 10 languages
suite = MultilingualTestSuite(model_fn=my_model)
result = suite.run()
print(result.summary())
# Shows per-language coherence, relevance, cross-lingual consistency, equity score

# Run tool calling tests
tool_test = ToolCallingTest(model_fn=my_model)
tool_results = tool_test.run_multilingual()
# Tests JSON generation + function selection in all 10 languages
```

With pytest:

```bash
pytest tests/ -v
```

### Dry Run (Verify Data Loading)

```bash
python scripts/run_climbmix_distill.py --dry-run
```

---

## Evaluation

### Benchmarks

| Benchmark | Description |
|-----------|-------------|
| **mGSM** | Multilingual grade school math across 10 languages, 250 problems each |
| **XCOPA** | Cross-lingual commonsense reasoning across 11 languages |
| **Tool Calling** | Structured JSON generation and function selection in 10+ languages |
| **CPU Throughput** | Tokens/sec, peak memory, time-to-first-token on consumer hardware |

### Languages: 10 Languages, 5 Typologies, 8 Families

| Language | Family | Script | Typology |
|----------|--------|--------|----------|
| English | Indo-European | Latin | Analytic |
| Spanish | Indo-European | Latin | Fusional |
| Hindi | Indo-European | Devanagari | Fusional |
| Mandarin | Sino-Tibetan | CJK | Analytic/Isolating |
| Arabic | Afroasiatic | Arabic | Fusional/Semitic |
| Swahili | Niger-Congo | Latin | Agglutinative |
| Turkish | Turkic | Latin | Agglutinative |
| Japanese | Japonic | CJK+Kana | Agglutinative |
| Indonesian | Austronesian | Latin | Analytic |
| Telugu | Dravidian | Telugu | Agglutinative |

### Degradation Equity Score

Key novel metric: the variance of accuracy drop across language families after distillation. Measures whether compression picks winners and losers. Lower is better — compression should not pick winners.

### Cohere Multilingual Eval Checklist

All evaluations are validated against the [Cohere multilingual LLM evaluation checklist](https://github.com/Cohere-Labs/multilingual-llm-evaluation-checklist) (22 items, 7 categories). Per-language scores are always primary — never hidden behind averages.

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

## Roadmap

| Phase | Name | Key Deliverables | Target |
|-------|------|-----------------|--------|
| 01 | Baseline | Load Tiny Aya, establish CPU tokens/sec in 5 languages. Validate tokenizer compatibility. | Q1 2026 |
| 02 | Distillation | Execute 3-stage MambaInLlama pipeline on RunPod A100. Layer alignment, KL distillation, then multilingual SFT with tool calling recovery. | Q2 2026 |
| 03 | Evaluation | Full benchmark suite across 10 languages. Equity analysis, ablation studies, and paper preparation targeting EMNLP / ACL / NeurIPS. | Q3 2026 |

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
