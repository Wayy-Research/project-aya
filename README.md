# Project Aya

**Compression, Equity, and the Architecture of Linguistic Inclusion**

Cohere Labs x Wayy Research | Buffalo, NY | Est. 2026 | Open Science / Open Weights

---

## Overview

Most efficient AI models are English-centric. The architecture of language technology is leaving billions behind. Project Aya investigates whether transformer-based multilingual models can be distilled into Mamba architectures while preserving multilingual capability, structured tool use, and cross-lingual reasoning — and whether compression degrades uniformly across language families.

### Core Research Question

> *Can transformer-based multilingual models be distilled into Mamba architectures while preserving multilingual capability, structured tool use, and cross-lingual reasoning — and does the compression degrade uniformly across language families?*

### Architecture: Aetheris (Hybrid Mamba-MoE)

A novel student architecture combining selective state spaces with sparse mixture-of-experts, distilled from Cohere's Tiny Aya (3.35B params, 70+ languages) into Aetheris (~500-800M params, Mamba-MoE).

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

## Repository Structure

```
project-aya/
├── README.md
├── CONTRIBUTING.md             # Branching strategy, commit conventions, PR workflow
├── LICENSE                     # MIT
├── requirements.txt            # PyTorch, Transformers, Datasets, etc.
│
├── eval/                       # Multilingual evaluation pipeline
│   ├── benchmarks.py           # mGSM + XCOPA harness (few-shot, per-language)
│   ├── metrics.py              # Bootstrap CIs, Degradation Equity Score, family aggregation
│   ├── throughput.py           # Tokens/sec, TTFT, peak memory benchmarking
│   ├── checklist.py            # Cohere multilingual eval checklist validator (22 items)
│   └── prompts/                # Native-language few-shot templates
│       ├── mgsm.py             # 8-shot mGSM prompts for 10 languages
│       └── xcopa.py            # 4-shot XCOPA prompts for 10 languages
│
├── distill/                    # 3-stage distillation pipeline
│   ├── alignment.py            # Stage 1: Layer alignment with CKA monitoring
│   ├── kl_distillation.py      # Stage 2: KL divergence with temperature scaling
│   ├── sft.py                  # Stage 3: Multilingual SFT with tool calling recovery
│   ├── cka.py                  # CKA module (linear, RBF, mini-batch, permutation test)
│   ├── hooks.py                # Activation extraction for teacher + student models
│   ├── data.py                 # Language-balanced multilingual data pipeline
│   └── climbmix.py             # NVIDIA ClimbMix 400B-token dataset loader
│
├── configs/
│   ├── eval_baseline.yaml      # Teacher baseline benchmark config
│   ├── student.yaml            # Aetheris student model (1024d, 24L, 4 experts)
│   ├── distill_stage1.yaml     # Layer alignment: 10k steps, CKA threshold 0.75
│   ├── distill_stage2.yaml     # KL distillation: 20k steps, T=2.0, alpha=0.7
│   └── distill_climbmix.yaml   # KL distillation with ClimbMix dataset
│
├── scripts/
│   ├── run_baseline.py         # Run teacher eval (mGSM + XCOPA + throughput)
│   ├── run_distill.py          # Run distillation stages 1/2/3
│   └── run_climbmix_distill.py # Quick-start ClimbMix distillation
│
├── docs/                       # Project documentation
│   ├── Project Aetheris Research Guide.pdf
│   ├── Project_Aya_Team_Doc.docx.pdf
│   ├── Project-Aya-Research-Scope.docx
│   ├── aetheris-presentation.html
│   └── ...
│
├── notebooks/                  # Exploration and analysis
├── data/                       # Data processing scripts
└── tests/                      # Test suite
```

---

## Quick Start

### Setup

```bash
git clone https://github.com/Wayy-Research/project-aya.git
cd project-aya

# Create your personal branch from dev
git checkout -b <your-name>/feature-description

# Set up environment
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
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
