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
| Hybrid Design | Delta | SSM blocks on odd layers, MoE on even layers. 24 total layers with weight-tied embeddings and gradient checkpointing |

### 3-Stage MambaInLlama Pipeline

1. **Layer Alignment** — Map transformer attention layers to Mamba SSM blocks with structural correspondence
2. **KL Distillation** — Soft-target training with KL divergence to transfer knowledge from teacher to student
3. **Supervised Fine-Tuning** — Restore multilingual capability and structured tool calling via targeted SFT

### Evaluation Scope: 10 Languages, 5 Families

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

### Benchmarks

| Benchmark | Description |
|-----------|-------------|
| mGSM | Multilingual grade school math across 10 languages, 250 problems each |
| XCOPA | Cross-lingual commonsense reasoning across 11 languages |
| Tool Calling | Structured JSON generation and function selection in 10+ languages |
| CPU Throughput | Tokens/sec, peak memory, time-to-first-token on consumer hardware |

**Degradation Equity Score** — Key novel metric: the variance of accuracy drop across language families after distillation. Lower is better.

### Technology Stack

| ML / Training | Infrastructure |
|---------------|---------------|
| PyTorch | Docker + GitHub Actions |
| HuggingFace Transformers | RunPod A100 |
| HuggingFace Datasets | FastAPI + SSE Streaming |
| Mamba SSM | llama.cpp (edge deployment) |
| CUDA | Pydantic |

### Roadmap

| Phase | Name | Key Deliverables | Target |
|-------|------|-----------------|--------|
| 01 | Baseline | Load Tiny Aya, establish CPU tokens/sec in 5 languages. Validate tokenizer compatibility. | Q1 2026 |
| 02 | Distillation | Execute 3-stage MambaInLlama pipeline on RunPod A100. Layer alignment, KL distillation, then multilingual SFT with tool calling recovery. | Q2 2026 |
| 03 | Evaluation | Full benchmark suite across 10 languages. Equity analysis, ablation studies, and paper preparation targeting EMNLP / ACL / NeurIPS. | Q3 2026 |

---

## Repository Structure

```
project-aya/
├── README.md              # This file
├── CONTRIBUTING.md        # Contribution guidelines (read this first!)
├── LICENSE
├── aetheris/              # Core model architecture (submodule or code)
├── configs/               # Training and evaluation configs
├── data/                  # Data processing scripts and manifests
├── eval/                  # Evaluation harness and benchmark runners
├── notebooks/             # Exploration and analysis notebooks
├── scripts/               # Training, distillation, and utility scripts
├── docs/                  # Extended documentation and paper drafts
└── tests/                 # Test suite
```

## Quick Start

```bash
git clone https://github.com/Wayy-Research/project-aya.git
cd project-aya
git checkout dev

# Create your personal branch from dev
git checkout -b <your-name>/feature-description

# Set up environment
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

---

## License

MIT

---

*Wayy Research — Buffalo, NY — Est. 2024*
*People for research, research for people.*
