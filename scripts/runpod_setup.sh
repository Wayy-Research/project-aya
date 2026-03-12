#!/bin/bash
# RunPod environment setup for Project Aya distillation
#
# This script bootstraps a RunPod pod with everything needed to run
# the full 3-stage distillation pipeline.
#
# Expected base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Usage:
#   # After SSH into RunPod:
#   bash /workspace/aya/project-aya/scripts/runpod_setup.sh
#
# Wayy Research, 2024-2026

set -euo pipefail

echo "============================================"
echo "  Project Aya — RunPod Setup"
echo "============================================"

WORKSPACE="/workspace/aya"

# 0. Clone repos if not already present
echo "[0/7] Cloning repositories..."
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

if [ ! -d "project-aya" ]; then
    git clone -b dev https://github.com/Wayy-Research/project-aya.git
else
    echo "  project-aya already exists, pulling latest..."
    cd project-aya && git pull origin dev && cd ..
fi

if [ ! -d "aetheris" ]; then
    git clone https://github.com/Wayy-Research/aetheris.git
else
    echo "  aetheris already exists, pulling latest..."
    cd aetheris && git pull origin main && cd ..
fi

# 1. Install uv for fast Python package management
echo "[1/6] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Create virtual environment
echo "[2/6] Setting up Python environment..."
cd "$WORKSPACE/project-aya"
if [ ! -d ".venv" ]; then
    uv venv --python 3.11
fi
source .venv/bin/activate

# 3. Install packages
echo "[3/6] Installing aya-distill..."
uv pip install -e ".[dev]"

echo "[3b/6] Installing aetheris..."
cd "$WORKSPACE/aetheris"
uv pip install -e ".[cuda,data]"

# 4. Install additional dependencies
echo "[4/6] Installing additional training deps..."
uv pip install wandb tensorboard

# 5. Pre-download models
echo "[5/6] Pre-downloading teacher model and tokenizer..."
cd "$WORKSPACE/project-aya"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Downloading tokenizer...')
tok = AutoTokenizer.from_pretrained('CohereForAI/aya-expanse-8b', trust_remote_code=True)
print(f'Tokenizer vocab: {len(tok)}')

print('Downloading teacher model...')
model = AutoModelForCausalLM.from_pretrained(
    'CohereForAI/aya-expanse-8b',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print(f'Teacher: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params')
del model
torch.cuda.empty_cache()
print('Done!')
"

# 6. Verify GPU and VRAM
echo "[6/6] Verifying GPU setup..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1e9:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
"

# Create checkpoints directory
mkdir -p "$WORKSPACE/project-aya/checkpoints"
mkdir -p "$WORKSPACE/project-aya/results/runpod"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To start training:"
echo "    cd $WORKSPACE/project-aya"
echo "    source .venv/bin/activate"
echo "    python scripts/run_full_distill.py --config configs/runpod_distill.yaml"
echo ""
echo "  To run a single stage:"
echo "    python scripts/run_full_distill.py --config configs/runpod_distill.yaml --stage 2"
echo ""
echo "  To monitor GPU:"
echo "    watch -n1 nvidia-smi"
echo "============================================"
