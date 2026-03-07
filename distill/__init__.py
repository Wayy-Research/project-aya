"""
Project Aya Distillation Pipeline
==================================

3-stage MambaInLlama distillation from Cohere Aya-Expanse-8B (teacher)
to Aetheris HybridMambaMoE (student):

    Stage 1: Layer Alignment — CKA-guided alignment of transformer layers
             to SSM/MoE blocks
    Stage 2: KL Distillation — Soft-target training with temperature-scaled
             KL divergence
    Stage 3: Supervised Fine-Tuning — Multilingual capability recovery with
             tool calling and chat format

Wayy Research, 2024-2026.
"""

from distill.cka import linear_cka, rbf_cka, minibatch_cka
from distill.alignment import LayerAlignmentTrainer
from distill.kl_distillation import KLDistillationTrainer
from distill.sft import SFTTrainer

__all__ = [
    "linear_cka",
    "rbf_cka",
    "minibatch_cka",
    "LayerAlignmentTrainer",
    "KLDistillationTrainer",
    "SFTTrainer",
]
