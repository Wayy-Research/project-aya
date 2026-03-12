#!/usr/bin/env python3
"""Upload Aetheris checkpoints to HuggingFace Hub.

Uploads model checkpoints with training metadata, config, and quality
metrics to wayyresearch/aetheris on the Hub.

Usage:
    # Upload latest best checkpoint from current stage
    python scripts/upload_checkpoint.py --stage 1

    # Upload a specific checkpoint file
    python scripts/upload_checkpoint.py --checkpoint checkpoints/stage1_alignment/best.pt

    # Upload with a custom commit message
    python scripts/upload_checkpoint.py --stage 1 --message "Stage 1 step 5000"

Wayy Research, 2024-2026.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, upload_folder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "wayyresearch/aetheris"

STAGE_DIRS = {
    1: "checkpoints/stage1_alignment",
    2: "checkpoints/stage2_kl",
    3: "checkpoints/stage3_sft",
}

STAGE_NAMES = {
    1: "layer-alignment",
    2: "kl-distillation",
    3: "sft",
}


def load_checkpoint_metadata(ckpt_path: Path) -> dict:
    """Extract training metadata from a checkpoint without loading full weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    meta = {
        "step": ckpt.get("step", "unknown"),
        "checkpoint_file": ckpt_path.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Stage 1 metadata
    if "cka_history" in ckpt:
        cka = ckpt["cka_history"]
        if cka:
            last = cka[-1] if isinstance(cka[-1], dict) else {}
            meta["cka_mean"] = last.get("mean", None)
            meta["cka_min"] = last.get("min", None)

    if "loss_history" in ckpt:
        losses = ckpt["loss_history"]
        if losses:
            meta["final_loss"] = losses[-1]
            meta["best_loss"] = min(losses)

    # Stage 2 metadata
    if "per_language_kl" in ckpt:
        lang_kl = ckpt["per_language_kl"]
        if lang_kl:
            means = {}
            for lang, scores in lang_kl.items():
                if scores:
                    recent = scores[-100:]
                    means[lang] = sum(recent) / len(recent)
            meta["per_language_kl_mean"] = means
            if means:
                meta["kl_mean"] = sum(means.values()) / len(means)
                meta["kl_max"] = max(means.values())
                meta["kl_min"] = min(means.values())

    if "config" in ckpt:
        meta["training_config"] = ckpt["config"]

    return meta


def prepare_upload_dir(ckpt_path: Path, stage: int, config_path: str | None = None) -> Path:
    """Prepare a directory with checkpoint + metadata for upload."""
    upload_dir = Path(f"/tmp/hf_upload_{stage}")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint
    import shutil
    dest_ckpt = upload_dir / f"stage{stage}_checkpoint.pt"
    shutil.copy2(ckpt_path, dest_ckpt)
    logger.info(f"Copied checkpoint: {ckpt_path} -> {dest_ckpt}")

    # Extract and save metadata
    meta = load_checkpoint_metadata(ckpt_path)
    meta["stage"] = stage
    meta["stage_name"] = STAGE_NAMES.get(stage, f"stage{stage}")

    meta_path = upload_dir / f"stage{stage}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Saved metadata: {meta_path}")

    # Copy student config if available
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, upload_dir / "student_config.yaml")

    # Copy training config
    runpod_config = Path("configs/runpod_distill.yaml")
    if runpod_config.exists():
        shutil.copy2(runpod_config, upload_dir / "training_config.yaml")

    # Generate model card
    model_card = generate_model_card(meta, stage)
    (upload_dir / "README.md").write_text(model_card)

    return upload_dir


def generate_model_card(meta: dict, stage: int) -> str:
    """Generate a HuggingFace model card."""
    stage_name = STAGE_NAMES.get(stage, f"stage{stage}")
    step = meta.get("step", "?")

    card = f"""---
language:
  - multilingual
  - en
  - es
  - hi
  - zh
  - ar
  - sw
  - tr
  - ja
  - id
  - te
license: apache-2.0
tags:
  - mamba
  - moe
  - ssm
  - multilingual
  - distillation
  - aya
library_name: aetheris
pipeline_tag: text-generation
---

# Aetheris — Hybrid Mamba-MoE Multilingual Model

**Aetheris** is a ~800M parameter hybrid SSM/MoE language model distilled from
[CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B).

Built by [Wayy Research](https://github.com/Wayy-Research).

## Architecture

- **Type**: Hybrid Mamba (SSM) + Mixture of Experts (MoE)
- **Layers**: 24 (interleaved: even=SSM, odd=MoE)
- **Hidden dim**: 1024
- **Experts**: 4 per MoE layer, top-1 routing
- **SSM state dim**: 16
- **Vocab size**: 256,000 (shared with tiny-aya-global)
- **Parameters**: ~800M

## Training

3-stage MambaInLlama distillation pipeline:

| Stage | Method | Data | Steps |
|-------|--------|------|-------|
| 1 | CKA-guided Layer Alignment | ClimbMix | 10,000 |
| 2 | KL Distillation (T=2.0, alpha=0.7) | ClimbMix | 20,000 |
| 3 | Supervised Fine-Tuning | aya_collection | 5,000 |

Key research findings applied:
- SSM 10x LR boost (compensates 27x gradient imbalance)
- SVD split for MoE expert initialization (CKA=0.097 diversity)
- Per-language KL tracking for multilingual equity

## Current Checkpoint

- **Stage**: {stage} ({stage_name})
- **Step**: {step}
"""

    if "final_loss" in meta:
        card += f"- **Loss**: {meta['final_loss']:.4f}\n"
    if "cka_mean" in meta and meta["cka_mean"] is not None:
        card += f"- **CKA mean**: {meta['cka_mean']:.4f}\n"
    if "kl_mean" in meta:
        card += f"- **KL mean**: {meta['kl_mean']:.4f}\n"

    card += f"- **Updated**: {meta.get('timestamp', 'unknown')}\n"

    card += """
## Languages

Supports 70+ languages inherited from tiny-aya-global. Core evaluation
languages: English, Spanish, Hindi, Chinese, Arabic, Swahili, Turkish,
Japanese, Indonesian, Telugu.

## Citation

```bibtex
@misc{aetheris2026,
  title={Aetheris: Hybrid Mamba-MoE Multilingual Model via Knowledge Distillation},
  author={Wayy Research},
  year={2026},
  url={https://huggingface.co/wayyresearch/aetheris}
}
```
"""
    return card


def upload_to_hub(
    upload_dir: Path,
    stage: int,
    commit_message: str | None = None,
) -> str:
    """Upload prepared directory to HuggingFace Hub."""
    api = HfApi()

    if commit_message is None:
        commit_message = f"Stage {stage} ({STAGE_NAMES.get(stage, '')}) checkpoint update"

    # Upload to a stage-specific subfolder + root README
    url = api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=commit_message,
    )

    logger.info(f"Uploaded to {REPO_ID}: {url}")
    return url


def main():
    parser = argparse.ArgumentParser(description="Upload Aetheris checkpoint to HuggingFace")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Training stage")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint file path")
    parser.add_argument("--message", type=str, help="Custom commit message")
    parser.add_argument("--config", type=str, default="configs/student.yaml", help="Student config path")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        stage = args.stage or 1
    elif args.stage:
        stage_dir = Path(STAGE_DIRS[args.stage])
        best = stage_dir / "best.pt"
        if best.exists():
            ckpt_path = best
        else:
            # Find latest step checkpoint
            pts = sorted(stage_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
            if not pts:
                logger.error(f"No checkpoints found in {stage_dir}")
                sys.exit(1)
            ckpt_path = pts[-1]
        stage = args.stage
    else:
        logger.error("Must specify --stage or --checkpoint")
        sys.exit(1)

    logger.info(f"Uploading checkpoint: {ckpt_path} (Stage {stage})")

    upload_dir = prepare_upload_dir(ckpt_path, stage, args.config)
    upload_to_hub(upload_dir, stage, args.message)

    logger.info("Done!")


if __name__ == "__main__":
    main()
