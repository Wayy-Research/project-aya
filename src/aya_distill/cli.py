"""Unified CLI for aya-distill."""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aya-distill",
        description="Multilingual evaluation and distillation toolkit — trains the Aetheris student model",
    )
    sub = parser.add_subparsers(dest="command")

    # aya-distill convert
    convert_p = sub.add_parser("convert", help="Convert Aya transformer to Aetheris Mamba-MoE")
    convert_p.add_argument("--strategy", choices=["weight_map", "hybrid"], default="weight_map")
    convert_p.add_argument("--output", default="checkpoints/converted/")
    convert_p.add_argument("--refinement-steps", type=int, default=0)
    convert_p.add_argument("--config", default="configs/student.yaml")

    # aya-distill distill
    distill_p = sub.add_parser("distill", help="Run distillation stages")
    distill_p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    distill_p.add_argument("--config", default=None)

    # aya-distill eval
    eval_p = sub.add_parser("eval", help="Run multilingual evaluation")
    eval_p.add_argument("--config", default="configs/eval_baseline.yaml")
    eval_p.add_argument("--model", default=None, help="Model path or HF name")

    # aya-distill profile
    profile_p = sub.add_parser("profile", help="Profile model resource usage")
    profile_p.add_argument("--device", default="auto")
    profile_p.add_argument("--quantize", choices=["4bit", "8bit"], default=None)
    profile_p.add_argument("--max-new-tokens", type=int, default=64)
    profile_p.add_argument("--output", default="results/profile_tiny_aya.json")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "convert":
        from aya_distill.distill.block_surgery import convert_model
        convert_model(
            teacher_name="CohereLabs/tiny-aya-global",
            student_config_path=args.config,
            output_dir=args.output,
        )
    elif args.command == "distill":
        if args.stage == 1:
            config_path = args.config or "configs/distill_stage1.yaml"
            # Import and run stage 1
            from aya_distill.distill.alignment import LayerAlignmentTrainer
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            trainer = LayerAlignmentTrainer(**cfg)
            trainer.train()
        elif args.stage == 2:
            config_path = args.config or "configs/distill_stage2.yaml"
            from aya_distill.distill.kl_distillation import KLDistillationTrainer
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            trainer = KLDistillationTrainer(**cfg)
            trainer.train()
        elif args.stage == 3:
            from aya_distill.distill.sft import SFTTrainer
            import yaml
            config_path = args.config or "configs/distill_stage3.yaml"
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            trainer = SFTTrainer(**cfg)
            trainer.train()
    elif args.command == "eval":
        import subprocess
        from pathlib import Path
        script = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_baseline.py"
        cmd = [sys.executable, str(script), "--config", args.config]
        if args.model:
            cmd.extend(["--model", args.model])
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    elif args.command == "profile":
        import subprocess
        from pathlib import Path
        script = Path(__file__).resolve().parent.parent.parent / "scripts" / "profile_tiny_aya.py"
        cmd = [
            sys.executable, str(script),
            "--device", args.device,
            "--output", args.output,
        ]
        if args.quantize:
            cmd.extend(["--quantize", args.quantize])
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
