#!/usr/bin/env python3
"""
Profile Tiny Aya (CohereForAI/aya-expanse-8b) resource consumption.

Measures loading time, memory, inference latency per language,
and tokenizer statistics across all 10 target languages.

Usage:
    # GPU with 4-bit quantization (fits 4GB VRAM):
    python scripts/profile_tiny_aya.py --quantize 4bit --output results/profile.json

    # CPU only:
    python scripts/profile_tiny_aya.py --device cpu --output results/profile.json

    # GPU + CPU offload (auto split):
    python scripts/profile_tiny_aya.py --device auto --output results/profile.json
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("profile_tiny_aya")

MODEL_NAME = "CohereForAI/aya-expanse-8b"

LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "zh": "Mandarin",
    "ar": "Arabic",
    "sw": "Swahili",
    "tr": "Turkish",
    "ja": "Japanese",
    "id": "Indonesian",
    "te": "Telugu",
}

# Sample prompts in each language for profiling
SAMPLE_PROMPTS = {
    "en": "Explain the concept of gravity in simple terms.",
    "es": "Explica el concepto de gravedad en terminos sencillos.",
    "hi": "गुरुत्वाकर्षण की अवधारणा को सरल शब्दों में समझाइए।",
    "zh": "用简单的语言解释引力的概念。",
    "ar": "اشرح مفهوم الجاذبية بعبارات بسيطة.",
    "sw": "Eleza dhana ya mvutano kwa maneno rahisi.",
    "tr": "Yercekim kavramini basit terimlerle aciklayin.",
    "ja": "重力の概念を簡単な言葉で説明してください。",
    "id": "Jelaskan konsep gravitasi dengan istilah sederhana.",
    "te": "గురుత్వాకర్షణ భావనను సరళమైన పదాలలో వివరించండి.",
}

# Longer text samples for tokenizer ratio analysis
TOKENIZER_SAMPLES = {
    "en": "The quick brown fox jumps over the lazy dog. Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    "es": "El rapido zorro marron salta sobre el perro perezoso. El aprendizaje automatico es un subconjunto de la inteligencia artificial que se centra en construir sistemas que aprenden de los datos.",
    "hi": "तेज भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है। मशीन लर्निंग कृत्रिम बुद्धिमत्ता का एक उपसमुच्चय है जो डेटा से सीखने वाली प्रणालियों के निर्माण पर केंद्रित है।",
    "zh": "敏捷的棕色狐狸跳过了懒狗。机器学习是人工智能的一个子集，专注于构建从数据中学习的系统。",
    "ar": "الثعلب البني السريع يقفز فوق الكلب الكسول. التعلم الآلي هو مجموعة فرعية من الذكاء الاصطناعي تركز على بناء أنظمة تتعلم من البيانات.",
    "sw": "Mbweha wa kahawia mwepesi anaruka juu ya mbwa mvivu. Ujifunzaji wa mashine ni sehemu ndogo ya akili bandia inayozingatia kujenga mifumo inayojifunza kutoka kwa data.",
    "tr": "Hizli kahverengi tilki tembel kopegin uzerinden atlar. Makine ogrenimi, verilerden ogrenen sistemler olusturmaya odaklanan yapay zekanin bir alt kumesidir.",
    "ja": "素早い茶色の狐が怠けた犬を飛び越える。機械学習はデータから学習するシステムの構築に焦点を当てた人工知能のサブセットです。",
    "id": "Rubah cokelat cepat melompati anjing malas. Pembelajaran mesin adalah bagian dari kecerdasan buatan yang berfokus pada membangun sistem yang belajar dari data.",
    "te": "వేగవంతమైన గోధుమ రంగు నక్క సోమరి కుక్క మీదుగా దూకుతుంది. మెషిన్ లర్నింగ్ అనేది డేటా నుండి నేర్చుకునే వ్యవస్థలను నిర్మించడంపై దృష్టి సారించే కృత్రిమ మేధస్సు యొక్క ఉపసమితి.",
}


def get_mem_stats() -> dict[str, float]:
    """Get current memory stats in GB."""
    stats: dict[str, float] = {}
    if torch.cuda.is_available():
        stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        stats["gpu_free_gb"] = torch.cuda.mem_get_info()[0] / 1024**3
    import psutil
    proc = psutil.Process()
    stats["ram_used_gb"] = proc.memory_info().rss / 1024**3
    stats["ram_total_gb"] = psutil.virtual_memory().total / 1024**3
    return stats


def load_model(
    device: str, quantize: str | None
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """Load Aya model and return load metrics."""
    logger.info("Loading %s (device=%s, quantize=%s)", MODEL_NAME, device, quantize)

    load_metrics: dict = {}
    mem_before = get_mem_stats()
    t0 = time.perf_counter()

    kwargs: dict = {"trust_remote_code": True}

    if quantize == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    elif quantize == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    elif device == "auto":
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    elif device == "cpu":
        kwargs["device_map"] = "cpu"
        kwargs["torch_dtype"] = torch.float32
    else:
        kwargs["device_map"] = device
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    model.eval()

    load_time = time.perf_counter() - t0
    mem_after = get_mem_stats()

    load_metrics["load_time_seconds"] = round(load_time, 2)
    load_metrics["mem_before"] = mem_before
    load_metrics["mem_after"] = mem_after
    load_metrics["model_params"] = sum(p.numel() for p in model.parameters())
    load_metrics["model_dtype"] = str(next(model.parameters()).dtype)
    load_metrics["quantization"] = quantize

    logger.info(
        "Loaded in %.1fs | Params: %s | RAM: %.2fGB | GPU: %.2fGB",
        load_time,
        f"{load_metrics['model_params']:,}",
        mem_after.get("ram_used_gb", 0),
        mem_after.get("gpu_allocated_gb", 0),
    )

    return model, tokenizer, load_metrics


@torch.inference_mode()
def profile_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 64,
    warmup: int = 1,
) -> dict[str, dict]:
    """Profile inference latency per language."""
    results: dict[str, dict] = {}

    # Warmup
    for _ in range(warmup):
        inputs = tokenizer("Hello", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        model.generate(**inputs, max_new_tokens=8, do_sample=False)

    for lang, prompt in SAMPLE_PROMPTS.items():
        logger.info("Profiling inference: %s (%s)", lang, LANGUAGES[lang])

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Time to first token
        t0 = time.perf_counter()
        out = model.generate(
            **inputs, max_new_tokens=1, do_sample=False
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        # Full generation
        t0 = time.perf_counter()
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0

        output_len = out.shape[1] - input_len
        tokens_per_sec = output_len / gen_time if gen_time > 0 else 0

        generated_text = tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True
        )

        results[lang] = {
            "language": LANGUAGES[lang],
            "input_tokens": input_len,
            "output_tokens": output_len,
            "ttft_seconds": round(ttft, 4),
            "total_gen_seconds": round(gen_time, 4),
            "tokens_per_second": round(tokens_per_sec, 2),
            "generated_preview": generated_text[:200],
        }

        logger.info(
            "  %s: %d→%d tokens, %.2f tok/s, TTFT=%.3fs",
            lang, input_len, output_len, tokens_per_sec, ttft,
        )

    return results


def profile_tokenizer(tokenizer: AutoTokenizer) -> dict[str, dict]:
    """Analyze tokenizer behavior per language."""
    results: dict[str, dict] = {}

    for lang, text in TOKENIZER_SAMPLES.items():
        tokens = tokenizer.encode(text)
        words = text.split()
        chars = len(text)

        results[lang] = {
            "language": LANGUAGES[lang],
            "char_count": chars,
            "word_count": len(words),
            "token_count": len(tokens),
            "tokens_per_word": round(len(tokens) / max(len(words), 1), 2),
            "tokens_per_char": round(len(tokens) / max(chars, 1), 4),
            "avg_token_length": round(chars / max(len(tokens), 1), 2),
        }

        logger.info(
            "  %s: %d words → %d tokens (%.2f tok/word)",
            lang, len(words), len(tokens), results[lang]["tokens_per_word"],
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Tiny Aya resource usage")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--quantize", default=None, choices=["4bit", "8bit"],
                        help="Quantization mode")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="results/profile_tiny_aya.json")
    args = parser.parse_args()

    results: dict = {
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "quantize": args.quantize,
        "system": {},
    }

    # System info
    if torch.cuda.is_available():
        results["system"]["gpu"] = torch.cuda.get_device_name(0)
        results["system"]["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1024**3, 2
        )
    results["system"]["cuda_available"] = torch.cuda.is_available()

    # Load model
    model, tokenizer, load_metrics = load_model(args.device, args.quantize)
    results["load"] = load_metrics

    # Tokenizer stats
    logger.info("=== Tokenizer Analysis ===")
    results["tokenizer"] = profile_tokenizer(tokenizer)
    results["tokenizer"]["vocab_size"] = tokenizer.vocab_size

    # Inference profiling
    logger.info("=== Inference Profiling ===")
    results["inference"] = profile_inference(
        model, tokenizer, max_new_tokens=args.max_new_tokens
    )

    # Memory after inference
    results["mem_post_inference"] = get_mem_stats()

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to %s", out_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Tiny Aya Profile Summary")
    print(f"{'='*60}")
    print(f"  Load time:  {load_metrics['load_time_seconds']}s")
    print(f"  Parameters: {load_metrics['model_params']:,}")
    print(f"  Dtype:      {load_metrics['model_dtype']}")
    print(f"  Quantize:   {args.quantize or 'none'}")
    print(f"\n  Tokenizer (tokens per word):")
    for lang, data in results["tokenizer"].items():
        if isinstance(data, dict):
            print(f"    {lang}: {data.get('tokens_per_word', 'N/A')}")
    print(f"\n  Inference (tokens/sec):")
    for lang, data in results["inference"].items():
        print(f"    {lang}: {data['tokens_per_second']} tok/s (TTFT={data['ttft_seconds']}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
