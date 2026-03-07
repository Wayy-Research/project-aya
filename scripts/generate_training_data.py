#!/usr/bin/env python3
"""
Generate multilingual training examples using Tiny Aya.

Creates SFT training data in all 10 target languages across multiple
task types: QA, math reasoning, tool calling, and causal reasoning.

Usage:
    python scripts/generate_training_data.py \
      --tasks qa math tool reasoning \
      --languages en es hi zh ar sw tr ja id te \
      --examples-per-task 100 \
      --output data/synthetic/ \
      --quantize 4bit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
logger = logging.getLogger("generate_training_data")

MODEL_NAME = "CohereForAI/aya-expanse-8b"

LANGUAGES = ["en", "es", "hi", "zh", "ar", "sw", "tr", "ja", "id", "te"]

LANG_NAMES = {
    "en": "English", "es": "Spanish", "hi": "Hindi", "zh": "Chinese",
    "ar": "Arabic", "sw": "Swahili", "tr": "Turkish", "ja": "Japanese",
    "id": "Indonesian", "te": "Telugu",
}

# -----------------------------------------------------------------------
# Prompt templates per task, in each language
# -----------------------------------------------------------------------

QA_SYSTEM = {
    "en": "You are a helpful assistant. Answer the following question clearly and concisely.",
    "es": "Eres un asistente util. Responde la siguiente pregunta de manera clara y concisa.",
    "hi": "आप एक सहायक सहायक हैं। निम्नलिखित प्रश्न का स्पष्ट और संक्षिप्त उत्तर दें।",
    "zh": "你是一个有用的助手。请清楚简洁地回答以下问题。",
    "ar": "أنت مساعد مفيد. أجب عن السؤال التالي بوضوح وإيجاز.",
    "sw": "Wewe ni msaidizi wa kusaidia. Jibu swali lifuatalo kwa uwazi na kwa ufupi.",
    "tr": "Yardimci bir asistansiniz. Asagidaki soruyu acik ve oz bir sekilde cevaplandirin.",
    "ja": "あなたは親切なアシスタントです。次の質問に明確かつ簡潔に答えてください。",
    "id": "Anda adalah asisten yang membantu. Jawab pertanyaan berikut dengan jelas dan ringkas.",
    "te": "మీరు సహాయకరమైన సహాయకుడు. కింది ప్రశ్నకు స్పష్టంగా మరియు సంక్షిప్తంగా సమాధానం ఇవ్వండి.",
}

QA_SEED_TOPICS = [
    "photosynthesis", "water cycle", "solar system", "gravity",
    "democracy", "climate change", "electricity", "ecosystems",
    "human body", "history of computers", "renewable energy",
    "ocean currents", "volcanoes", "nutrition", "space exploration",
    "ancient civilizations", "economics basics", "music theory",
    "biodiversity", "the internet",
]

MATH_PROMPTS = {
    "en": "Write a grade school math word problem and solve it step by step. Topic: {topic}. Show your work with #### before the final numerical answer.",
    "es": "Escribe un problema matematico de escuela primaria y resuelvelo paso a paso. Tema: {topic}. Muestra tu trabajo con #### antes de la respuesta numerica final.",
    "hi": "एक प्राथमिक विद्यालय गणित शब्द समस्या लिखें और इसे चरण दर चरण हल करें। विषय: {topic}। अंतिम संख्यात्मक उत्तर से पहले #### दिखाएं।",
    "zh": "写一道小学数学应用题并逐步解答。主题：{topic}。在最终数字答案前显示 ####。",
    "ar": "اكتب مسألة رياضيات كلامية للمرحلة الابتدائية وحلها خطوة بخطوة. الموضوع: {topic}. أظهر عملك مع #### قبل الإجابة الرقمية النهائية.",
    "sw": "Andika tatizo la hesabu la shule ya msingi na ulitatue hatua kwa hatua. Mada: {topic}. Onyesha kazi yako na #### kabla ya jibu la mwisho la nambari.",
    "tr": "Bir ilkokul matematik problemi yazin ve adim adim cozun. Konu: {topic}. Son sayisal cevaptan once #### ile calismanizi gosterin.",
    "ja": "小学校の算数の文章題を書いて、段階的に解いてください。トピック：{topic}。最終的な数値の答えの前に #### を表示してください。",
    "id": "Tulis soal cerita matematika sekolah dasar dan selesaikan langkah demi langkah. Topik: {topic}. Tunjukkan pekerjaan Anda dengan #### sebelum jawaban numerik akhir.",
    "te": "ప్రాథమిక పాఠశాల గణిత పద సమస్యను వ్రాసి దశలవారీగా పరిష్కరించండి. అంశం: {topic}. చివరి సంఖ్యాత్మక సమాధానానికి ముందు #### చూపించండి.",
}

MATH_TOPICS = [
    "shopping", "distance and speed", "sharing equally",
    "fractions", "time", "money", "area", "percentages",
    "ratios", "averages",
]

TOOL_SYSTEM = {
    "en": "You are an AI assistant with access to tools. When the user asks you to perform an action, respond with a JSON function call. Available tools: get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str).",
    "es": "Eres un asistente de IA con acceso a herramientas. Cuando el usuario te pida realizar una accion, responde con una llamada de funcion JSON. Herramientas disponibles: get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str).",
    "hi": "आप एक AI सहायक हैं जिसके पास टूल्स तक पहुंच है। जब उपयोगकर्ता आपसे कोई कार्य करने के लिए कहे, तो JSON फ़ंक्शन कॉल के साथ जवाब दें। उपलब्ध टूल्स: get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str)।",
    "zh": "你是一个可以使用工具的AI助手。当用户要求你执行操作时，请用JSON函数调用来回应。可用工具：get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str)。",
    "ar": "أنت مساعد ذكاء اصطناعي لديه وصول إلى أدوات. عندما يطلب منك المستخدم تنفيذ إجراء، استجب باستدعاء دالة JSON. الأدوات المتاحة: get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str).",
    "sw": "Wewe ni msaidizi wa AI wenye ufikiaji wa zana. Mtumiaji anapokuuliza kufanya kitendo, jibu kwa wito wa kazi ya JSON. Zana zinazopatikana: get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str).",
    "tr": "Araclara erisimi olan bir yapay zeka asistanisiniz. Kullanici sizden bir eylem yapmanizi istediginde JSON fonksiyon cagrisi ile yanit verin. Mevcut araclar: get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str).",
    "ja": "あなたはツールにアクセスできるAIアシスタントです。ユーザーがアクションの実行を依頼した場合、JSON関数呼び出しで応答してください。利用可能なツール：get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str)。",
    "id": "Anda adalah asisten AI dengan akses ke alat. Ketika pengguna meminta Anda melakukan tindakan, respons dengan panggilan fungsi JSON. Alat yang tersedia: get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str).",
    "te": "మీరు సాధనాలకు ప్రాప్యత ఉన్న AI సహాయకుడు. వినియోగదారు మిమ్మల్ని చర్య చేయమని అడిగినప్పుడు, JSON ఫంక్షన్ కాల్‌తో స్పందించండి. అందుబాటులో ఉన్న సాధనాలు: get_weather(city: str), calculate(expression: str), search_web(query: str), translate(text: str, target_language: str), set_reminder(message: str, time: str).",
}

TOOL_QUERIES = {
    "en": [
        "What's the weather like in Tokyo?",
        "Calculate 15% of 230",
        "Set a reminder to call mom at 3pm",
        "Translate 'hello world' to French",
        "Search for recent news about climate change",
    ],
    "es": [
        "Como esta el clima en Madrid?",
        "Calcula el 20% de 450",
        "Pon un recordatorio para llamar al doctor a las 5pm",
        "Traduce 'buenos dias' al ingles",
        "Busca noticias recientes sobre energia renovable",
    ],
}

REASONING_PROMPTS = {
    "en": "Think step by step about the following scenario and explain the cause and effect: {scenario}",
    "es": "Piensa paso a paso sobre el siguiente escenario y explica la causa y el efecto: {scenario}",
    "hi": "निम्नलिखित परिदृश्य के बारे में चरण दर चरण सोचें और कारण और प्रभाव की व्याख्या करें: {scenario}",
    "zh": "请逐步思考以下场景并解释因果关系：{scenario}",
    "ar": "فكر خطوة بخطوة في السيناريو التالي واشرح السبب والنتيجة: {scenario}",
    "sw": "Fikiria hatua kwa hatua kuhusu hali ifuatayo na ueleze sababu na athari: {scenario}",
    "tr": "Asagidaki senaryo hakkinda adim adim dusunun ve neden-sonuc iliskisini aciklayin: {scenario}",
    "ja": "次のシナリオについて段階的に考え、因果関係を説明してください：{scenario}",
    "id": "Pikirkan langkah demi langkah tentang skenario berikut dan jelaskan sebab dan akibatnya: {scenario}",
    "te": "కింది దృశ్యం గురించి దశలవారీగా ఆలోచించండి మరియు కారణం మరియు ప్రభావాన్ని వివరించండి: {scenario}",
}

REASONING_SCENARIOS = [
    "A city plants more trees along its streets",
    "A factory starts using renewable energy",
    "A school introduces free lunch for all students",
    "Ocean temperatures rise by 2 degrees",
    "A country invests heavily in public transportation",
    "Bees start disappearing from an ecosystem",
    "A new social media platform becomes popular among teenagers",
    "Water prices double in a drought-affected region",
    "A village gains access to high-speed internet for the first time",
    "A major river is dammed for hydroelectric power",
]


# Flag for graceful shutdown
_shutdown = False


def _signal_handler(sig: int, frame: Any) -> None:
    global _shutdown
    logger.info("Interrupt received, saving partial results...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)


def load_model(
    device: str, quantize: str | None
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Aya model."""
    kwargs: dict = {"trust_remote_code": True}
    if quantize == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    elif device == "auto":
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["device_map"] = device
        kwargs["torch_dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a response from chat messages."""
    # Use chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "\n".join(
            f"<|{m['role']}|>\n{m['content']}" for m in messages
        ) + "\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.9 if temperature > 0 else None,
    )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def get_completed(output_dir: Path) -> set[str]:
    """Get set of already-generated (lang, task) combos for resume support."""
    completed: set[str] = set()
    for f in output_dir.glob("*.jsonl"):
        # filename: {lang}_{task}.jsonl
        stem = f.stem
        if "_" in stem:
            completed.add(stem)
    return completed


def generate_qa(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lang: str,
    n_examples: int,
) -> list[dict]:
    """Generate QA training examples."""
    examples = []
    system = QA_SYSTEM.get(lang, QA_SYSTEM["en"])

    for i in range(n_examples):
        if _shutdown:
            break
        topic = QA_SEED_TOPICS[i % len(QA_SEED_TOPICS)]

        # Ask model to generate a question about the topic
        question_prompt = f"Ask a question about {topic} in {LANG_NAMES[lang]}. Just write the question, nothing else."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question_prompt},
        ]
        question = generate(model, tokenizer, messages, max_new_tokens=128, temperature=0.8)

        # Now generate the answer
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question.strip()},
        ]
        answer = generate(model, tokenizer, messages, max_new_tokens=512, temperature=0.7)

        examples.append({
            "language": lang,
            "task": "qa",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question.strip()},
                {"role": "assistant", "content": answer.strip()},
            ],
            "metadata": {
                "model": MODEL_NAME,
                "topic": topic,
                "temperature": 0.7,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "index": i,
            },
        })

        if (i + 1) % 10 == 0:
            logger.info("  [%s/qa] Generated %d/%d", lang, i + 1, n_examples)

    return examples


def generate_math(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lang: str,
    n_examples: int,
) -> list[dict]:
    """Generate math reasoning examples."""
    examples = []
    template = MATH_PROMPTS.get(lang, MATH_PROMPTS["en"])

    for i in range(n_examples):
        if _shutdown:
            break
        topic = MATH_TOPICS[i % len(MATH_TOPICS)]
        prompt = template.format(topic=topic)

        messages = [{"role": "user", "content": prompt}]
        response = generate(model, tokenizer, messages, max_new_tokens=512, temperature=0.7)

        examples.append({
            "language": lang,
            "task": "math",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.strip()},
            ],
            "metadata": {
                "model": MODEL_NAME,
                "topic": topic,
                "temperature": 0.7,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "index": i,
            },
        })

        if (i + 1) % 10 == 0:
            logger.info("  [%s/math] Generated %d/%d", lang, i + 1, n_examples)

    return examples


def generate_tool(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lang: str,
    n_examples: int,
) -> list[dict]:
    """Generate tool calling examples."""
    examples = []
    system = TOOL_SYSTEM.get(lang, TOOL_SYSTEM["en"])

    # Use language-specific queries if available, else ask model to generate
    seed_queries = TOOL_QUERIES.get(lang, TOOL_QUERIES["en"])

    for i in range(n_examples):
        if _shutdown:
            break

        if i < len(seed_queries):
            query = seed_queries[i]
        else:
            # Ask model to generate a tool-use query
            gen_prompt = (
                f"Write a short user request in {LANG_NAMES[lang]} that would "
                f"require using one of these tools: weather, calculator, web search, "
                f"translation, or reminder. Just write the request, nothing else."
            )
            query = generate(
                model, tokenizer,
                [{"role": "user", "content": gen_prompt}],
                max_new_tokens=100, temperature=0.9,
            ).strip()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        response = generate(model, tokenizer, messages, max_new_tokens=256, temperature=0.3)

        examples.append({
            "language": lang,
            "task": "tool",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response.strip()},
            ],
            "metadata": {
                "model": MODEL_NAME,
                "temperature": 0.3,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "index": i,
            },
        })

        if (i + 1) % 10 == 0:
            logger.info("  [%s/tool] Generated %d/%d", lang, i + 1, n_examples)

    return examples


def generate_reasoning(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lang: str,
    n_examples: int,
) -> list[dict]:
    """Generate causal reasoning examples."""
    examples = []
    template = REASONING_PROMPTS.get(lang, REASONING_PROMPTS["en"])

    for i in range(n_examples):
        if _shutdown:
            break
        scenario = REASONING_SCENARIOS[i % len(REASONING_SCENARIOS)]
        prompt = template.format(scenario=scenario)

        messages = [{"role": "user", "content": prompt}]
        response = generate(model, tokenizer, messages, max_new_tokens=512, temperature=0.7)

        examples.append({
            "language": lang,
            "task": "reasoning",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.strip()},
            ],
            "metadata": {
                "model": MODEL_NAME,
                "scenario": scenario,
                "temperature": 0.7,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "index": i,
            },
        })

        if (i + 1) % 10 == 0:
            logger.info("  [%s/reasoning] Generated %d/%d", lang, i + 1, n_examples)

    return examples


TASK_GENERATORS = {
    "qa": generate_qa,
    "math": generate_math,
    "tool": generate_tool,
    "reasoning": generate_reasoning,
}


def save_examples(examples: list[dict], path: Path) -> None:
    """Save examples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples to %s", len(examples), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multilingual training data with Tiny Aya")
    parser.add_argument("--tasks", nargs="+", default=["qa", "math", "tool", "reasoning"],
                        choices=["qa", "math", "tool", "reasoning"])
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--examples-per-task", type=int, default=100)
    parser.add_argument("--output", default="data/synthetic/")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantize", default=None, choices=["4bit", "8bit"])
    parser.add_argument("--resume", action="store_true", help="Skip already-generated files")
    args = parser.parse_args()

    output_dir = Path(args.output)
    completed = get_completed(output_dir) if args.resume else set()

    logger.info("Loading model...")
    model, tokenizer = load_model(args.device, args.quantize)
    logger.info("Model loaded. Starting generation.")

    total = len(args.languages) * len(args.tasks) * args.examples_per_task
    generated = 0

    for lang in args.languages:
        for task in args.tasks:
            if _shutdown:
                break

            key = f"{lang}_{task}"
            if key in completed:
                logger.info("Skipping %s (already exists)", key)
                continue

            logger.info("Generating %s/%s (%d examples)", lang, task, args.examples_per_task)

            gen_fn = TASK_GENERATORS[task]
            examples = gen_fn(model, tokenizer, lang, args.examples_per_task)
            generated += len(examples)

            save_examples(examples, output_dir / f"{key}.jsonl")

            logger.info(
                "Progress: %d/%d examples (%d%%)",
                generated, total, int(100 * generated / total),
            )

    # Write manifest
    manifest = {
        "model": MODEL_NAME,
        "tasks": args.tasks,
        "languages": args.languages,
        "examples_per_task": args.examples_per_task,
        "total_generated": generated,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": sorted(str(p.name) for p in output_dir.glob("*.jsonl")),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Done. Generated %d total examples.", generated)


if __name__ == "__main__":
    main()
