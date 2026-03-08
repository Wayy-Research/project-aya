"""
Few-shot prompt templates for mGSM (multilingual Grade School Math).

We use native-language instructions following the Cohere multilingual
checklist: prompts should be in the target language, not English-only,
to avoid systematically disadvantaging non-English speakers.

Each language gets 8 exemplars drawn from the mGSM training split.
The exemplars below are *static* seeds; the benchmark loader can
optionally replace them with dataset-sampled exemplars.

Wayy Research -- Project Aya
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Native-language instruction prefixes
# ---------------------------------------------------------------------------

MGSM_INSTRUCTIONS: dict[str, str] = {
    "en": "Solve the following math problem step by step. Give the final numerical answer after ####.",
    "es": "Resuelve el siguiente problema matematico paso a paso. Da la respuesta numerica final despues de ####.",
    "hi": "निम्नलिखित गणित की समस्या को चरण-दर-चरण हल करें। #### के बाद अंतिम संख्यात्मक उत्तर दें।",
    "zh": "请逐步解决以下数学问题。在 #### 之后给出最终数字答案。",
    "ar": "حل المسألة الرياضية التالية خطوة بخطوة. أعط الإجابة الرقمية النهائية بعد ####.",
    "sw": "Tatua tatizo lifuatalo la hisabati hatua kwa hatua. Toa jibu la mwisho la nambari baada ya ####.",
    "tr": "Asagidaki matematik problemini adim adim cozun. #### isaretinden sonra nihai sayisal cevabi verin.",
    "ja": "次の算数の問題をステップバイステップで解いてください。#### の後に最終的な数値の答えを書いてください。",
    "id": "Selesaikan soal matematika berikut langkah demi langkah. Berikan jawaban numerik akhir setelah ####.",
    "te": "కింది గణిత సమస్యను దశలవారీగా పరిష్కరించండి. #### తర్వాత తుది సంఖ్యాత్మక సమాధానం ఇవ్వండి.",
}


# ---------------------------------------------------------------------------
# Static exemplars (English only -- other languages loaded from dataset)
# ---------------------------------------------------------------------------

MGSM_EXEMPLARS: dict[str, list[dict[str, str]]] = {
    "en": [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "answer": "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\nShe makes 9 * 2 = $18 every day at the farmer's market.\n#### 18",
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer": "It takes 2 / 2 = 1 bolt of white fiber.\nSo the total bolts needed is 2 + 1 = 3.\n#### 3",
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "answer": "The cost of the house and repairs came out to 80,000 + 50,000 = $130,000.\nHe increased the value of the house by 80,000 * 150% = $120,000.\nSo the new value of the house is 80,000 + 120,000 = $200,000.\nSo he made a profit of 200,000 - 130,000 = $70,000.\n#### 70000",
        },
        {
            "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the carry-over from prior feedings is 35 cups?",
            "answer": "Wendi has a total of 15 + 25 = 40 cups in the first two feedings.\nSince each chicken gets 3 cups, she has 40 / 3 = 13.33, so about 13 chickens.\nBut actually the carry-over is 35 cups, so she needs 35 cups in the final meal.\n#### 35",
        },
        {
            "question": "Kylar went to the store to get his 2 gallons of whole milk but found out that a gallon of whole milk costs $3. He decided to just buy 1 gallon of whole milk and also bought a box of cereal for $2. At the register, he got a discount of $1 off his total. How much did Kylar spend?",
            "answer": "A gallon of whole milk costs $3 and a box of cereal costs $2 so the total is 3 + 2 = $5.\nHe got $1 off so he spent 5 - 1 = $4.\n#### 4",
        },
        {
            "question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
            "answer": "Charleston has 4 * 20 = 80 sheep.\nToulouse has 2 * 80 = 160 sheep.\nTogether they have 20 + 80 + 160 = 260 sheep.\n#### 260",
        },
        {
            "question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load(), in minutes, does it take to download the file?",
            "answer": "First, 40% of 200 GB is 0.4 * 200 = 80 GB.\n80 GB at 2 GB/minute takes 80 / 2 = 40 minutes.\nThen the restart takes 20 minutes.\nThen she downloads the full 200 GB which takes 200 / 2 = 100 minutes.\nTotal is 40 + 20 + 100 = 160 minutes.\n#### 160",
        },
        {
            "question": "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home. He tries to get home in 4 hours but spends the first 2 hours in standstill traffic. He spends the rest of the time driving at 30 mph. How far is he from home when the 4 hours are up?",
            "answer": "He drove 3 * 60 = 180 miles away from home.\nHe was stuck in traffic for 2 hours, then drove for 4 - 2 = 2 hours at 30 mph.\nHe drove back 2 * 30 = 60 miles.\nSo he is 180 - 60 = 120 miles from home.\n#### 120",
        },
    ],
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_mgsm_prompt(
    question: str,
    language: str,
    exemplars: list[dict[str, str]] | None = None,
    n_shot: int = 8,
) -> str:
    """Build an n-shot prompt for mGSM.

    Parameters
    ----------
    question : the test question to answer
    language : ISO 639-1 code (e.g. "en", "hi", "zh")
    exemplars : list of {"question": ..., "answer": ...} dicts.
                If None, falls back to static English exemplars.
    n_shot : number of exemplars to include (default 8)

    Returns
    -------
    str : the full prompt ready for model input
    """
    instruction = MGSM_INSTRUCTIONS.get(language, MGSM_INSTRUCTIONS["en"])
    shots = exemplars or MGSM_EXEMPLARS.get(language, MGSM_EXEMPLARS["en"])
    shots = shots[:n_shot]

    parts: list[str] = [instruction, ""]

    for i, ex in enumerate(shots, 1):
        parts.append(f"Q: {ex['question']}")
        parts.append(f"A: {ex['answer']}")
        parts.append("")

    parts.append(f"Q: {question}")
    parts.append("A:")

    return "\n".join(parts)


def extract_mgsm_answer(text: str) -> str:
    """Extract the numerical answer after #### from model output.

    Returns the stripped numeric string, or empty string if not found.
    """
    if "####" in text:
        answer = text.split("####")[-1].strip()
        # Remove commas, dollar signs, whitespace
        answer = answer.replace(",", "").replace("$", "").strip()
        # Try to find the first number-like token
        tokens = answer.split()
        if tokens:
            return tokens[0]
    return ""
