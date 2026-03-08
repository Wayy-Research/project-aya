"""
Few-shot prompt templates for XCOPA (Cross-lingual Choice of Plausible
Alternatives).

XCOPA tests causal reasoning: given a premise, choose the more plausible
cause or effect from two candidates. We use native-language instructions
and 4-shot exemplars following the Cohere multilingual checklist.

Wayy Research -- Project Aya
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Native-language instruction prefixes
# ---------------------------------------------------------------------------

XCOPA_CAUSE_INSTRUCTIONS: dict[str, str] = {
    "en": "Given the following premise, choose which alternative (1 or 2) is the more plausible cause. Answer with just the number 1 or 2.",
    "es": "Dada la siguiente premisa, elige cual alternativa (1 o 2) es la causa mas plausible. Responde solo con el numero 1 o 2.",
    "hi": "निम्नलिखित आधार वाक्य को देखते हुए, चुनें कि कौन सा विकल्प (1 या 2) अधिक संभावित कारण है। केवल संख्या 1 या 2 से उत्तर दें।",
    "zh": "根据以下前提，选择哪个选项（1或2）是更合理的原因。只回答数字1或2。",
    "ar": "بالنظر إلى المقدمة التالية، اختر البديل (1 أو 2) الذي يمثل السبب الأكثر معقولية. أجب بالرقم 1 أو 2 فقط.",
    "sw": "Kwa kuzingatia hali ifuatayo, chagua ni chaguo lipi (1 au 2) ndilo sababu inayowezekana zaidi. Jibu kwa nambari 1 au 2 tu.",
    "tr": "Asagidaki onculu goz onune alarak, hangi secenegin (1 veya 2) daha makul bir neden oldugunu secin. Sadece 1 veya 2 rakami ile cevaplayin.",
    "ja": "次の前提を踏まえて、どちらの選択肢（1または2）がより妥当な原因かを選んでください。1または2の数字だけで答えてください。",
    "id": "Berdasarkan premis berikut, pilih alternatif mana (1 atau 2) yang merupakan penyebab yang lebih masuk akal. Jawab hanya dengan angka 1 atau 2.",
    "te": "కింది ఆధార వాక్యాన్ని బట్టి, ఏ ప్రత్యామ్నాయం (1 లేదా 2) ఎక్కువ సహేతుకమైన కారణమో ఎంచుకోండి. కేవలం 1 లేదా 2 సంఖ్యతో సమాధానం ఇవ్వండి.",
}

XCOPA_EFFECT_INSTRUCTIONS: dict[str, str] = {
    "en": "Given the following premise, choose which alternative (1 or 2) is the more plausible effect. Answer with just the number 1 or 2.",
    "es": "Dada la siguiente premisa, elige cual alternativa (1 o 2) es el efecto mas plausible. Responde solo con el numero 1 o 2.",
    "hi": "निम्नलिखित आधार वाक्य को देखते हुए, चुनें कि कौन सा विकल्प (1 या 2) अधिक संभावित प्रभाव है। केवल संख्या 1 या 2 से उत्तर दें।",
    "zh": "根据以下前提，选择哪个选项（1或2）是更合理的结果。只回答数字1或2。",
    "ar": "بالنظر إلى المقدمة التالية، اختر البديل (1 أو 2) الذي يمثل التأثير الأكثر معقولية. أجب بالرقم 1 أو 2 فقط.",
    "sw": "Kwa kuzingatia hali ifuatayo, chagua ni chaguo lipi (1 au 2) ndilo athari inayowezekana zaidi. Jibu kwa nambari 1 au 2 tu.",
    "tr": "Asagidaki onculu goz onune alarak, hangi secenegin (1 veya 2) daha makul bir sonuc oldugunu secin. Sadece 1 veya 2 rakami ile cevaplayin.",
    "ja": "次の前提を踏まえて、どちらの選択肢（1または2）がより妥当な結果かを選んでください。1または2の数字だけで答えてください。",
    "id": "Berdasarkan premis berikut, pilih alternatif mana (1 atau 2) yang merupakan efek yang lebih masuk akal. Jawab hanya dengan angka 1 atau 2.",
    "te": "కింది ఆధార వాక్యాన్ని బట్టి, ఏ ప్రత్యామ్నాయం (1 లేదా 2) ఎక్కువ సహేతుకమైన ప్రభావమో ఎంచుకోండి. కేవలం 1 లేదా 2 సంఖ్యతో సమాధానం ఇవ్వండి.",
}


# ---------------------------------------------------------------------------
# Static exemplars (English -- other languages loaded from dataset)
# ---------------------------------------------------------------------------

XCOPA_EXEMPLARS: dict[str, list[dict[str, str]]] = {
    "en": [
        {
            "premise": "The man turned on the faucet.",
            "choice1": "The toilet flushed.",
            "choice2": "Water flowed from the spout.",
            "question": "effect",
            "label": "2",
        },
        {
            "premise": "The girl found a bug in her food.",
            "choice1": "She lost her appetite.",
            "choice2": "She finished her meal.",
            "question": "effect",
            "label": "1",
        },
        {
            "premise": "The woman hired a lawyer.",
            "choice1": "She decided to sue her employer.",
            "choice2": "She decided to drop the charges.",
            "question": "cause",
            "label": "1",
        },
        {
            "premise": "The politician lost the election.",
            "choice1": "He conceded defeat.",
            "choice2": "He declared victory.",
            "question": "effect",
            "label": "1",
        },
    ],
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _format_xcopa_example(
    example: dict[str, str],
    language: str,
) -> str:
    """Format a single XCOPA example for the prompt."""
    q_type = example["question"]
    lines = [
        f"Premise: {example['premise']}",
        f"1: {example['choice1']}",
        f"2: {example['choice2']}",
        f"Answer: {example['label']}",
    ]
    return "\n".join(lines)


def build_xcopa_prompt(
    premise: str,
    choice1: str,
    choice2: str,
    question_type: str,
    language: str,
    exemplars: list[dict[str, str]] | None = None,
    n_shot: int = 4,
) -> str:
    """Build an n-shot prompt for XCOPA.

    Parameters
    ----------
    premise : the premise text
    choice1, choice2 : the two candidate answers
    question_type : "cause" or "effect"
    language : ISO 639-1 code
    exemplars : optional list of exemplar dicts
    n_shot : number of exemplars (default 4)

    Returns
    -------
    str : full prompt ready for model input
    """
    if question_type == "cause":
        instructions = XCOPA_CAUSE_INSTRUCTIONS
    else:
        instructions = XCOPA_EFFECT_INSTRUCTIONS

    instruction = instructions.get(language, instructions["en"])
    shots = exemplars or XCOPA_EXEMPLARS.get(language, XCOPA_EXEMPLARS["en"])
    shots = [s for s in shots if s["question"] == question_type][:n_shot]

    # If not enough exemplars of the right type, use all available
    if len(shots) < n_shot:
        all_shots = exemplars or XCOPA_EXEMPLARS.get(
            language, XCOPA_EXEMPLARS["en"]
        )
        shots = all_shots[:n_shot]

    parts: list[str] = [instruction, ""]

    for ex in shots:
        parts.append(_format_xcopa_example(ex, language))
        parts.append("")

    # Test question
    parts.append(f"Premise: {premise}")
    parts.append(f"1: {choice1}")
    parts.append(f"2: {choice2}")
    parts.append("Answer:")

    return "\n".join(parts)


def extract_xcopa_answer(text: str) -> str:
    """Extract 1 or 2 from model output.

    Looks for the first occurrence of '1' or '2' in the generated text.
    Returns '1', '2', or '' if neither found.
    """
    text = text.strip()
    for char in text:
        if char in ("1", "2"):
            return char
    return ""
