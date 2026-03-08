"""Test fixtures and sample data for multilingual testing."""
from __future__ import annotations

# General knowledge prompts across all 10 target languages.
# Each key is a prompt category; each value maps lang_code -> prompt.
MULTILINGUAL_PROMPTS: dict[str, dict[str, str]] = {
    "general_knowledge": {
        "en": "Explain the concept of gravity in simple terms.",
        "es": "Explica el concepto de gravedad en terminos sencillos.",
        "hi": "\u0917\u0941\u0930\u0941\u0924\u094d\u0935\u093e\u0915\u0930\u094d\u0937\u0923 \u0915\u0940 \u0905\u0935\u0927\u093e\u0930\u0923\u093e \u0915\u094b \u0938\u0930\u0932 \u0936\u092c\u094d\u0926\u094b\u0902 \u092e\u0947\u0902 \u0938\u092e\u091d\u093e\u0907\u090f\u0964",
        "zh": "\u7528\u7b80\u5355\u7684\u8bed\u8a00\u89e3\u91ca\u5f15\u529b\u7684\u6982\u5ff5\u3002",
        "ar": "\u0627\u0634\u0631\u062d \u0645\u0641\u0647\u0648\u0645 \u0627\u0644\u062c\u0627\u0630\u0628\u064a\u0629 \u0628\u0639\u0628\u0627\u0631\u0627\u062a \u0628\u0633\u064a\u0637\u0629.",
        "sw": "Eleza dhana ya mvutano kwa maneno rahisi.",
        "tr": "Yercekim kavramini basit terimlerle aciklayin.",
        "ja": "\u91cd\u529b\u306e\u6982\u5ff5\u3092\u7c21\u5358\u306a\u8a00\u8449\u3067\u8aac\u660e\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
        "id": "Jelaskan konsep gravitasi dengan istilah sederhana.",
        "te": "\u0c17\u0c41\u0c30\u0c41\u0c24\u0c4d\u0c35\u0c3e\u0c15\u0c30\u0c4d\u0c37\u0c23 \u0c2d\u0c3e\u0c35\u0c28\u0c28\u0c41 \u0c38\u0c30\u0c33\u0c2e\u0c48\u0c28 \u0c2a\u0c26\u0c3e\u0c32\u0c32\u0c4b \u0c35\u0c3f\u0c35\u0c30\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f.",
    },
    "reasoning": {
        "en": (
            "If a train leaves at 3pm going 60mph, and another at 4pm going "
            "80mph on the same track, when does the second catch up?"
        ),
        "es": (
            "Si un tren sale a las 3pm a 60mph, y otro a las 4pm a 80mph "
            "en la misma via, cuando alcanza el segundo al primero?"
        ),
        "hi": (
            "\u092f\u0926\u093f \u090f\u0915 \u091f\u094d\u0930\u0947\u0928 3pm \u092a\u0930 60mph \u0938\u0947 \u091a\u0932\u0924\u0940 \u0939\u0948, \u0914\u0930 \u0926\u0942\u0938\u0930\u0940 4pm \u092a\u0930 80mph \u0938\u0947 "
            "\u0909\u0938\u0940 \u091f\u094d\u0930\u0948\u0915 \u092a\u0930, \u0924\u094b \u0926\u0942\u0938\u0930\u0940 \u0915\u092c \u092a\u0939\u0932\u0940 \u0915\u094b \u092a\u0915\u0921\u093c\u0924\u0940 \u0939\u0948?"
        ),
        "zh": (
            "\u5982\u679c\u4e00\u5217\u706b\u8f66\u4e0b\u5348 3 \u70b9\u4ee5 60 \u82f1\u91cc\u7684\u901f\u5ea6\u51fa\u53d1\uff0c\u53e6\u4e00\u5217\u4e0b\u5348 4 \u70b9\u4ee5 80 "
            "\u82f1\u91cc\u7684\u901f\u5ea6\u51fa\u53d1\uff0c\u7b2c\u4e8c\u5217\u4f55\u65f6\u8ffd\u4e0a\u7b2c\u4e00\u5217\uff1f"
        ),
        "ar": (
            "\u0625\u0630\u0627 \u063a\u0627\u062f\u0631 \u0642\u0637\u0627\u0631 \u0627\u0644\u0633\u0627\u0639\u0629 3 \u0645\u0633\u0627\u0621 \u0628\u0633\u0631\u0639\u0629 60 \u0645\u064a\u0644 \u0641\u064a \u0627\u0644\u0633\u0627\u0639\u0629\u060c "
            "\u0648\u0622\u062e\u0631 \u0627\u0644\u0633\u0627\u0639\u0629 4 \u0628\u0633\u0631\u0639\u0629 80\u060c \u0645\u062a\u0649 \u064a\u0644\u062d\u0642 \u0627\u0644\u062b\u0627\u0646\u064a \u0628\u0627\u0644\u0623\u0648\u0644\u061f"
        ),
        "sw": (
            "Treni ikiondoka saa 3 usiku kwa 60mph, na nyingine saa 4 "
            "kwa 80mph, ya pili inafika lini?"
        ),
        "tr": (
            "Bir tren saat 3'te 60mph ile, digeri 4'te 80mph ile ayni "
            "hattan kalkarsa, ikincisi ne zaman yakalar?"
        ),
        "ja": (
            "\u5348\u5f8c 3 \u6642\u306b\u6642\u901f 60 \u30de\u30a4\u30eb\u3067\u51fa\u767a\u3057\u305f\u96fb\u8eca\u3068\u3001\u5348\u5f8c 4 \u6642\u306b\u6642\u901f 80 "
            "\u30de\u30a4\u30eb\u3067\u51fa\u767a\u3057\u305f\u96fb\u8eca\u30012 \u756a\u76ee\u306f\u3044\u3064\u8ffd\u3044\u3064\u304d\u307e\u3059\u304b\uff1f"
        ),
        "id": (
            "Jika kereta berangkat jam 3 sore dengan 60mph, dan lainnya "
            "jam 4 dengan 80mph di jalur yang sama, kapan yang kedua menyusul?"
        ),
        "te": (
            "\u0c30\u0c48\u0c32\u0c41 \u0c2e\u0c27\u0c4d\u0c2f\u0c3e\u0c39\u0c4d\u0c28\u0c02 3 \u0c17\u0c02\u0c1f\u0c32\u0c15\u0c41 60mph \u0c35\u0c47\u0c17\u0c02\u0c24\u0c4b \u0c2c\u0c2f\u0c32\u0c41\u0c26\u0c47\u0c30\u0c3f, "
            "\u0c2e\u0c30\u0c4a\u0c15\u0c1f\u0c3f 4 \u0c17\u0c02\u0c1f\u0c32\u0c15\u0c41 80mph \u0c24\u0c4b \u0c2c\u0c2f\u0c32\u0c41\u0c26\u0c47\u0c30\u0c3f\u0c24\u0c47, "
            "\u0c30\u0c46\u0c02\u0c21\u0c35\u0c26\u0c3f \u0c0e\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 \u0c1a\u0c47\u0c30\u0c41\u0c15\u0c41\u0c02\u0c1f\u0c41\u0c02\u0c26\u0c3f?"
        ),
    },
}

# Tool schemas for tool calling tests - matching common function calling formats.
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "translate",
        "description": "Translate text between languages",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source_lang": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
    },
]

# Tool calling scenarios per language.
# Each scenario asks the model to call get_weather for that language's capital.
TOOL_SCENARIOS: dict[str, dict] = {
    "en": {
        "prompt": (
            "You have these tools: get_weather, search_web, translate. "
            'The user says: "What is the weather in Tokyo?" '
            "Respond with a JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Tokyo"},
    },
    "es": {
        "prompt": (
            "Tienes estas herramientas: get_weather, search_web, translate. "
            'El usuario dice: "\u00bfCual es el clima en Madrid?" '
            "Responde con una llamada JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Madrid"},
    },
    "hi": {
        "prompt": (
            "\u0906\u092a\u0915\u0947 \u092a\u093e\u0938 \u092f\u0947 \u091f\u0942\u0932 \u0939\u0948\u0902: get_weather, search_web, translate\u0964 "
            '\u0909\u092a\u092f\u094b\u0917\u0915\u0930\u094d\u0924\u093e \u0915\u0939\u0924\u093e \u0939\u0948: "\u0926\u093f\u0932\u094d\u0932\u0940 \u0915\u093e \u092e\u094c\u0938\u092e \u0915\u094d\u092f\u093e \u0939\u0948?" '
            "JSON \u091f\u0942\u0932 \u0915\u0949\u0932 \u0938\u0947 \u091c\u0935\u093e\u092c \u0926\u0947\u0902\u0964"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Delhi"},
    },
    "zh": {
        "prompt": (
            "\u4f60\u6709\u8fd9\u4e9b\u5de5\u5177: get_weather, search_web, translate\u3002"
            '\u7528\u6237\u8bf4\uff1a"\u5317\u4eac\u7684\u5929\u6c14\u600e\u4e48\u6837\uff1f" '
            "\u7528JSON\u5de5\u5177\u8c03\u7528\u56de\u7b54\u3002"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Beijing"},
    },
    "ar": {
        "prompt": (
            "\u0644\u062f\u064a\u0643 \u0647\u0630\u0647 \u0627\u0644\u0623\u062f\u0648\u0627\u062a: get_weather, search_web, translate. "
            '\u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 \u064a\u0642\u0648\u0644: "\u0645\u0627 \u0627\u0644\u0637\u0642\u0633 \u0641\u064a \u0627\u0644\u0642\u0627\u0647\u0631\u0629\u061f" '
            "\u0623\u062c\u0628 \u0628\u0646\u062f\u0627\u0621 \u0623\u062f\u0627\u0629 JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Cairo"},
    },
    "sw": {
        "prompt": (
            "Una zana hizi: get_weather, search_web, translate. "
            'Mtumiaji anasema: "Hali ya hewa Nairobi ikoje?" '
            "Jibu kwa JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Nairobi"},
    },
    "tr": {
        "prompt": (
            "Su araclariniz var: get_weather, search_web, translate. "
            "Kullanici diyor ki: \"Istanbul'da hava nasil?\" "
            "JSON arac cagrisi ile yanitlayin."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Istanbul"},
    },
    "ja": {
        "prompt": (
            "\u3053\u308c\u3089\u306e\u30c4\u30fc\u30eb\u304c\u3042\u308a\u307e\u3059: get_weather, search_web, translate\u3002"
            '\u30e6\u30fc\u30b6\u30fc\u304c\u8a00\u3044\u307e\u3059\uff1a\u300c\u6771\u4eac\u306e\u5929\u6c17\u306f\uff1f\u300d'
            "JSON\u30c4\u30fc\u30eb\u30b3\u30fc\u30eb\u3067\u5fdc\u7b54\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Tokyo"},
    },
    "id": {
        "prompt": (
            "Anda memiliki alat: get_weather, search_web, translate. "
            'Pengguna berkata: "Bagaimana cuaca di Jakarta?" '
            "Jawab dengan JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Jakarta"},
    },
    "te": {
        "prompt": (
            "\u0c2e\u0c40 \u0c35\u0c26\u0c4d\u0c26 \u0c08 \u0c38\u0c3e\u0c27\u0c28\u0c3e\u0c32\u0c41 \u0c09\u0c28\u0c4d\u0c28\u0c3e\u0c2f\u0c3f: get_weather, search_web, translate. "
            '\u0c35\u0c3f\u0c28\u0c3f\u0c2f\u0c4b\u0c17\u0c26\u0c3e\u0c30\u0c41 \u0c1a\u0c46\u0c2a\u0c4d\u0c2a\u0c3e\u0c30\u0c41: "\u0c39\u0c48\u0c26\u0c30\u0c3e\u0c2c\u0c3e\u0c26\u0c4d \u0c35\u0c3e\u0c24\u0c3e\u0c35\u0c30\u0c23\u0c02 \u0c0e\u0c32\u0c3e \u0c09\u0c02\u0c26\u0c3f?" '
            "JSON \u0c1f\u0c42\u0c32\u0c4d \u0c15\u0c3e\u0c32\u0c4d\u200c\u0c24\u0c4b \u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02 \u0c07\u0c35\u0c4d\u0c35\u0c02\u0c21\u0c3f."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Hyderabad"},
    },
}
