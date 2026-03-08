"""Language registry for Project Aya multilingual evaluation.

CohereLabs/tiny-aya-global supports 70+ languages across multiple families,
scripts, and typologies. This module is the single source of truth for
language metadata used throughout aya-distill.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str          # ISO 639-1 (or 639-3 where no 639-1 exists)
    name: str
    family: str
    script: str
    typology: str
    resourcedness: str  # "high", "medium", "low" per Cohere checklist
    region: str         # Geographic region


LANGUAGES: dict[str, Language] = {
    # European — Romance
    "en": Language("en", "English", "Indo-European", "Latin", "Analytic", "high", "Europe"),
    "nl": Language("nl", "Dutch", "Indo-European", "Latin", "Fusional", "high", "Europe"),
    "fr": Language("fr", "French", "Indo-European", "Latin", "Fusional", "high", "Europe"),
    "it": Language("it", "Italian", "Indo-European", "Latin", "Fusional", "high", "Europe"),
    "pt": Language("pt", "Portuguese", "Indo-European", "Latin", "Fusional", "high", "Europe"),
    "ro": Language("ro", "Romanian", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    "es": Language("es", "Spanish", "Indo-European", "Latin", "Fusional", "high", "Europe"),
    "ca": Language("ca", "Catalan", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    "gl": Language("gl", "Galician", "Indo-European", "Latin", "Fusional", "low", "Europe"),
    # European — Germanic
    "de": Language("de", "German", "Indo-European", "Latin", "Fusional", "high", "Europe"),
    "da": Language("da", "Danish", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    "sv": Language("sv", "Swedish", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    "no": Language("no", "Norwegian", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    # European — Slavic
    "cs": Language("cs", "Czech", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    "pl": Language("pl", "Polish", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    "uk": Language("uk", "Ukrainian", "Indo-European", "Cyrillic", "Fusional", "medium", "Europe"),
    "ru": Language("ru", "Russian", "Indo-European", "Cyrillic", "Fusional", "high", "Europe"),
    "hr": Language("hr", "Croatian", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    "sk": Language("sk", "Slovak", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    "sl": Language("sl", "Slovenian", "Indo-European", "Latin", "Fusional", "medium", "Europe"),
    "sr": Language("sr", "Serbian", "Indo-European", "Cyrillic", "Fusional", "medium", "Europe"),
    "bg": Language("bg", "Bulgarian", "Indo-European", "Cyrillic", "Fusional", "medium", "Europe"),
    # European — Baltic
    "lv": Language("lv", "Latvian", "Indo-European", "Latin", "Fusional", "low", "Europe"),
    "lt": Language("lt", "Lithuanian", "Indo-European", "Latin", "Fusional", "low", "Europe"),
    # European — Other
    "el": Language("el", "Greek", "Indo-European", "Greek", "Fusional", "medium", "Europe"),
    "et": Language("et", "Estonian", "Uralic", "Latin", "Agglutinative", "low", "Europe"),
    "fi": Language("fi", "Finnish", "Uralic", "Latin", "Agglutinative", "medium", "Europe"),
    "hu": Language("hu", "Hungarian", "Uralic", "Latin", "Agglutinative", "medium", "Europe"),
    "eu": Language("eu", "Basque", "Language isolate", "Latin", "Agglutinative", "low", "Europe"),
    "cy": Language("cy", "Welsh", "Indo-European", "Latin", "Fusional", "low", "Europe"),
    "ga": Language("ga", "Irish", "Indo-European", "Latin", "Fusional", "low", "Europe"),
    "mt": Language("mt", "Maltese", "Afroasiatic", "Latin", "Fusional", "low", "Europe"),
    # Middle East & Central Asia
    "ar": Language("ar", "Arabic", "Afroasiatic", "Arabic", "Fusional", "high", "Middle East"),
    "fa": Language("fa", "Persian", "Indo-European", "Arabic", "Fusional", "medium", "Middle East"),
    "ur": Language("ur", "Urdu", "Indo-European", "Arabic", "Fusional", "medium", "South Asia"),
    "tr": Language("tr", "Turkish", "Turkic", "Latin", "Agglutinative", "medium", "Middle East"),
    "he": Language("he", "Hebrew", "Afroasiatic", "Hebrew", "Fusional", "medium", "Middle East"),
    # South Asia
    "hi": Language("hi", "Hindi", "Indo-European", "Devanagari", "Fusional", "high", "South Asia"),
    "mr": Language("mr", "Marathi", "Indo-European", "Devanagari", "Fusional", "medium", "South Asia"),
    "bn": Language("bn", "Bengali", "Indo-European", "Bengali", "Fusional", "medium", "South Asia"),
    "gu": Language("gu", "Gujarati", "Indo-European", "Gujarati", "Fusional", "low", "South Asia"),
    "pa": Language("pa", "Punjabi", "Indo-European", "Gurmukhi", "Fusional", "low", "South Asia"),
    "ta": Language("ta", "Tamil", "Dravidian", "Tamil", "Agglutinative", "medium", "South Asia"),
    "te": Language("te", "Telugu", "Dravidian", "Telugu", "Agglutinative", "medium", "South Asia"),
    "ne": Language("ne", "Nepali", "Indo-European", "Devanagari", "Fusional", "low", "South Asia"),
    # Southeast Asia
    "tl": Language("tl", "Tagalog", "Austronesian", "Latin", "Austronesian", "medium", "Southeast Asia"),
    "ms": Language("ms", "Malay", "Austronesian", "Latin", "Analytic", "medium", "Southeast Asia"),
    "id": Language("id", "Indonesian", "Austronesian", "Latin", "Analytic", "medium", "Southeast Asia"),
    "vi": Language("vi", "Vietnamese", "Austroasiatic", "Latin", "Analytic", "medium", "Southeast Asia"),
    "jv": Language("jv", "Javanese", "Austronesian", "Latin", "Agglutinative", "low", "Southeast Asia"),
    "km": Language("km", "Khmer", "Austroasiatic", "Khmer", "Analytic", "low", "Southeast Asia"),
    "th": Language("th", "Thai", "Kra-Dai", "Thai", "Analytic", "medium", "Southeast Asia"),
    "lo": Language("lo", "Lao", "Kra-Dai", "Lao", "Analytic", "low", "Southeast Asia"),
    "my": Language("my", "Burmese", "Sino-Tibetan", "Myanmar", "Agglutinative", "low", "Southeast Asia"),
    # East Asia
    "zh": Language("zh", "Chinese", "Sino-Tibetan", "CJK", "Analytic", "high", "East Asia"),
    "ja": Language("ja", "Japanese", "Japonic", "CJK+Kana", "Agglutinative", "high", "East Asia"),
    "ko": Language("ko", "Korean", "Koreanic", "Hangul", "Agglutinative", "high", "East Asia"),
    # Africa
    "am": Language("am", "Amharic", "Afroasiatic", "Ethiopic", "Fusional", "low", "Africa"),
    "ha": Language("ha", "Hausa", "Afroasiatic", "Latin", "Fusional", "low", "Africa"),
    "ig": Language("ig", "Igbo", "Niger-Congo", "Latin", "Analytic", "low", "Africa"),
    "mg": Language("mg", "Malagasy", "Austronesian", "Latin", "Austronesian", "low", "Africa"),
    "sn": Language("sn", "Shona", "Niger-Congo", "Latin", "Agglutinative", "low", "Africa"),
    "sw": Language("sw", "Swahili", "Niger-Congo", "Latin", "Agglutinative", "low", "Africa"),
    "wo": Language("wo", "Wolof", "Niger-Congo", "Latin", "Agglutinative", "low", "Africa"),
    "xh": Language("xh", "Xhosa", "Niger-Congo", "Latin", "Agglutinative", "low", "Africa"),
    "yo": Language("yo", "Yoruba", "Niger-Congo", "Latin", "Analytic", "low", "Africa"),
    "zu": Language("zu", "Zulu", "Niger-Congo", "Latin", "Agglutinative", "low", "Africa"),
}

LANGUAGE_CODES: list[str] = sorted(LANGUAGES.keys())
LANGUAGE_NAMES: dict[str, str] = {k: v.name for k, v in LANGUAGES.items()}

# Group by family
LANGUAGE_FAMILIES: dict[str, list[str]] = {}
for _code, _lang in LANGUAGES.items():
    LANGUAGE_FAMILIES.setdefault(_lang.family, []).append(_code)

# Group by resourcedness (per Cohere checklist)
RESOURCE_GROUPS: dict[str, list[str]] = {}
for _code, _lang in LANGUAGES.items():
    RESOURCE_GROUPS.setdefault(_lang.resourcedness, []).append(_code)

# Group by region
REGION_GROUPS: dict[str, list[str]] = {}
for _code, _lang in LANGUAGES.items():
    REGION_GROUPS.setdefault(_lang.region, []).append(_code)

# Group by script
SCRIPT_GROUPS: dict[str, list[str]] = {}
for _code, _lang in LANGUAGES.items():
    SCRIPT_GROUPS.setdefault(_lang.script, []).append(_code)

# Original 10 research-focus languages for distillation experiments
DISTILLATION_LANGUAGES: list[str] = [
    "ar", "en", "es", "hi", "id", "ja", "sw", "te", "tr", "zh",
]

# Latin-script languages (for script consistency checks in testing)
LATIN_SCRIPT_LANGUAGES: frozenset[str] = frozenset(
    code for code, lang in LANGUAGES.items() if lang.script == "Latin"
)
