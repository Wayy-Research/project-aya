"""Target languages for Project Aya multilingual evaluation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    family: str
    script: str
    typology: str


LANGUAGES: dict[str, Language] = {
    "en": Language("en", "English", "Indo-European", "Latin", "Analytic"),
    "es": Language("es", "Spanish", "Indo-European", "Latin", "Fusional"),
    "hi": Language("hi", "Hindi", "Indo-European", "Devanagari", "Fusional"),
    "zh": Language("zh", "Mandarin", "Sino-Tibetan", "CJK", "Analytic"),
    "ar": Language("ar", "Arabic", "Afroasiatic", "Arabic", "Fusional"),
    "sw": Language("sw", "Swahili", "Niger-Congo", "Latin", "Agglutinative"),
    "tr": Language("tr", "Turkish", "Turkic", "Latin", "Agglutinative"),
    "ja": Language("ja", "Japanese", "Japonic", "CJK+Kana", "Agglutinative"),
    "id": Language("id", "Indonesian", "Austronesian", "Latin", "Analytic"),
    "te": Language("te", "Telugu", "Dravidian", "Telugu", "Agglutinative"),
}

LANGUAGE_CODES: list[str] = list(LANGUAGES.keys())
LANGUAGE_NAMES: dict[str, str] = {k: v.name for k, v in LANGUAGES.items()}

# Group by family for equity analysis
LANGUAGE_FAMILIES: dict[str, list[str]] = {}
for code, lang in LANGUAGES.items():
    LANGUAGE_FAMILIES.setdefault(lang.family, []).append(code)
