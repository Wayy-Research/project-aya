"""Prompt templates for multilingual benchmarks."""
from .mgsm import build_mgsm_prompt, MGSM_EXEMPLARS
from .xcopa import build_xcopa_prompt, XCOPA_EXEMPLARS

__all__ = [
    "build_mgsm_prompt",
    "MGSM_EXEMPLARS",
    "build_xcopa_prompt",
    "XCOPA_EXEMPLARS",
]
