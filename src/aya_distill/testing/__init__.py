"""
Multilingual testing extensions for testkitLLM.

Provides language-aware semantic testing, tool calling verification,
and cross-lingual consistency checks for Aetheris distilled models.

Usage::

    from aya_distill.testing import MultilingualTestSuite
    suite = MultilingualTestSuite(model_fn=my_model_fn, languages=["en", "es", "hi"])
    results = suite.run()
"""
from .multilingual import (
    MultilingualTestSuite,
    MultilingualTestResult,
    LanguageTestResult,
)
from .tool_testing import (
    ToolCallingTest,
    ToolCallingResult,
)
from .fixtures import (
    MULTILINGUAL_PROMPTS,
    TOOL_SCHEMAS,
)

__all__ = [
    "MultilingualTestSuite",
    "MultilingualTestResult",
    "LanguageTestResult",
    "ToolCallingTest",
    "ToolCallingResult",
    "MULTILINGUAL_PROMPTS",
    "TOOL_SCHEMAS",
]
