"""
Multilingual testing extensions for testkitLLM.

Provides language-aware semantic testing, tool calling verification,
reasoning evaluation, and cross-lingual consistency checks for Aetheris
distilled models.

Usage::

    from aya_distill.testing import MultilingualTestSuite, ReasoningTestSuite
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
from .reasoning import (
    ReasoningTestSuite,
    ReasoningResult,
    MultilingualReasoningResult,
)
from .fixtures import (
    MULTILINGUAL_PROMPTS,
    TOOL_SCHEMAS,
    REASONING_SCENARIOS,
)

__all__ = [
    "MultilingualTestSuite",
    "MultilingualTestResult",
    "LanguageTestResult",
    "ToolCallingTest",
    "ToolCallingResult",
    "ReasoningTestSuite",
    "ReasoningResult",
    "MultilingualReasoningResult",
    "MULTILINGUAL_PROMPTS",
    "TOOL_SCHEMAS",
    "REASONING_SCENARIOS",
]
