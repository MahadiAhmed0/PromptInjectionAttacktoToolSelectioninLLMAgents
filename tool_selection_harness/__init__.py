"""Tool-selection robustness evaluation harness.

This package provides data structures and utilities for *benchmarking* how
robustly LLM agents select among documented tools when the tool registry may
contain untrusted or injected tool descriptions. It is intended for defensive
research and evaluation only: measuring retrieval/selection accuracy, not
producing exploits.
"""

from tool_selection_harness.core.tool_document import ToolDocument
from tool_selection_harness.core.tool_library import ToolLibrary

__all__ = ["ToolDocument", "ToolLibrary"]
