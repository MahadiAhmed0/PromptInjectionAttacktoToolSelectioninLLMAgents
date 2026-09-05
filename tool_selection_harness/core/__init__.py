"""Core data structures for the tool-selection robustness harness.

This module is part of a defensive research harness: it exists to support
benchmarking and evaluation of tool retrieval/selection robustness in LLM
agents, not to generate or facilitate attacks.
"""

from tool_selection_harness.core.tool_document import ToolDocument
from tool_selection_harness.core.tool_library import ToolLibrary

__all__ = ["ToolDocument", "ToolLibrary"]
