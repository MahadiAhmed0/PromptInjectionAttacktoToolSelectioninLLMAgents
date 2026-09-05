"""Tool document model for the tool-selection robustness harness.

The :class:`ToolDocument` dataclass models a single tool as presented to an
LLM agent: a name plus a natural-language description. This is the unit of
study for defensive benchmarking of tool retrieval/selection robustness --
researchers construct libraries of documents (benign or modified), expose
them to an agent, and measure whether the agent selects the tool that matches
the user's intent.

This module contains no attack tooling; it only defines the data shape used
by evaluation harnesses.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolDocument:
    """A documented tool as seen by an LLM agent during tool selection.

    Instances are immutable (frozen) so that shared documents can be safely
    reused across experiments without accidental mutation.

    Attributes:
        tool_name: Unique identifier for the tool.
        tool_description: Natural-language description used by the agent
            to decide whether to select this tool for a task.
        metadata: Optional extra key-value pairs (e.g., category, tags,
            provenance) used by the evaluation harness for bookkeeping.
    """

    tool_name: str
    tool_description: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that names and descriptions are non-empty strings."""
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not self.tool_description or not self.tool_description.strip():
            raise ValueError("tool_description must be a non-empty string")
