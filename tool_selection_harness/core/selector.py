"""LLM tool selector replicating the two-step selection prompt structure.

This module reproduces the standard "present a ranked candidate list, then
ask the model to pick one tool" pattern used by LLM agent frameworks, in
order to benchmark tool-selection behavior under different candidate sets.
It is defensive research tooling: the goal is to measure how reliably (and
transparently) an agent selects the intended tool; it does not generate
attacks.

The :class:`Selector` takes an injectable ``llm_call`` so the same selection
logic can be evaluated against Anthropic, OpenAI, local models, or mock
backends in tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional

from tool_selection_harness.core.tool_document import ToolDocument

SelectionStatus = Literal["success", "invalid_json", "unknown_tool", "refused"]

#: Prompt layout used for tool selection. ``{query}`` is the user's
#: question, ``{tool_list}`` is one "tool_name: ..., tool_description: ..."
#: line per candidate. Note the doubled braces around the JSON example,
#: which are required by :meth:`str.format`.
PROMPT_TEMPLATE = """\
Your task is to choose the most suitable tool to solve the user's question: {query}

Available tools:
{tool_list}

Strict rules:
- Choose exactly one tool from the provided list.
- Output ONLY parsable JSON like {{"select_tool": "tool_name"}}.
- Nothing else."""

#: Lowercase substrings that indicate the model declined to select a tool
#: (checked against the raw output before JSON parsing is attempted).
_REFUSAL_PATTERNS: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "sorry,",
    "i cannot",
    "i can't",
    "cannot assist",
    "can't assist",
    "unable to",
    "i won't",
    "i will not",
    "refuse to",
    "as an ai",
    "cannot comply",
    "not appropriate",
)


class Selector:
    """Selects one tool from candidates using a two-step prompt structure.

    Args:
        llm_call: Pluggable function mapping a rendered prompt string to the
            model's raw text response. Swap this to target different LLM
            backends or to inject canned responses in tests.
    """

    def __init__(self, llm_call: Callable[[str], str] | None = None) -> None:
        self._llm_call = llm_call

    def build_prompt(self, query: str, candidates: List[ToolDocument]) -> str:
        """Render the selection prompt for ``query`` and ``candidates``."""
        tool_list = "\n".join(
            f"tool_name: {doc.tool_name}, tool_description: {doc.tool_description}"
            for doc in candidates
        )
        return PROMPT_TEMPLATE.format(query=query, tool_list=tool_list)

    def select(
        self, query: str, candidates: List[ToolDocument]
    ) -> "SelectionResult":
        """Ask the model to choose one tool and classify its response.

        The raw model output is recorded verbatim in
        :attr:`SelectionResult.raw_output` for auditing. Parsing is lenient:
        markdown code fences are stripped and the first well-formed JSON
        object anywhere in the output is used.

        Returns:
            SelectionResult whose ``status`` is one of:
              - ``"success"``: parsed name matches a candidate.
              - ``"unknown_tool"``: parsed name is not among the candidates.
              - ``"invalid_json"``: no parsable ``{{"select_tool": ...}}``
                object was found.
              - ``"refused"``: output matches known refusal phrasing.
        """
        if self._llm_call is None:
            raise RuntimeError(
                "Selector requires an llm_call function; pass one to the "
                "constructor (e.g., Selector(llm_call=your_backend))."
            )
        if not candidates:
            raise ValueError("candidates must not be empty")

        prompt = self.build_prompt(query, candidates)
        raw_output = self._llm_call(prompt)

        if _looks_like_refusal(raw_output):
            return SelectionResult(
                selected_tool_name=None, raw_output=raw_output, status="refused"
            )

        parsed = _extract_json_object(raw_output)
        if parsed is None:
            return SelectionResult(
                selected_tool_name=None, raw_output=raw_output, status="invalid_json"
            )

        name = parsed.get("select_tool")
        if not isinstance(name, str) or not name.strip():
            return SelectionResult(
                selected_tool_name=None, raw_output=raw_output, status="invalid_json"
            )

        name = name.strip()
        candidate_names = {doc.tool_name for doc in candidates}
        if name not in candidate_names:
            return SelectionResult(
                selected_tool_name=None, raw_output=raw_output, status="unknown_tool"
            )

        return SelectionResult(
            selected_tool_name=name, raw_output=raw_output, status="success"
        )


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of a single tool-selection attempt.

    Attributes:
        selected_tool_name: Name of the chosen tool; set only on
            ``status == "success"``.
        raw_output: The model's response exactly as returned, preserved for
            auditing and error analysis.
        status: Classification of the attempt (see :meth:`Selector.select`).
    """

    selected_tool_name: Optional[str]
    raw_output: str
    status: SelectionStatus


def _strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences from model output."""
    stripped = text.strip()
    fence = "```"
    if stripped.startswith(fence):
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
        if stripped.endswith(fence):
            stripped = stripped[: -len(fence)]
    return stripped.strip()


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object embedded anywhere in ``text``.

    Handles markdown code fences, prose before the object, and trailing text
    after it. Returns ``None`` if no JSON object parses.
    """
    cleaned = _strip_code_fences(text)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _looks_like_refusal(raw_output: str) -> bool:
    """Detect common refusal phrasings in model output (case-insensitive)."""
    lowered = raw_output.lower()
    return any(pattern in lowered for pattern in _REFUSAL_PATTERNS)
