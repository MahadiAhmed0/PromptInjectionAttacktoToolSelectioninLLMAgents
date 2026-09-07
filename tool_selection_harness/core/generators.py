"""Synthetic data generation utilities for the tool-selection harness.

The generators in this module use an LLM backend (injected via ``llm_call``)
to produce realistic evaluation material: diverse user queries for a target
task, and benign tool documents for populating synthetic tool libraries.
The output is validated and normalized into the harness's data structures.

Defensive research note: generated content is used to populate evaluation
sets for benchmarking tool-selection robustness; this module does not
generate attacks.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List

from tool_selection_harness.core.selector import (
    _extract_json_object,
    _strip_code_fences,
)
from tool_selection_harness.core.tool_document import ToolDocument

TASK_QUERY_PROMPT_TEMPLATE = """\
Generate {num} diverse, realistic user queries for the following task:

Task: {target_task}

Requirements:
- Vary phrasing, complexity, and length across the queries.
- Each query must be a single natural-language request a user might make.
- Output ONLY a JSON list of strings, like ["query one", "query two"]."""

TOOL_DOCS_PROMPT_TEMPLATE = """\
Generate {num} plausible tool documents relevant to the example user queries below.

Example queries:
{queries}

Requirements:
- Each tool is a distinct, benign capability an agent might expose.
- Output ONLY a JSON list of objects, where each object has exactly the keys
  "tool_name" and "tool_description".
Example: [{{"tool_name": "get_weather", "tool_description": "Fetch current conditions"}}]"""


def generate_task_descriptions(
    target_task: str, num: int, llm_call: Callable[[str], str]
) -> List[str]:
    """Generate ``num`` diverse user queries for ``target_task``.

    Args:
        target_task: Natural-language description of the task (e.g.,
            "checking the current weather in a city").
        num: Number of queries to request.
        llm_call: Pluggable LLM backend taking a prompt and returning raw
            text (JSON list, fenced or not, or a numbered list).

    Returns:
        Up to ``num`` parsed, non-empty query strings.

    Raises:
        ValueError: If ``num`` is not positive or no queries could be parsed.
    """
    if num <= 0:
        raise ValueError(f"num must be positive, got {num}")
    prompt = TASK_QUERY_PROMPT_TEMPLATE.format(num=num, target_task=target_task)
    raw_output = llm_call(prompt)
    queries = _parse_string_list(raw_output)
    if not queries:
        raise ValueError(
            f"Could not parse any queries from LLM output: {raw_output!r}"
        )
    return queries[:num]


def generate_tool_documents(
    context_queries: List[str], num: int, llm_call: Callable[[str], str]
) -> List[ToolDocument]:
    """Generate plausible benign tool documents relevant to example queries.

    Args:
        context_queries: Example user queries the tools should serve.
        num: Number of tool documents to request.
        llm_call: Pluggable LLM backend taking a prompt and returning raw
            text containing a JSON list of tool documents.

    Returns:
        Up to ``num`` validated :class:`ToolDocument` objects.

    Raises:
        ValueError: If ``num`` is not positive, no documents parse, or a
            returned entry is missing/invalid fields.
    """
    if num <= 0:
        raise ValueError(f"num must be positive, got {num}")
    if not context_queries:
        raise ValueError("context_queries must not be empty")

    query_lines = "\n".join(f"- {q}" for q in context_queries)
    prompt = TOOL_DOCS_PROMPT_TEMPLATE.format(num=num, queries=query_lines)
    raw_output = llm_call(prompt)

    items = _extract_json_list(raw_output)
    if items is None:
        raise ValueError(
            f"Could not parse a JSON list of tool documents from LLM "
            f"output: {raw_output!r}"
        )

    documents: List[ToolDocument] = []
    for item in items[:num]:
        if not isinstance(item, dict):
            raise ValueError(
                f"Each tool document must be a JSON object, got "
                f"{type(item).__name__}: {item!r}"
            )
        try:
            documents.append(
                ToolDocument(
                    tool_name=str(item["tool_name"]),
                    tool_description=str(item["tool_description"]),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Tool document missing required key {exc.args[0]!r}: {item!r}"
            ) from exc
    if not documents:
        raise ValueError("No valid tool documents were returned by the LLM")
    return documents


def _parse_string_list(raw_output: str) -> List[str]:
    """Parse a list of strings from LLM output, tolerantly.

    Tries, in order: a JSON list of strings, a fenced JSON list, then a
    line-based fallback that strips numbering/bullets and quotes.
    """
    cleaned = _strip_code_fences(raw_output)
    as_list = _extract_json_list(cleaned)
    if as_list is not None and all(isinstance(item, str) for item in as_list):
        return [item.strip() for item in as_list if item.strip()]

    queries: List[str] = []
    for line in cleaned.splitlines():
        line = line.strip().lstrip("-*•").strip()
        line = line.lstrip("0123456789").lstrip(".)").strip()
        if line.startswith('"') and line.endswith('"') and len(line) >= 2:
            line = line[1:-1].strip()
        if line:
            queries.append(line)
    return queries


def _extract_json_list(text: str) -> List[Any] | None:
    """Extract the first JSON list embedded anywhere in ``text``."""
    cleaned = _strip_code_fences(text)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return obj
    return None
