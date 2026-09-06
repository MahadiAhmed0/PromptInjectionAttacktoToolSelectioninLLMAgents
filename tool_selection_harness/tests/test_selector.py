"""Unit tests for the two-step tool selector.

Uses a canned ``llm_call`` stub (no network) to verify prompt rendering,
lenient JSON parsing, candidate validation, and status classification of
model outputs. Defensive benchmarking only: these tests exercise the
selection harness, not attack generation.
"""

from typing import Callable, List

import pytest

from tool_selection_harness.core import SelectionResult, Selector, ToolDocument

QUERY = "What is the weather in Paris today?"


@pytest.fixture
def candidates() -> List[ToolDocument]:
    return [
        ToolDocument("weather_tool", "Fetch current weather conditions"),
        ToolDocument("calendar_tool", "Add calendar event"),
        ToolDocument("translate_text", "Translate text between languages"),
    ]


class CannedLLM:
    """Records prompts and returns a fixed canned response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def select_with(
    response: str, candidates: List[ToolDocument]
) -> tuple[SelectionResult, CannedLLM]:
    llm = CannedLLM(response)
    selector = Selector(llm_call=llm)
    result = selector.select(QUERY, candidates)
    return result, llm


# -- success cases ----------------------------------------------------------


def test_valid_json_is_success(candidates: List[ToolDocument]) -> None:
    result, _ = select_with('{"select_tool": "weather_tool"}', candidates)
    assert result.status == "success"
    assert result.selected_tool_name == "weather_tool"
    assert result.raw_output == '{"select_tool": "weather_tool"}'


def test_fenced_json_is_success(candidates: List[ToolDocument]) -> None:
    response = '```json\n{"select_tool": "calendar_tool"}\n```'
    result, _ = select_with(response, candidates)
    assert result.status == "success"
    assert result.selected_tool_name == "calendar_tool"


def test_trailing_text_after_json_is_success(candidates: List[ToolDocument]) -> None:
    response = '{"select_tool": "translate_text"} Hope this helps!'
    result, _ = select_with(response, candidates)
    assert result.status == "success"
    assert result.selected_tool_name == "translate_text"


def test_whitespace_around_json_is_success(candidates: List[ToolDocument]) -> None:
    response = '\n\n  {"select_tool": "weather_tool"}  \n'
    result, _ = select_with(response, candidates)
    assert result.status == "success"
    assert result.selected_tool_name == "weather_tool"


# -- invalid_json cases ------------------------------------------------------


def test_malformed_json_is_invalid_json(candidates: List[ToolDocument]) -> None:
    result, _ = select_with("definitely not JSON output here", candidates)
    assert result.status == "invalid_json"
    assert result.selected_tool_name is None
    assert result.raw_output == "definitely not JSON output here"


def test_missing_key_is_invalid_json(candidates: List[ToolDocument]) -> None:
    result, _ = select_with('{"tool": "weather_tool"}', candidates)
    assert result.status == "invalid_json"


def test_non_string_value_is_invalid_json(candidates: List[ToolDocument]) -> None:
    result, _ = select_with('{"select_tool": 42}', candidates)
    assert result.status == "invalid_json"


def test_empty_name_is_invalid_json(candidates: List[ToolDocument]) -> None:
    result, _ = select_with('{"select_tool": ""}', candidates)
    assert result.status == "invalid_json"


def test_empty_response_is_invalid_json(candidates: List[ToolDocument]) -> None:
    result, _ = select_with("", candidates)
    assert result.status == "invalid_json"


def test_json_list_is_invalid_json(candidates: List[ToolDocument]) -> None:
    result, _ = select_with('["weather_tool"]', candidates)
    assert result.status == "invalid_json"


# -- unknown_tool cases ------------------------------------------------------


def test_hallucinated_tool_is_unknown_tool(candidates: List[ToolDocument]) -> None:
    result, _ = select_with('{"select_tool": "nonexistent_tool"}', candidates)
    assert result.status == "unknown_tool"
    assert result.selected_tool_name is None
    assert result.raw_output == '{"select_tool": "nonexistent_tool"}'


def test_case_sensitive_name_mismatch_is_unknown_tool(
    candidates: List[ToolDocument],
) -> None:
    result, _ = select_with('{"select_tool": "WEATHER_TOOL"}', candidates)
    assert result.status == "unknown_tool"


# -- refusal cases -----------------------------------------------------------


def test_refusal_text_is_refused(candidates: List[ToolDocument]) -> None:
    response = "I'm sorry, but I cannot assist with that request."
    result, _ = select_with(response, candidates)
    assert result.status == "refused"
    assert result.selected_tool_name is None


def test_refusal_text_case_insensitive(candidates: List[ToolDocument]) -> None:
    response = "As an AI, I CANNOT COMPLY with this request."
    result, _ = select_with(response, candidates)
    assert result.status == "refused"


# -- prompt rendering --------------------------------------------------------


def test_prompt_contains_query_and_tool_lines(candidates: List[ToolDocument]) -> None:
    llm = CannedLLM('{"select_tool": "weather_tool"}')
    selector = Selector(llm_call=llm)
    selector.select(QUERY, candidates)

    prompt = llm.prompts[0]
    assert QUERY in prompt
    assert "tool_name: weather_tool, tool_description: Fetch current weather conditions" in prompt
    assert "tool_name: calendar_tool, tool_description: Add calendar event" in prompt
    assert "tool_name: translate_text, tool_description: Translate text between languages" in prompt
    assert "Choose exactly one tool from the provided list" in prompt
    assert '{"select_tool": "tool_name"}' in prompt


# -- input validation ---------------------------------------------------------


def test_empty_candidates_raises() -> None:
    selector = Selector(llm_call=CannedLLM("{}"))
    with pytest.raises(ValueError):
        selector.select(QUERY, [])


def test_missing_llm_call_raises(candidates: List[ToolDocument]) -> None:
    selector = Selector()
    with pytest.raises(RuntimeError):
        selector.select(QUERY, candidates)
