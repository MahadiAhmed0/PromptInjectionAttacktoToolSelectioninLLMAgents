"""Unit tests for the tool-selection evaluation metrics.

Handcrafted :class:`EvalRecord` objects (no retriever/selector needed) are
used to verify each metric's numerator/denominator semantics, including the
baseline-vs-test-document filtering. Defensive benchmarking metrics only.
"""

from typing import List

import pytest

from tool_selection_harness.core import (
    EvalRecord,
    SelectionResult,
    ToolDocument,
    accuracy,
    compute_all,
    hit_rate_at_k,
    status_breakdown,
    target_retrieval_rate,
    target_selection_rate,
)

DOC_A = ToolDocument("tool_a", "description of tool a")
DOC_B = ToolDocument("tool_b", "description of tool b")
DOC_T = ToolDocument("tool_t", "variant document under test")

SUCCESS_A = SelectionResult("tool_a", '{"select_tool": "tool_a"}', "success")
SUCCESS_B = SelectionResult("tool_b", '{"select_tool": "tool_b"}', "success")
SUCCESS_T = SelectionResult("tool_t", '{"select_tool": "tool_t"}', "success")
INVALID = SelectionResult(None, "not json", "invalid_json")
UNKNOWN = SelectionResult(None, '{"select_tool": "tool_z"}', "unknown_tool")


def make_record(
    retrieved: List[ToolDocument],
    selection: SelectionResult,
    expected: str,
    test_doc: ToolDocument | None = None,
) -> EvalRecord:
    return EvalRecord(
        query="q",
        retrieved_docs=retrieved,
        selection_result=selection,
        expected_tool=expected,
        test_document=test_doc,
    )


# -- accuracy ----------------------------------------------------------------


def test_accuracy_fraction_of_correct_baseline_selections() -> None:
    records = [
        make_record([DOC_A], SUCCESS_A, "tool_a"),            # correct
        make_record([DOC_B], SUCCESS_B, "tool_a"),            # wrong tool
        make_record([DOC_A, DOC_B], SUCCESS_A, "tool_a"),     # correct
    ]
    assert accuracy(records) == pytest.approx(2 / 3)


def test_accuracy_ignores_test_document_records() -> None:
    records = [
        make_record([DOC_A], SUCCESS_A, "tool_a"),                       # correct
        make_record([DOC_T], SUCCESS_T, "tool_t", test_doc=DOC_T),       # ignored
        make_record([DOC_B], SUCCESS_B, "tool_a"),                       # wrong tool
        make_record([DOC_T], SUCCESS_A, "tool_a", test_doc=DOC_T),       # ignored
    ]
    assert accuracy(records) == pytest.approx(0.5)


def test_accuracy_counts_only_successful_selections() -> None:
    records = [
        make_record([DOC_A], INVALID, "tool_a"),     # invalid JSON
        make_record([DOC_A], UNKNOWN, "tool_a"),     # hallucinated tool
        make_record([DOC_A], SUCCESS_A, "tool_a"),   # correct
    ]
    assert accuracy(records) == pytest.approx(1 / 3)


def test_accuracy_no_baseline_records_is_zero() -> None:
    records = [make_record([DOC_T], SUCCESS_T, "tool_t", test_doc=DOC_T)]
    assert accuracy(records) == 0.0


def test_accuracy_empty_results_is_zero() -> None:
    assert accuracy([]) == 0.0


# -- hit_rate_at_k -----------------------------------------------------------


def test_hit_rate_at_k_counts_expected_tool_in_retrieved() -> None:
    records = [
        make_record([DOC_A, DOC_B], SUCCESS_A, "tool_a"),   # hit
        make_record([DOC_B], INVALID, "tool_a"),            # miss
        make_record([DOC_A], SUCCESS_A, "tool_a"),          # hit
    ]
    assert hit_rate_at_k(records) == pytest.approx(2 / 3)


def test_hit_rate_at_k_includes_test_document_records() -> None:
    records = [
        make_record([DOC_A, DOC_B], SUCCESS_A, "tool_a"),                  # hit
        make_record([DOC_T, DOC_A], SUCCESS_T, "tool_t", test_doc=DOC_T),  # hit
        make_record([DOC_B], SUCCESS_B, "tool_t", test_doc=DOC_T),         # miss
    ]
    assert hit_rate_at_k(records) == pytest.approx(2 / 3)


def test_hit_rate_at_k_empty_results_is_zero() -> None:
    assert hit_rate_at_k([]) == 0.0


# -- target_selection_rate ----------------------------------------------------


def test_target_selection_rate_over_test_document_records() -> None:
    records = [
        make_record([DOC_T], SUCCESS_T, "tool_t", test_doc=DOC_T),        # selected target
        make_record([DOC_T, DOC_A], SUCCESS_A, "tool_a", test_doc=DOC_T),  # selected other
        make_record([DOC_A], SUCCESS_A, "tool_a"),                        # baseline, ignored
    ]
    assert target_selection_rate(records) == pytest.approx(0.5)


def test_target_selection_rate_invalid_selection_not_counted() -> None:
    records = [
        make_record([DOC_T], SUCCESS_T, "tool_t", test_doc=DOC_T),   # hit
        make_record([DOC_T], INVALID, "tool_t", test_doc=DOC_T),     # invalid JSON
    ]
    assert target_selection_rate(records) == pytest.approx(0.5)


def test_target_selection_rate_no_test_documents_is_zero() -> None:
    records = [make_record([DOC_A], SUCCESS_A, "tool_a")]
    assert target_selection_rate(records) == 0.0


# -- target_retrieval_rate -----------------------------------------------------


def test_target_retrieval_rate_counts_exact_document_in_retrieved() -> None:
    records = [
        make_record([DOC_T, DOC_A], SUCCESS_A, "tool_a", test_doc=DOC_T),  # retrieved
        make_record([DOC_A, DOC_B], SUCCESS_A, "tool_a", test_doc=DOC_T),  # not retrieved
    ]
    assert target_retrieval_rate(records) == pytest.approx(0.5)


def test_target_retrieval_rate_requires_exact_match() -> None:
    lookalike = ToolDocument("tool_t", "different description")
    records = [
        make_record([lookalike], SUCCESS_T, "tool_t", test_doc=DOC_T),
    ]
    assert target_retrieval_rate(records) == 0.0


def test_target_retrieval_rate_no_test_documents_is_zero() -> None:
    records = [make_record([DOC_A], SUCCESS_A, "tool_a")]
    assert target_retrieval_rate(records) == 0.0


# -- status_breakdown ----------------------------------------------------------


def test_status_breakdown_counts_all_statuses() -> None:
    records = [
        make_record([DOC_A], SUCCESS_A, "tool_a"),
        make_record([DOC_A], SUCCESS_B, "tool_a"),
        make_record([DOC_A], INVALID, "tool_a"),
        make_record([DOC_A], UNKNOWN, "tool_a"),
    ]
    breakdown = status_breakdown(records)
    assert breakdown == {
        "success": 2,
        "invalid_json": 1,
        "unknown_tool": 1,
        "refused": 0,
    }


def test_status_breakdown_empty_results() -> None:
    assert status_breakdown([]) == {
        "success": 0,
        "invalid_json": 0,
        "unknown_tool": 0,
        "refused": 0,
    }


# -- compute_all ----------------------------------------------------------------


def test_compute_all_returns_all_metric_keys() -> None:
    records = [make_record([DOC_A], SUCCESS_A, "tool_a")]
    metrics = compute_all(records)
    assert set(metrics) == {
        "accuracy",
        "hit_rate_at_k",
        "target_selection_rate",
        "target_retrieval_rate",
    }
    assert metrics["accuracy"] == 1.0
    assert metrics["hit_rate_at_k"] == 1.0
