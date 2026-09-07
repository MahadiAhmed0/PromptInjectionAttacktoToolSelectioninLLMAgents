"""Unit tests for the benchmark runner orchestration layer.

Uses a deterministic fake retriever and a canned fake selector (no network,
no embeddings) to verify the retrieve -> select loop, results dict contents,
JSON round-trip, and console summary table. Defensive benchmarking only.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from tool_selection_harness.core import (
    BenchmarkRunner,
    SelectionResult,
    ToolDocument,
    ToolLibrary,
)

DOC_A = ToolDocument("tool_a", "description of tool a")
DOC_B = ToolDocument("tool_b", "description of tool b")
DOC_C = ToolDocument("tool_c", "description of tool c")
DOC_T = ToolDocument("tool_t", "test document variant")


class FakeRetriever:
    """Deterministic retriever: returns ranked tool names per query."""

    def __init__(self, rankings: Dict[str, List[str]]) -> None:
        self.rankings = rankings
        self.top_k_calls = 0

    def top_k(self, query: str, library: ToolLibrary, k: int, metric: str = "cosine"):
        self.top_k_calls += 1
        by_name = {doc.tool_name: doc for doc in library.documents}
        names = self.rankings[query][:k]
        return [(by_name[name], 1.0 - i * 0.1) for i, name in enumerate(names)]


class FakeSelector:
    """Returns canned SelectionResult objects keyed by query."""

    def __init__(self, outcomes: Dict[str, SelectionResult]) -> None:
        self.outcomes = outcomes
        self.select_calls = 0

    def select(self, query: str, candidates: List[ToolDocument]) -> SelectionResult:
        self.select_calls += 1
        return self.outcomes[query]


@pytest.fixture
def library() -> ToolLibrary:
    return ToolLibrary(documents=[DOC_A, DOC_B, DOC_C])


@pytest.fixture
def queries() -> List[Tuple[str, str]]:
    return [("q1", "tool_a"), ("q2", "tool_b"), ("q3", "tool_c")]


@pytest.fixture
def retriever() -> FakeRetriever:
    return FakeRetriever(
        {
            "q1": ["tool_a", "tool_b"],
            "q2": ["tool_b", "tool_c"],
            "q3": ["tool_c", "tool_a"],
        }
    )


@pytest.fixture
def selector() -> FakeSelector:
    return FakeSelector(
        {
            "q1": SelectionResult("tool_a", '{"select_tool": "tool_a"}', "success"),
            "q2": SelectionResult(None, "not json at all", "invalid_json"),
            "q3": SelectionResult("tool_b", '{"select_tool": "tool_b"}', "success"),
        }
    )


# -- run loop ----------------------------------------------------------------


def test_run_collects_records_per_query(
    library: ToolLibrary,
    retriever: FakeRetriever,
    selector: FakeSelector,
    queries: List[Tuple[str, str]],
) -> None:
    runner = BenchmarkRunner(
        library=library, retriever=retriever, selector=selector, queries=queries, k=2
    )
    results = runner.run()

    assert retriever.top_k_calls == 3
    assert selector.select_calls == 3
    assert len(results["records"]) == 3

    first = results["records"][0]
    assert first["query"] == "q1"
    assert first["expected_tool"] == "tool_a"
    assert first["test_document"] is None
    assert [doc["tool_name"] for doc in first["retrieved_docs"]] == [
        "tool_a",
        "tool_b",
    ]
    assert first["selection_result"] == {
        "selected_tool_name": "tool_a",
        "raw_output": '{"select_tool": "tool_a"}',
        "status": "success",
    }


def test_run_computes_metrics(
    library: ToolLibrary,
    retriever: FakeRetriever,
    selector: FakeSelector,
    queries: List[Tuple[str, str]],
) -> None:
    runner = BenchmarkRunner(
        library=library, retriever=retriever, selector=selector, queries=queries, k=2
    )
    metrics = runner.run()["metrics"]

    # q1 correct; q2 invalid JSON; q3 selected tool_b instead of tool_c.
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["hit_rate_at_k"] == pytest.approx(1.0)
    assert metrics["target_selection_rate"] == 0.0
    assert metrics["target_retrieval_rate"] == 0.0
    assert metrics["selector_status_counts"] == {
        "success": 2,
        "invalid_json": 1,
        "unknown_tool": 0,
        "refused": 0,
    }


def test_run_config_metadata(
    library: ToolLibrary,
    retriever: FakeRetriever,
    selector: FakeSelector,
    queries: List[Tuple[str, str]],
) -> None:
    runner = BenchmarkRunner(
        library=library, retriever=retriever, selector=selector, queries=queries, k=5
    )
    config = runner.run()["config"]
    assert config == {"k": 5, "num_queries": 3, "test_document": None}


def test_run_with_test_document(
    library: ToolLibrary,
    selector: FakeSelector,
    queries: List[Tuple[str, str]],
) -> None:
    retriever = FakeRetriever(
        {
            "q1": ["tool_a", "tool_t"],
            "q2": ["tool_t", "tool_b"],
            "q3": ["tool_c", "tool_a"],
        }
    )
    selector = FakeSelector(
        {
            "q1": SelectionResult("tool_t", '{"select_tool": "tool_t"}', "success"),
            "q2": SelectionResult("tool_b", '{"select_tool": "tool_b"}', "success"),
            "q3": SelectionResult("tool_c", '{"select_tool": "tool_c"}', "success"),
        }
    )
    runner = BenchmarkRunner(
        library=library,
        retriever=retriever,
        selector=selector,
        queries=queries,
        k=2,
        test_document=DOC_T,
    )
    results = runner.run()

    config = results["config"]
    assert config["test_document"]["tool_name"] == "tool_t"
    assert config["test_document"]["tool_description"] == "test document variant"

    # All records belong to the injection pass; accuracy denominator is empty.
    metrics = results["metrics"]
    assert metrics["accuracy"] == 0.0
    assert metrics["hit_rate_at_k"] == pytest.approx(1.0)
    assert metrics["target_selection_rate"] == pytest.approx(1 / 3)
    assert metrics["target_retrieval_rate"] == pytest.approx(2 / 3)

    assert all(r["test_document"]["tool_name"] == "tool_t" for r in results["records"])


def test_run_does_not_mutate_baseline_library(
    library: ToolLibrary,
    retriever: FakeRetriever,
    selector: FakeSelector,
    queries: List[Tuple[str, str]],
) -> None:
    runner = BenchmarkRunner(
        library=library,
        retriever=retriever,
        selector=selector,
        queries=queries,
        k=2,
        test_document=DOC_T,
    )
    runner.run()
    assert len(library) == 3
    assert library.get("tool_t") is None


def test_run_empty_queries_returns_zero_metrics(
    library: ToolLibrary, retriever: FakeRetriever, selector: FakeSelector
) -> None:
    runner = BenchmarkRunner(
        library=library, retriever=retriever, selector=selector, queries=[], k=2
    )
    results = runner.run()
    assert results["records"] == []
    metrics = results["metrics"]
    assert metrics["accuracy"] == 0.0
    assert metrics["hit_rate_at_k"] == 0.0
    assert metrics["selector_status_counts"]["success"] == 0


# -- persistence --------------------------------------------------------------


def test_save_json_roundtrip(
    library: ToolLibrary,
    retriever: FakeRetriever,
    selector: FakeSelector,
    queries: List[Tuple[str, str]],
    tmp_path: Path,
) -> None:
    runner = BenchmarkRunner(
        library=library, retriever=retriever, selector=selector, queries=queries, k=2
    )
    results = runner.run()
    out_path = tmp_path / "results.json"

    runner.save_json(out_path)

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == results
    assert loaded["metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert len(loaded["records"]) == 3


def test_save_json_accepts_explicit_results(
    library: ToolLibrary,
    retriever: FakeRetriever,
    selector: FakeSelector,
    queries: List[Tuple[str, str]],
    tmp_path: Path,
) -> None:
    runner = BenchmarkRunner(
        library=library, retriever=retriever, selector=selector, queries=queries, k=2
    )
    results = runner.run()
    out_path = tmp_path / "explicit.json"
    runner.save_json(out_path, results=results)
    assert json.loads(out_path.read_text(encoding="utf-8")) == results


def test_save_json_before_run_raises(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        library=ToolLibrary(documents=[DOC_A]),
        retriever=FakeRetriever({}),
        selector=FakeSelector({}),
        queries=[],
        k=2,
    )
    with pytest.raises(RuntimeError):
        runner.save_json(tmp_path / "never.json")


# -- summary table ------------------------------------------------------------


def test_print_summary_baseline(
    library: ToolLibrary,
    retriever: FakeRetriever,
    selector: FakeSelector,
    queries: List[Tuple[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = BenchmarkRunner(
        library=library, retriever=retriever, selector=selector, queries=queries, k=2
    )
    runner.run()
    runner.print_summary()

    out = capsys.readouterr().out
    assert "Tool-Selection Benchmark Results" in out
    assert "k=2  queries=3  test_document=-" in out
    assert "accuracy" in out and "0.3333" in out
    assert "hit_rate_at_k" in out and "1.0000" in out
    assert "target_selection_rate" in out and "n/a" in out
    assert "status_success" in out and "2" in out
    assert "status_invalid_json" in out and "1" in out


def test_print_summary_with_test_document(
    library: ToolLibrary,
    selector: FakeSelector,
    queries: List[Tuple[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    retriever = FakeRetriever(
        {
            "q1": ["tool_t", "tool_a"],
            "q2": ["tool_b", "tool_c"],
            "q3": ["tool_c", "tool_a"],
        }
    )
    runner = BenchmarkRunner(
        library=library,
        retriever=retriever,
        selector=selector,
        queries=queries,
        k=2,
        test_document=DOC_T,
    )
    runner.run()
    runner.print_summary()

    out = capsys.readouterr().out
    assert "test_document=tool_t" in out
    assert "target_selection_rate" in out and "n/a" not in out
    assert "target_retrieval_rate" in out and "0.3333" in out


def test_print_summary_before_run_raises(capsys: pytest.CaptureFixture[str]) -> None:
    runner = BenchmarkRunner(
        library=ToolLibrary(documents=[DOC_A]),
        retriever=FakeRetriever({}),
        selector=FakeSelector({}),
        queries=[],
        k=2,
    )
    with pytest.raises(RuntimeError):
        runner.print_summary()
