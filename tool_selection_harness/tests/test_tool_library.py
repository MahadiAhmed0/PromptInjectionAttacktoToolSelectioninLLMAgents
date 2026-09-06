"""Unit tests for the tool-selection robustness harness data structures.

These tests verify the defensive benchmarking plumbing: library
construction, copy-on-write semantics (especially for ``inject``), and JSON
persistence. No attack generation is performed or tested here.
"""

from pathlib import Path

import pytest

from tool_selection_harness.core import ToolDocument, ToolLibrary

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "data" / "sample_tools.json"


@pytest.fixture
def doc_a() -> ToolDocument:
    return ToolDocument("tool_a", "Description of tool A")


@pytest.fixture
def doc_b() -> ToolDocument:
    return ToolDocument("tool_b", "Description of tool B", metadata={"cat": "x"})


@pytest.fixture
def library(doc_a: ToolDocument, doc_b: ToolDocument) -> ToolLibrary:
    return ToolLibrary(documents=[doc_a, doc_b])


# -- ToolDocument ---------------------------------------------------------


def test_document_fields(doc_a: ToolDocument) -> None:
    assert doc_a.tool_name == "tool_a"
    assert doc_a.tool_description == "Description of tool A"
    assert doc_a.metadata == {}


def test_document_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ToolDocument("", "desc")


def test_document_rejects_blank_description() -> None:
    with pytest.raises(ValueError):
        ToolDocument("tool_x", "   ")


def test_document_is_frozen(doc_a: ToolDocument) -> None:
    with pytest.raises(Exception):
        doc_a.tool_name = "mutated"  # type: ignore[misc]


# -- ToolLibrary basics ----------------------------------------------------


def test_add_returns_new_library_with_doc(
    library: ToolLibrary, doc_b: ToolDocument
) -> None:
    bigger = library.add(doc_b)
    assert len(library) == 2
    assert len(bigger) == 3
    assert bigger.names() == ["tool_a", "tool_b", "tool_b"]


def test_remove_returns_new_library_without_doc(library: ToolLibrary) -> None:
    smaller = library.remove("tool_a")
    assert len(library) == 2
    assert len(smaller) == 1
    assert smaller.names() == ["tool_b"]


def test_remove_missing_name_raises(library: ToolLibrary) -> None:
    with pytest.raises(KeyError):
        library.remove("does_not_exist")


def test_get_and_contains(library: ToolLibrary, doc_a: ToolDocument) -> None:
    assert library.get("tool_a") == doc_a
    assert library.get("missing") is None
    assert "tool_a" in library
    assert "missing" not in library


# -- inject: copy-on-write semantics --------------------------------------


def test_inject_returns_copy_not_mutation(library: ToolLibrary) -> None:
    injected_doc = ToolDocument("injected", "Suspicious tool description")
    modified = library.inject(injected_doc)

    # Original library must be untouched.
    assert len(library) == 2
    assert library.get("injected") is None

    # Modified library contains the injected document.
    assert len(modified) == 3
    assert modified.get("injected") == injected_doc
    assert modified.names() == ["tool_a", "tool_b", "injected"]


def test_inject_twice_does_not_accumulate_on_original(library: ToolLibrary) -> None:
    first = library.inject(ToolDocument("inj_1", "first"))
    second = first.inject(ToolDocument("inj_2", "second"))
    assert library.names() == ["tool_a", "tool_b"]
    assert first.names() == ["tool_a", "tool_b", "inj_1"]
    assert second.names() == ["tool_a", "tool_b", "inj_1", "inj_2"]


# -- JSON persistence -------------------------------------------------------


def test_load_sample_tools() -> None:
    library = ToolLibrary.load_json(SAMPLE_DATA)
    assert len(library) == 15
    assert "get_current_weather" in library
    assert library.get("get_current_weather") is not None


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    docs = [
        ToolDocument("t1", "desc one"),
        ToolDocument("t2", "desc two", metadata={"cat": "y"}),
    ]
    library = ToolLibrary(documents=docs)
    out_path = tmp_path / "tools.json"

    library.save_json(out_path)

    assert out_path.exists()
    reloaded = ToolLibrary.load_json(out_path)
    assert reloaded == library
    assert reloaded.to_dicts() == library.to_dicts()


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ToolLibrary.load_json(Path("no_such_file.json"))


def test_load_invalid_shape_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError):
        ToolLibrary.load_json(bad_path)
