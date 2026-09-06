"""Unit tests for the embedding-based tool retriever.

Tests use a deterministic bag-of-words fake embedder (no sentence-transformers
download required) to verify ranking, similarity math, metric validation, and
document-embedding caching. This module evaluates defensive retrieval
benchmarking only; it generates no attack content.
"""

from typing import Callable, List, Tuple

import numpy as np
import pytest

from tool_selection_harness.core import Retriever, ToolDocument, ToolLibrary

STOPWORDS = {
    "the", "a", "an", "for", "in", "is", "of", "to", "and", "or",
    "what", "with", "please", "into",
}


def _tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, and filter stopwords/short tokens."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return [w for w in cleaned.split() if len(w) > 2 and w not in STOPWORDS]


class BagEmbedder:
    """Deterministic one-hot bag-of-words embedder with call counting."""

    def __init__(self, vocab: list[str]) -> None:
        self._index = {word: i for i, word in enumerate(vocab)}
        self.call_count = 0
        self.seen_texts: list[str] = []

    def __call__(self, text: str) -> np.ndarray:
        self.call_count += 1
        self.seen_texts.append(text)
        vec = np.zeros(len(self._index), dtype=np.float32)
        for word in _tokens(text):
            idx = self._index.get(word)
            if idx is not None:
                vec[idx] = 1.0
        return vec


DOCS = {
    "weather_tool": "Fetch current weather conditions city",
    "calendar_tool": "Add calendar event tomorrow meeting",
    "translate_text": "Translate text between languages spanish french",
    "calculate": "Evaluate mathematical expression result number",
}

QUERIES = {
    "weather": "what is the weather forecast in paris",
    "calendar": "add calendar event tomorrow",
}


def _doc_text(name: str, description: str) -> str:
    return f"{name}: {description}"


@pytest.fixture
def vocab() -> list[str]:
    corpus = [_doc_text(n, d) for n, d in DOCS.items()]
    corpus.extend(QUERIES.values())
    return sorted({w for text in corpus for w in _tokens(text)})


@pytest.fixture
def embedder(vocab: list[str]) -> BagEmbedder:
    return BagEmbedder(vocab)


@pytest.fixture
def retriever(embedder: BagEmbedder) -> Retriever:
    return Retriever(embed_fn=embedder)


@pytest.fixture
def library() -> ToolLibrary:
    return ToolLibrary(
        documents=[ToolDocument(name, desc) for name, desc in DOCS.items()]
    )


# -- similarity -------------------------------------------------------------


def test_cosine_identical_vectors_are_one(retriever: Retriever) -> None:
    vec = np.array([1.0, 2.0, 3.0])
    assert retriever.similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_are_zero(retriever: Retriever) -> None:
    assert retriever.similarity(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero(retriever: Retriever) -> None:
    assert retriever.similarity(
        np.zeros(3), np.array([1.0, 1.0, 1.0])
    ) == pytest.approx(0.0)


def test_dot_metric_is_dot_product(retriever: Retriever) -> None:
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert retriever.similarity(a, b, metric="dot") == pytest.approx(11.0)


def test_unknown_metric_raises(retriever: Retriever) -> None:
    with pytest.raises(ValueError):
        retriever.similarity(np.zeros(2), np.zeros(2), metric="euclidean")


# -- embedding --------------------------------------------------------------


def test_embed_document_concatenates_name_and_description(
    retriever: Retriever, embedder: BagEmbedder
) -> None:
    doc = ToolDocument("weather_tool", "Fetch current weather conditions city")
    retriever.embed_document(doc)
    assert embedder.seen_texts[-1] == (
        "weather_tool: Fetch current weather conditions city"
    )


# -- top_k ordering ---------------------------------------------------------


def test_top_k_weather_query_ranks_weather_first(
    retriever: Retriever, library: ToolLibrary
) -> None:
    results = retriever.top_k(QUERIES["weather"], library, k=4)
    names = [doc.tool_name for doc, _ in results]
    assert names[0] == "weather_tool"
    assert names == ["weather_tool", "calendar_tool", "translate_text", "calculate"]
    assert results[0][1] > 0.0
    assert all(score == pytest.approx(0.0) for _, score in results[1:])


def test_top_k_calendar_query_ranks_calendar_first(
    retriever: Retriever, library: ToolLibrary
) -> None:
    results = retriever.top_k(QUERIES["calendar"], library, k=4)
    names = [doc.tool_name for doc, _ in results]
    assert names[0] == "calendar_tool"
    assert names == ["calendar_tool", "weather_tool", "translate_text", "calculate"]
    assert results[0][1] > results[1][1]


def test_top_k_scores_sorted_descending(
    retriever: Retriever, library: ToolLibrary
) -> None:
    results = retriever.top_k(QUERIES["calendar"], library, k=4)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_respects_k(retriever: Retriever, library: ToolLibrary) -> None:
    results = retriever.top_k(QUERIES["weather"], library, k=1)
    assert len(results) == 1
    assert results[0][0].tool_name == "weather_tool"


def test_top_k_dot_metric_ranks_weather_first(
    retriever: Retriever, library: ToolLibrary
) -> None:
    results = retriever.top_k(QUERIES["weather"], library, k=1, metric="dot")
    assert results[0][0].tool_name == "weather_tool"


def test_top_k_larger_than_library_returns_all(
    retriever: Retriever, library: ToolLibrary
) -> None:
    results = retriever.top_k(QUERIES["weather"], library, k=100)
    assert len(results) == len(library)


def test_top_k_empty_library_returns_empty(retriever: Retriever) -> None:
    assert retriever.top_k("any query", ToolLibrary(), k=5) == []


def test_top_k_zero_k_returns_empty(retriever: Retriever, library: ToolLibrary) -> None:
    assert retriever.top_k(QUERIES["weather"], library, k=0) == []


def test_top_k_negative_k_raises(retriever: Retriever, library: ToolLibrary) -> None:
    with pytest.raises(ValueError):
        retriever.top_k(QUERIES["weather"], library, k=-1)


def test_top_k_unknown_metric_raises(
    retriever: Retriever, library: ToolLibrary
) -> None:
    with pytest.raises(ValueError):
        retriever.top_k(QUERIES["weather"], library, k=3, metric="euclidean")


# -- caching ----------------------------------------------------------------


def test_document_embeddings_are_cached(
    retriever: Retriever, embedder: BagEmbedder, library: ToolLibrary
) -> None:
    retriever.top_k(QUERIES["weather"], library, k=4)
    calls_after_first = embedder.call_count  # 4 docs + 1 query

    retriever.top_k(QUERIES["weather"], library, k=4)
    # Only the query is re-embedded; all four documents hit the cache.
    assert embedder.call_count == calls_after_first + 1


def test_cache_reused_across_injected_library(
    retriever: Retriever, embedder: BagEmbedder, library: ToolLibrary
) -> None:
    retriever.top_k(QUERIES["weather"], library, k=4)
    calls_before = embedder.call_count

    injected = ToolDocument("new_tool", "Fetch weather forecast hourly")
    modified = library.inject(injected)
    results = retriever.top_k(QUERIES["weather"], modified, k=5)

    # Only the injected document (plus the query) is newly embedded; the
    # four baseline documents come from the cache.
    assert embedder.call_count == calls_before + 2
    assert len(results) == 5
    assert results[0][0].tool_name == "new_tool"
