"""Embedding-based tool retriever for the tool-selection robustness harness.

:class:`Retriever` ranks the documents in a :class:`ToolLibrary` by
similarity between an embedded user query and embedded tool documents. It is
part of a defensive research harness: retrieval quality is measured under
baseline and modified libraries to study tool-selection robustness in LLM
agents. This module performs ranking and bookkeeping only; it contains no
attack tooling.

Dependencies are deliberately light: ``numpy`` plus an embedding backend.
The default backend is sentence-transformers' ``all-MiniLM-L6-v2`` (lazy
imported, so the module is importable without it as long as a custom
embedding callback is supplied).
"""

from __future__ import annotations

import hashlib
import re
import zlib
from typing import Callable, List, Tuple

import numpy as np

from tool_selection_harness.core.tool_document import ToolDocument
from tool_selection_harness.core.tool_library import ToolLibrary

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

#: A pluggable embedding callback: takes a single text string and returns a
#: 1-D float vector. Pass any callable with this signature to
#: :class:`Retriever` to swap in a different backend (e.g., a mock embedder
#: in tests or an API-based embedding service).
EmbeddingFn = Callable[[str], np.ndarray]

SUPPORTED_METRICS = ("cosine", "dot")


class Retriever:
    """Ranks tool documents against a query using vector similarity.

    Args:
        embed_fn: Optional pluggable embedding callback mapping text to a
            vector. If omitted, a sentence-transformers backend using
            ``model_name`` is created lazily on first use.
        model_name: Name of the sentence-transformers model used by the
            default backend (ignored when ``embed_fn`` is provided).
    """

    def __init__(
        self,
        embed_fn: EmbeddingFn | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self._embed_fn = embed_fn if embed_fn is not None else _lazy_st_embedder(model_name)
        self._doc_cache: dict[str, np.ndarray] = {}

    # -- embedding --------------------------------------------------------

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a raw user query into a vector."""
        return np.asarray(self._embed_fn(text), dtype=np.float32)

    def embed_document(self, tool_doc: ToolDocument) -> np.ndarray:
        """Embed a tool document.

        The embedded text is the concatenation of ``tool_name`` and
        ``tool_description``, so both the identifier and the semantics
        contribute to retrieval.
        """
        text = f"{tool_doc.tool_name}: {tool_doc.tool_description}"
        return np.asarray(self._embed_fn(text), dtype=np.float32)

    # -- similarity --------------------------------------------------------

    @staticmethod
    def similarity(vec_a: np.ndarray, vec_b: np.ndarray, metric: str = "cosine") -> float:
        """Score two vectors with ``'cosine'`` or ``'dot'`` similarity."""
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}; expected one of {SUPPORTED_METRICS}"
            )
        a = np.asarray(vec_a, dtype=np.float64)
        b = np.asarray(vec_b, dtype=np.float64)
        if metric == "cosine":
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        return float(np.dot(a, b))

    # -- retrieval ----------------------------------------------------------

    def top_k(
        self,
        query: str,
        library: ToolLibrary,
        k: int,
        metric: str = "cosine",
    ) -> List[Tuple[ToolDocument, float]]:
        """Return the ``k`` most similar documents for ``query``.

        Results are sorted in descending order of similarity score. Document
        embeddings are cached per unique (tool_name, tool_description)
        content hash, so repeated ``top_k`` calls against the same (or
        partially overlapping) libraries do not re-embed unchanged
        documents.

        Args:
            query: Natural-language query string.
            library: Library of tool documents to rank.
            k: Number of top results to return. If larger than the library
                size, all documents are returned.
            metric: ``'cosine'`` or ``'dot'``.
        """
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}; expected one of {SUPPORTED_METRICS}"
            )
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if k == 0 or not library.documents:
            return []

        query_vec = self.embed_query(query)
        scored: list[tuple[ToolDocument, float]] = []
        for doc in library.documents:
            doc_vec = self._cached_document_embedding(doc)
            score = self.similarity(query_vec, doc_vec, metric=metric)
            scored.append((doc, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    # -- cache management ----------------------------------------------------

    def _cached_document_embedding(self, doc: ToolDocument) -> np.ndarray:
        """Return the cached embedding for ``doc``, computing it if needed."""
        key = _document_cache_key(doc)
        cached = self._doc_cache.get(key)
        if cached is None:
            cached = self.embed_document(doc)
            self._doc_cache[key] = cached
        return cached

    def clear_cache(self) -> None:
        """Drop all cached document embeddings."""
        self._doc_cache.clear()


def hashing_embedder(n: int = 3, dim: int = 256) -> EmbeddingFn:
    """Deterministic, dependency-free embedder via word + character n-gram hashing.

    Useful as an offline fallback (no model downloads) for experiments,
    demos, and tests. Text is represented by feature-hashed word unigrams,
    word bigrams, and character ``n``-grams, L2-normalized so cosine
    similarity behaves well. Deterministic across processes and platforms
    (uses CRC32, not Python's salted ``hash``).
    """

    def embed(text: str) -> np.ndarray:
        lowered = text.lower()
        features: list[str] = []
        words = re.findall(r"[a-z0-9]+", lowered)
        features.extend(words)
        features.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
        features.extend(lowered[i : i + n] for i in range(len(lowered) - n + 1))
        if not features:
            features = [lowered]
        vec = np.zeros(dim, dtype=np.float32)
        for feature in features:
            vec[zlib.crc32(feature.encode("utf-8")) % dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    return embed


def _document_cache_key(doc: ToolDocument) -> str:
    """Hash a document's identity (name + description) for cache lookup.

    The hash covers exactly the fields that participate in embedding, so two
    documents that would produce identical embeddings share a cache entry.
    """
    payload = f"{doc.tool_name}\x00{doc.tool_description}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lazy_st_embedder(model_name: str) -> EmbeddingFn:
    """Build the default sentence-transformers backend, imported lazily.

    The model is downloaded/loaded only on the first call, keeping import of
    this module cheap and making ``embed_fn`` the only hard dependency for
    users who supply their own backend.
    """
    model = None

    def embed(text: str) -> np.ndarray:
        nonlocal model
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ImportError(
                    "The default retriever backend requires the "
                    "'sentence-transformers' package. Install it with "
                    "`pip install sentence-transformers` or pass a custom "
                    "embed_fn to Retriever."
                ) from exc
            model = SentenceTransformer(model_name)
        return np.asarray(model.encode(text), dtype=np.float32)

    return embed
