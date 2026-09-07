"""Unit tests for detector interfaces and threshold calibration.

The PerplexityDetector scoring path is tested with stubbed tokenizer/model
objects (no model download); the threshold classifier and edge cases are
tested with numeric scores. Defensive benchmarking utilities only.
"""

from typing import List

import numpy as np
import pytest

from tool_selection_harness.core import ToolDocument
from tool_selection_harness.core.defenses import (
    Detector,
    PerplexityDetector,
    ThresholdClassifier,
    _threshold_for_fpr,
)

torch = pytest.importorskip("torch")


class StubTokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor(
            [[float(len(token)) for token in text.split()]], dtype=torch.long
        )
        return {"input_ids": ids}


class StubModel:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def eval(self):
        return self

    def __call__(self, **kwargs):
        labels = kwargs["labels"].float()
        loss = labels.mean()
        return type("Outputs", (), {"loss": loss})()


@pytest.fixture
def stub_transformers(monkeypatch):
    import types

    fake = types.SimpleNamespace(
        AutoTokenizer=StubTokenizer, AutoModelForCausalLM=StubModel
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
    return fake


# -- Detector protocol ---------------------------------------------------------


def test_detector_protocol_accepts_score_implementations() -> None:
    class ConstantDetector:
        def score(self, doc: ToolDocument) -> float:
            return 0.5

    detector: Detector = ConstantDetector()
    assert detector.score(ToolDocument("t", "d")) == 0.5


# -- PerplexityDetector ---------------------------------------------------------


def test_perplexity_detector_scores_mean_nll(stub_transformers) -> None:
    detector = PerplexityDetector()
    doc = ToolDocument("tool", "ab cd")  # token lengths 2 and 2 -> mean 2.0
    assert detector.score(doc) == pytest.approx(2.0)


def test_perplexity_detector_uses_description_not_name(stub_transformers) -> None:
    detector = PerplexityDetector()
    doc = ToolDocument("tool", "xyz")  # single token of length 3
    assert detector.score(doc) == pytest.approx(3.0)


# -- ThresholdClassifier ---------------------------------------------------------


def test_fit_threshold_approximates_target_fpr() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=10_000).tolist()
    threshold = _threshold_for_fpr(scores, 0.10)
    flagged = sum(1 for s in scores if s > threshold)
    assert flagged / len(scores) == pytest.approx(0.10, abs=0.02)


def test_fit_threshold_zero_fpr_flags_nothing() -> None:
    threshold = _threshold_for_fpr([1.0, 2.0, 3.0], 0.0)
    assert all(s <= threshold for s in [1.0, 2.0, 3.0])


def test_fit_threshold_one_fpr_flags_everything() -> None:
    threshold = _threshold_for_fpr([1.0, 2.0, 3.0], 1.0)
    assert all(s > threshold for s in [1.0, 2.0, 3.0])


def test_fit_threshold_rejects_empty_scores() -> None:
    classifier = ThresholdClassifier()
    with pytest.raises(ValueError):
        classifier.fit_threshold([], 0.1)


def test_fit_threshold_rejects_out_of_range_fpr() -> None:
    classifier = ThresholdClassifier()
    with pytest.raises(ValueError):
        classifier.fit_threshold([1.0, 2.0], 1.5)


def test_fit_threshold_sets_and_returns_threshold() -> None:
    classifier = ThresholdClassifier()
    threshold = classifier.fit_threshold([1.0, 2.0, 3.0], 0.0)
    assert classifier.threshold == threshold


def test_classify_without_detector_raises() -> None:
    classifier = ThresholdClassifier()
    classifier.fit_threshold([1.0, 2.0], 0.5)
    with pytest.raises(RuntimeError):
        classifier.classify(ToolDocument("t", "d"))


def test_classify_before_fit_raises() -> None:
    class FakeDetector:
        def score(self, doc: ToolDocument) -> float:
            return 1.0

    classifier = ThresholdClassifier(detector=FakeDetector())
    with pytest.raises(RuntimeError):
        classifier.classify(ToolDocument("t", "d"))


def test_classify_flags_scores_above_threshold() -> None:
    class FakeDetector:
        def score(self, doc: ToolDocument) -> float:
            return {"low": 1.0, "high": 5.0}[doc.tool_name]

    classifier = ThresholdClassifier(detector=FakeDetector())
    classifier.fit_threshold([1.0, 1.0, 2.0], 0.5)
    assert classifier.classify(ToolDocument("low", "d")) is False
    assert classifier.classify(ToolDocument("high", "d")) is True
