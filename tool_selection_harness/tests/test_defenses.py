"""Unit tests for detector interfaces and threshold calibration.

The LM-based detector scoring paths are tested with stubbed
tokenizer/model objects (no model download); the threshold classifier and
edge cases are tested with numeric scores. Defensive benchmarking
utilities only.
"""

import math
from typing import List

import numpy as np
import pytest

from tool_selection_harness.core import ToolDocument
from tool_selection_harness.core.defenses import (
    Detector,
    KnownAnswerDetector,
    PerplexityDetector,
    PerplexityWindowedDetector,
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
    """Logits shaped like a real LM (one distribution per input token,
    predicting the next token) so the per-token NLL of each predicted
    token equals that token's id value (see _token_nlls consumption)."""

    VOCAB = 8

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def eval(self):
        return self

    def __call__(self, input_ids=None, labels=None, **kwargs):
        length = input_ids.shape[-1]
        logits = torch.zeros(1, length, self.VOCAB)
        for position in range(length - 1):
            label = int(input_ids[0, position + 1])
            nll = float(input_ids[0, position + 1])
            logits[0, position, label] = math.log(
                (self.VOCAB - 1) / (math.exp(nll) - 1)
            )
        loss = None
        if labels is not None:
            loss = labels.float().mean()
        return type("Outputs", (), {"logits": logits, "loss": loss})()


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
    doc = ToolDocument("tool", "ab cd")  # per-token NLLs [2.0] -> mean 2.0
    assert detector.score(doc) == pytest.approx(2.0)


def test_perplexity_detector_uses_description_not_name(stub_transformers) -> None:
    detector = PerplexityDetector()
    doc = ToolDocument("tool", "xyz abc")  # NLLs [3.0] -> mean 3.0
    assert detector.score(doc) == pytest.approx(3.0)


def test_perplexity_detector_single_token_is_zero(stub_transformers) -> None:
    detector = PerplexityDetector()
    assert detector.score(ToolDocument("tool", "xyz")) == 0.0


# -- PerplexityWindowedDetector ---------------------------------------------------


def test_windowed_detector_returns_max_window_mean(stub_transformers) -> None:
    detector = PerplexityWindowedDetector(window_size=1)
    doc = ToolDocument("tool", "ab cd xyz")  # NLLs [2.0, 3.0] -> max 3.0
    assert detector.score(doc) == pytest.approx(3.0)


def test_windowed_detector_windows_average(stub_transformers) -> None:
    detector = PerplexityWindowedDetector(window_size=2)
    doc = ToolDocument("tool", "ab cd xyz")  # windows [2,3] -> max 2.5
    assert detector.score(doc) == pytest.approx(2.5)


def test_windowed_detector_short_text_uses_single_window(stub_transformers) -> None:
    detector = PerplexityWindowedDetector(window_size=5)
    doc = ToolDocument("tool", "ab cd")  # fewer tokens than window
    assert detector.score(doc) == pytest.approx(2.0)


def test_windowed_detector_rejects_bad_window_size() -> None:
    with pytest.raises(ValueError):
        PerplexityWindowedDetector(window_size=0)


# -- KnownAnswerDetector -----------------------------------------------------------


def test_known_answer_detector_accepts_secret_response() -> None:
    def llm_call(prompt: str) -> str:
        assert "Hello World!" in prompt
        return "Hello World!"

    detector = KnownAnswerDetector(llm_call=llm_call)
    assert detector.score(ToolDocument("t", "d")) == 0.0


def test_known_answer_detector_flags_broken_response() -> None:
    detector = KnownAnswerDetector(llm_call=lambda prompt: "Some other output")
    assert detector.score(ToolDocument("t", "d")) == 1.0


def test_known_answer_detector_fails_safe_on_backend_error() -> None:
    def broken(prompt: str) -> str:
        raise RuntimeError("api down")

    detector = KnownAnswerDetector(llm_call=broken)
    assert detector.score(ToolDocument("t", "d")) == 1.0


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
