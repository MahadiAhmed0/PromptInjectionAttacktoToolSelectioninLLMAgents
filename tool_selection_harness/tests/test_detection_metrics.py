"""Unit tests for detection error-rate metrics (FPR/FNR) and evaluation."""

from typing import List

import pytest

from tool_selection_harness.core import (
    ToolDocument,
    detection_auc,
    evaluate_detector,
    false_negative_rate,
    false_positive_rate,
)
from tool_selection_harness.core.detection_metrics import confusion_counts


def test_confusion_counts_mixed_set() -> None:
    labels = [False, False, False, True, True]
    predictions = [False, True, False, True, False]
    assert confusion_counts(labels, predictions) == {
        "tp": 1,
        "fp": 1,
        "tn": 2,
        "fn": 1,
    }


def test_false_positive_rate() -> None:
    labels = [False, False, False, True]
    predictions = [True, False, False, True]
    assert false_positive_rate(labels, predictions) == pytest.approx(1 / 3)


def test_false_negative_rate() -> None:
    labels = [False, True, True, True]
    predictions = [False, True, False, False]
    assert false_negative_rate(labels, predictions) == pytest.approx(2 / 3)


def test_false_positive_rate_no_benign_documents_is_zero() -> None:
    labels = [True, True]
    predictions = [True, False]
    assert false_positive_rate(labels, predictions) == 0.0


def test_false_negative_rate_no_test_documents_is_zero() -> None:
    labels = [False, False]
    predictions = [False, True]
    assert false_negative_rate(labels, predictions) == 0.0


def test_empty_lists() -> None:
    assert false_positive_rate([], []) == 0.0
    assert false_negative_rate([], []) == 0.0
    assert confusion_counts([], []) == {"tp": 0, "fp": 0, "tn": 0, "fn": 0}


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        false_positive_rate([True], [])
    with pytest.raises(ValueError):
        false_negative_rate([True, False], [True])


# -- detection_auc --------------------------------------------------------------


def test_detection_auc_perfect_separation_is_one() -> None:
    assert detection_auc([1.0, 2.0, 3.0], [10.0, 11.0]) == pytest.approx(1.0)


def test_detection_auc_overlap_is_between_zero_and_one() -> None:
    auc = detection_auc([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert 0.0 < auc < 1.0


def test_detection_auc_empty_sets_is_zero() -> None:
    assert detection_auc([], [1.0]) == 0.0
    assert detection_auc([1.0], []) == 0.0


# -- evaluate_detector ------------------------------------------------------------


class DictDetector:
    """Scores each document by a fixed numeric value."""

    def __init__(self, scores: List[float]) -> None:
        self.scores = list(scores)
        self.calls = 0

    def score(self, doc: ToolDocument) -> float:
        value = self.scores[self.calls]
        self.calls += 1
        return value


def _docs(n: int, prefix: str = "doc") -> List[ToolDocument]:
    return [ToolDocument(f"{prefix}_{i}", f"description {i}") for i in range(n)]


def test_evaluate_detector_clean_separation() -> None:
    benign, test = _docs(100, "b"), _docs(5, "t")
    detector = DictDetector(
        [float(i) for i in range(100)] + [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
    )
    result = evaluate_detector(detector, benign, test, target_fpr=0.01)
    assert result["fnr"] == pytest.approx(0.0)
    assert result["fpr"] == pytest.approx(0.01, abs=0.02)
    assert result["auc"] == pytest.approx(1.0)
    assert result["counts"]["tp"] == 5


def test_evaluate_detector_misses_everything() -> None:
    benign, test = _docs(10, "b"), _docs(4, "t")
    detector = DictDetector([float(i) for i in range(10)] + [0.0, 0.0, 0.0, 0.0])
    result = evaluate_detector(detector, benign, test, target_fpr=0.1)
    assert result["fnr"] == pytest.approx(1.0)
    assert result["counts"]["fn"] == 4


def test_evaluate_detector_requires_benign_docs() -> None:
    with pytest.raises(ValueError):
        evaluate_detector(DictDetector([]), [], _docs(2), target_fpr=0.01)
