"""Unit tests for detection error-rate metrics (FPR/FNR)."""

from typing import List

import pytest

from tool_selection_harness.core import (
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
