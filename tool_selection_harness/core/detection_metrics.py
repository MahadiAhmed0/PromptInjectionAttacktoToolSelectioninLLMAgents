"""Binary detection evaluation metrics.

Given detector predictions and ground-truth labels on a mixed set of benign
and test documents, these functions compute standard error rates.

Convention: *positive* = a test document (the kind a researcher expects the
detector to flag, e.g., an injected or variant description); *negative* = a
benign document. Predictions are booleans where ``True`` means flagged.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from tool_selection_harness.core.tool_document import ToolDocument


def confusion_counts(
    labels: List[bool], predictions: List[bool]
) -> Dict[str, int]:
    """Compute true/false positive/negative counts.

    Args:
        labels: Ground truth; ``True`` marks a test document.
        predictions: Detector output; ``True`` marks a flagged document.
    """
    if len(labels) != len(predictions):
        raise ValueError(
            f"labels and predictions must match in length, got "
            f"{len(labels)} and {len(predictions)}"
        )
    tp = sum(1 for label, pred in zip(labels, predictions) if label and pred)
    fp = sum(1 for label, pred in zip(labels, predictions) if not label and pred)
    tn = sum(1 for label, pred in zip(labels, predictions) if not label and not pred)
    fn = sum(1 for label, pred in zip(labels, predictions) if label and not pred)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def false_positive_rate(
    labels: List[bool], predictions: List[bool]
) -> float:
    """Fraction of benign documents that were flagged (fp / negatives).

    Returns 0.0 when the set contains no benign documents.
    """
    counts = confusion_counts(labels, predictions)
    negatives = counts["fp"] + counts["tn"]
    return counts["fp"] / negatives if negatives else 0.0


def false_negative_rate(
    labels: List[bool], predictions: List[bool]
) -> float:
    """Fraction of test documents that were missed (fn / positives).

    Returns 0.0 when the set contains no test documents.
    """
    counts = confusion_counts(labels, predictions)
    positives = counts["tp"] + counts["fn"]
    return counts["fn"] / positives if positives else 0.0


def detection_auc(
    benign_scores: List[float], test_scores: List[float]
) -> float:
    """ROC AUC of a detector's scores on benign vs. test documents.

    Computed rank-based (Mann-Whitney U): the fraction of benign-test pairs
    where the test document scores higher, with ties counting half. Returns
    0.0 when either set is empty.
    """
    if not benign_scores or not test_scores:
        return 0.0
    benign = np.asarray(benign_scores, dtype=np.float64)
    test = np.asarray(test_scores, dtype=np.float64)
    auc = float(
        np.mean(
            [
                np.mean(test > score) + 0.5 * np.mean(test == score)
                for score in benign
            ]
        )
    )
    return auc


def evaluate_detector(
    detector: Any,
    benign_docs: List[ToolDocument],
    test_docs: List[ToolDocument],
    target_fpr: float = 0.01,
) -> Dict[str, Any]:
    """Evaluate a detector on a labeled mixed set (the paper's Table X).

    Calibrates the threshold on the benign set via the dataset-adaptive
    strategy (FPR on calibration set approximately ``target_fpr``), then
    reports FPR/FNR on the mixed set, the ROC AUC, the fitted threshold,
    confusion counts, and the raw scores.

    Args:
        detector: Any object with ``score(doc) -> float``.
        benign_docs: Known-benign calibration documents.
        test_docs: Test documents (e.g., injected or variant tool docs).
        target_fpr: Desired calibration FPR (the paper uses 1%).

    Raises:
        ValueError: If ``benign_docs`` is empty.
    """
    from tool_selection_harness.core.defenses import ThresholdClassifier

    if not benign_docs:
        raise ValueError("benign_docs must not be empty")
    benign_scores = [float(detector.score(doc)) for doc in benign_docs]
    test_scores = [float(detector.score(doc)) for doc in test_docs]

    threshold = ThresholdClassifier().fit_threshold(benign_scores, target_fpr)
    labels = [False] * len(benign_scores) + [True] * len(test_scores)
    predictions = [score > threshold for score in benign_scores + test_scores]

    return {
        "target_fpr": target_fpr,
        "threshold": threshold,
        "fpr": false_positive_rate(labels, predictions),
        "fnr": false_negative_rate(labels, predictions),
        "auc": detection_auc(benign_scores, test_scores),
        "counts": confusion_counts(labels, predictions),
        "benign_scores": benign_scores,
        "test_scores": test_scores,
    }
