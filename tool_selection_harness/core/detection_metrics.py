"""Binary detection evaluation metrics.

Given detector predictions and ground-truth labels on a mixed set of benign
and test documents, these functions compute standard error rates.

Convention: *positive* = a test document (the kind a researcher expects the
detector to flag, e.g., an injected or variant description); *negative* = a
benign document. Predictions are booleans where ``True`` means flagged.
"""

from __future__ import annotations

from typing import Dict, List


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
