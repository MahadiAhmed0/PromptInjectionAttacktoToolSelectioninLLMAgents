"""Pluggable anomaly detectors for tool documents.

Detectors assign a suspicion score to a tool document (higher = more
suspicious). They are used in defensive evaluations of tool-selection
robustness: researchers measure how well a scoring rule separates ordinary
tool documents from researcher-supplied test documents (e.g., injected or
variant descriptions introduced for benchmarking). This module contains
scoring and calibration utilities only; it does not generate attacks.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from tool_selection_harness.core.tool_document import ToolDocument


@runtime_checkable
class Detector(Protocol):
    """Interface for document anomaly detectors.

    Implementations must provide :meth:`score`, returning a float where
    higher values indicate more suspicious documents.
    """

    def score(self, doc: ToolDocument) -> float:
        """Return a suspicion score for ``doc`` (higher = more suspicious)."""
        ...


class PerplexityDetector:
    """Score documents by description perplexity under a small local LM.

    Computes the average token negative log-likelihood (NLL) of the
    ``tool_description`` under a causal language model (default: gpt2).
    Unusual or incoherent descriptions tend to receive higher NLL, making
    this a simple baseline anomaly score. The model is loaded lazily on the
    first call.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _load(self):
        """Load tokenizer + model on first use (lazy import)."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ImportError(
                    "PerplexityDetector requires the 'transformers' and "
                    "'torch' packages. Install with "
                    "`pip install transformers torch`."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
        return self._tokenizer, self._model

    def score(self, doc: ToolDocument) -> float:
        """Return mean token NLL of ``doc.tool_description``."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "PerplexityDetector requires 'torch'. Install with "
                "`pip install torch`."
            ) from exc

        tokenizer, model = self._load()
        inputs = tokenizer(doc.tool_description, return_tensors="pt")
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
        return float(outputs.loss.item())


class ThresholdClassifier:
    """Binary flagging based on a calibrated score threshold.

    ``fit_threshold`` calibrates the threshold from scores of known-benign
    documents so that approximately ``target_fpr`` of benign documents would
    be flagged. ``classify`` flags a document when its detector score
    exceeds the threshold.

    Args:
        detector: Optional detector used by :meth:`classify` to score
            documents. If omitted, only the threshold-fitting half is
            usable.
        threshold: Optional pre-set threshold (e.g., loaded from a previous
            calibration).
    """

    def __init__(
        self,
        detector: Optional[Detector] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.detector = detector
        self.threshold = threshold

    def fit_threshold(self, scores: List[float], target_fpr: float) -> float:
        """Calibrate the threshold from known-benign scores.

        Args:
            scores: Detector scores over a calibration set of known-benign
                documents.
            target_fpr: Desired false positive rate on the calibration set
                (fraction of benign documents flagged).

        Returns:
            The fitted threshold, also stored on the instance.
        """
        if not scores:
            raise ValueError("scores must not be empty")
        if not 0.0 <= target_fpr <= 1.0:
            raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
        self.threshold = _threshold_for_fpr(scores, target_fpr)
        return self.threshold

    def classify(self, doc: ToolDocument) -> bool:
        """Return True when ``doc`` is flagged as suspicious."""
        if self.detector is None:
            raise RuntimeError(
                "ThresholdClassifier has no detector; construct with "
                "ThresholdClassifier(detector=...)"
            )
        if self.threshold is None:
            raise RuntimeError("Threshold not set; call fit_threshold first.")
        return self.detector.score(doc) > self.threshold


def _threshold_for_fpr(scores: List[float], target_fpr: float) -> float:
    """Empirical (1 - fpr) quantile of benign scores, with edge handling."""
    import numpy as np

    arr = np.asarray(scores, dtype=np.float64)
    if target_fpr <= 0.0:
        return float(np.nextafter(arr.max(), np.inf))
    if target_fpr >= 1.0:
        return float(np.nextafter(arr.min(), -np.inf))
    return float(np.quantile(arr, 1.0 - target_fpr))
