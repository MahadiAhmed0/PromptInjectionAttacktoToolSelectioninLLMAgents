"""Core data structures for the tool-selection robustness harness.

This module is part of a defensive research harness: it exists to support
benchmarking and evaluation of tool retrieval/selection robustness in LLM
agents, not to generate or facilitate attacks.
"""

from tool_selection_harness.core.defenses import (
    Detector,
    KnownAnswerDetector,
    PerplexityDetector,
    PerplexityWindowedDetector,
    ThresholdClassifier,
)
from tool_selection_harness.core.detection_metrics import (
    confusion_counts,
    detection_auc,
    evaluate_detector,
    false_negative_rate,
    false_positive_rate,
)
from tool_selection_harness.core.metrics import (
    EvalRecord,
    accuracy,
    compute_all,
    hit_rate_at_k,
    status_breakdown,
    target_retrieval_rate,
    target_selection_rate,
)
from tool_selection_harness.core.retriever import Retriever
from tool_selection_harness.core.runner import BenchmarkRunner
from tool_selection_harness.core.selector import SelectionResult, Selector
from tool_selection_harness.core.tool_document import ToolDocument
from tool_selection_harness.core.tool_library import ToolLibrary

__all__ = [
    "ToolDocument",
    "ToolLibrary",
    "Retriever",
    "Selector",
    "SelectionResult",
    "EvalRecord",
    "accuracy",
    "hit_rate_at_k",
    "target_selection_rate",
    "target_retrieval_rate",
    "status_breakdown",
    "compute_all",
    "BenchmarkRunner",
    "Detector",
    "PerplexityDetector",
    "PerplexityWindowedDetector",
    "KnownAnswerDetector",
    "ThresholdClassifier",
    "confusion_counts",
    "false_positive_rate",
    "false_negative_rate",
    "detection_auc",
    "evaluate_detector",
]
