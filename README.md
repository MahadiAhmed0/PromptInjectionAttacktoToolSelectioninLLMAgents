# Tool-Selection Robustness Evaluation Harness

A Python research harness for **benchmarking tool-selection robustness in LLM
agents** and for **evaluating detection methods**. It models the standard
"retrieve -> select" pipeline used by LLM agent frameworks (tool documents, an
embedding retriever, a two-step selection prompt) and measures how reliably an
agent selects the intended tool -- including under modified tool registries.

> **Scope note:** this project is defensive research tooling. It exists to
> *measure* selection/retrieval behavior and detection quality. It does not
> generate or optimize attacks.

## What it measures

| Metric | Meaning |
| --- | --- |
| `accuracy` | Fraction of baseline queries where the expected tool was selected |
| `hit_rate_at_k` | Fraction of queries where the expected tool appeared in the retrieved top-k set |
| `target_selection_rate` | Fraction of passes where a researcher-supplied *test document* (e.g., an injected or variant tool description) was the selected tool |
| `target_retrieval_rate` | Fraction of passes where the test document appeared in the top-k set |
| `selector_status_counts` | Breakdown of selector outcomes (`success`, `invalid_json`, `unknown_tool`, `refused`) |
| FNR / FPR | Detector error rates on a labeled mixed set of benign and test documents |

## Project structure

```
tool_selection_harness/
  core/
    tool_document.py     # ToolDocument dataclass (name + description)
    tool_library.py      # ToolLibrary: add/remove/inject/load/save (copy-on-write)
    retriever.py         # Embedding retriever with pluggable embed_fn + cache
    selector.py          # Two-step selection prompt, robust JSON parsing
    metrics.py           # EvalRecord + benchmark metrics
    runner.py            # BenchmarkRunner: retrieve -> select loop + reporting
    generators.py        # LLM-based synthetic query/tool-document generation
    defenses.py          # Detector protocol, PerplexityDetector, ThresholdClassifier
    detection_metrics.py # FPR/FNR for detector evaluation
  data/
    sample_tools.json    # 15 synthetic tool documents across categories
  examples/
    run_benchmark.py     # End-to-end example (see below)
  tests/                 # pytest suite (115 tests, no network required)
```

## Installation

```
pip install -r requirements.txt
# optional, for the perplexity detector and real LLM APIs:
pip install transformers torch openai anthropic
```

## Quick start

```
python examples/run_benchmark.py --task weather --num-queries 10 --k 3
```

The example loads `data/sample_tools.json`, generates (or reuses) user queries
for one task, then runs:

1. **Baseline pass** -- no modifications; reports accuracy and hit rate.
2. **Variant pass** -- inserts one hand-written, benign-looking alternative
   tool document (not an optimized adversarial one) and reports
   `target_selection_rate` / `target_retrieval_rate`.
3. **Detection pass** -- calibrates a `PerplexityDetector` (gpt2) threshold at a
   target FPR on the benign library and reports FNR/FPR for the variant
   document. Skipped gracefully when the model cannot be downloaded.

Use `--offline` to avoid all model downloads (deterministic character/word
n-gram embeddings, no detection pass). Full JSON results are written to
`example_results.json`.

## Tests

```
python -m pytest tool_selection_harness/tests -q
```

All tests use deterministic mock backends (no network, no model downloads).
