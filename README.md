# Tool-Selection Robustness Evaluation Harness

A Python research harness for **benchmarking tool-selection robustness in LLM
agents** and for **evaluating detection methods**. It models the standard
"retrieve -> select" pipeline used by LLM agent frameworks (tool documents, an
embedding retriever, a two-step selection prompt) and measures how reliably an
agent selects the intended tool — including under modified tool registries.

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
  tests/                 # pytest suite (112 tests, no network required)
```

## Installation

```
pip install -r requirements.txt
# optional, for the perplexity detector and real LLM APIs:
pip install transformers torch openai anthropic
```

## Web UI

A Streamlit frontend (`app.py`) exposes the whole harness interactively:

```
streamlit run app.py
```

Four tabs:

- **Tool Library** — load `data/sample_tools.json` or upload a custom JSON,
  edit documents in an editable table, add/remove tools via a form, and
  generate synthetic tools through `core/generators.py` using the selected
  LLM backend.
- **Run Benchmark** — pick the embedding backend (MiniLM or offline
  hashing), similarity metric, and top-k; paste or auto-generate queries;
  optionally inject one hand-written *benign comparison variant* document
  (clearly labeled as metric-testing material, not an attack payload). Runs
  `core/runner.py`'s `BenchmarkRunner` with a progress bar and shows metric
  cards, a baseline-vs-injected bar chart, and raw per-query records.
- **Detection** — calibrates `PerplexityDetector` (gpt2, one-time download)
  on the current library, plots the benign score histogram with the variant
  marked, lets you sweep the FPR target (live FNR/FPR and flagged-document
  table), and draws the FNR-vs-FPR tradeoff line.
- **History** — saved runs (`benchmark_history/`) can be reloaded and
  compared side by side.

The LLM provider is chosen in the sidebar (Anthropic / OpenAI / mock); all
calls go through the same pluggable `selector.py` / `generators.py`
interfaces, so the UI never hardcodes a provider.

## Quick start

```
python examples/run_benchmark.py --task weather --num-queries 10 --k 3
```

The example loads `data/sample_tools.json`, generates (or reuses) user queries
for one task, then runs:

1. **Baseline pass** — no modifications; reports accuracy and hit rate.
2. **Variant pass** — inserts one hand-written, benign-looking alternative
   tool document (not an optimized adversarial one) and reports
   `target_selection_rate` / `target_retrieval_rate`.
3. **Detection pass** — calibrates a `PerplexityDetector` (gpt2) threshold at a
   target FPR on the benign library and reports FNR/FPR for the variant
   document. Skipped gracefully when the model cannot be downloaded.

Use `--offline` to avoid all model downloads (deterministic character/word
n-gram embeddings, no detection pass). Full JSON results are written to
`example_results.json`.

## Plugging in real backends

The harness is deliberately backend-agnostic; every external component is an
injectable callback.

**Embeddings** — `Retriever(embed_fn=...)` accepts any `text -> vector`
callable; the default is sentence-transformers `all-MiniLM-L6-v2` (lazy
loaded):

```python
from tool_selection_harness.core import Retriever

retriever = Retriever()  # MiniLM backend, downloads on first use
# or:
retriever = Retriever(embed_fn=my_custom_embedding_fn)
```

**LLM selection** — `Selector(llm_call=...)` accepts any `prompt -> str`
function:

```python
from openai import OpenAI
from tool_selection_harness.core import Selector

client = OpenAI()

def call_openai(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

selector = Selector(llm_call=call_openai)
```

The example script wires these automatically when `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` is set (used for both query generation via
`core/generators.py` and selection); otherwise it falls back to canned
queries and a greedy prompt-parsing selector so the pipeline stays runnable
offline.

**Detection** — implement the `Detector` protocol to plug in your own scoring
rule:

```python
from tool_selection_harness.core import Detector, ThresholdClassifier

class MyDetector(Detector):
    def score(self, doc: ToolDocument) -> float:
        return suspiciousness_of(doc)  # higher = more suspicious

classifier = ThresholdClassifier(detector=MyDetector())
threshold = classifier.fit_threshold(benign_scores, target_fpr=0.10)
flagged = classifier.classify(candidate_doc)
```

## Tests

```
python -m pytest tool_selection_harness/tests -q
```

All tests use deterministic mock backends (no network, no model downloads).
