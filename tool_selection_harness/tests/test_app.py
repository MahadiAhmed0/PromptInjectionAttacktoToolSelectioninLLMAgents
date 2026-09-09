"""Smoke tests for the Streamlit frontend (app.py).

These use Streamlit's official AppTest harness to execute the app script
headlessly and assert it renders without exceptions. Skipped automatically
when streamlit is not installed.
"""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "app.py"


def test_app_renders_without_errors() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4


def test_app_shows_library_count() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    metrics = [m.value for m in at.metric]
    assert any(m == "15" for m in metrics)


def test_app_offline_benchmark_run() -> None:
    """Run a benchmark end to end with offline backend + mock LLM."""
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception

    # All tab bodies execute on every run, so widgets are all reachable.
    selectboxes = {sb.label: sb for sb in at.selectbox}
    selectboxes["Embedding backend"].set_value("Offline hashing (no download)")
    selectboxes["Expected tool for all queries (ground truth)"].set_value(
        "get_current_weather"
    )
    at.run()
    assert not at.exception

    text_areas = {ta.label: ta for ta in at.text_area}
    text_areas["One query per line"].set_value(
        "What is the weather in Paris today?\nWill it rain tomorrow?"
    )
    at.run()
    assert not at.exception

    buttons = {b.label: b for b in at.button}
    buttons["Run Benchmark"].click()
    at.run()
    assert not at.exception

    metric_values = [m.value for m in at.metric]
    assert any(m in ("n/a", "0.000", "0.500", "1.000") for m in metric_values)
