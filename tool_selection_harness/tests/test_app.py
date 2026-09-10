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


def test_app_generate_tools_with_mock() -> None:
    """Regression: generating synthetic tools via the mock LLM works."""
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception

    text_areas = {ta.label: ta for ta in at.text_area}
    text_areas[
        "Context queries (one per line; used to shape the generated tools)"
    ].set_value("check the weather in Dhaka\nwill it rain tomorrow?")
    at.run()
    assert not at.exception

    buttons = {b.label: b for b in at.button}
    buttons["Generate tools"].click()
    at.run()
    assert not at.exception

    # 15 seed tools + 3 generated = 18; the count metric reflects it.
    metrics = [m.value for m in at.metric]
    assert "18" in metrics


def test_app_auto_generate_queries() -> None:
    """Regression: auto-generated queries appear in the benchmark text area."""
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception

    text_inputs = {t.label: t for t in at.text_input}
    text_inputs["Target task"].set_value("checking the weather")
    at.run()
    assert not at.exception

    buttons = {b.label: b for b in at.button}
    buttons["Generate queries"].click()
    at.run()
    assert not at.exception

    text_areas = {ta.label: ta for ta in at.text_area}
    assert "checking the weather" in text_areas["One query per line"].value


def test_app_detection_run_without_variant() -> None:
    """Regression: run a detector with no variant document set (offline)."""
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception

    selectboxes = {sb.label: sb for sb in at.selectbox}
    selectboxes["Detector"].set_value("Known-answer (LLM)")
    at.run()
    assert not at.exception

    buttons = {b.label: b for b in at.button}
    buttons["Run detector"].click()
    at.run()
    assert not at.exception

    # Scores render after the run: the histogram and flagged table appear.
    assert any("Calibration FPR target" in s.label for s in at.slider)
