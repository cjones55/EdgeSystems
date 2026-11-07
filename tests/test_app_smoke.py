"""Render the Streamlit app end to end on an archived slate."""
import dataclasses

import pytest

from edgesim import pipeline


def _run_app(monkeypatch, tmp_path, strip_lines: bool):
    from streamlit.testing.v1 import AppTest
    real = pipeline.demo_context

    def ctx(day, sport="nba"):
        c = real(day, sport=sport)
        if strip_lines:     # late at night ESPN drops the lines of games that started
            c.games = [dataclasses.replace(g, spread=None, total=None, home_ml=None, away_ml=None,
                                           open_spread=None, open_total=None) for g in c.games]
        return c
    monkeypatch.setattr(pipeline, "demo_context", ctx)
    monkeypatch.setattr(pipeline, "write_report", lambda res: None)
    at = AppTest.from_file("../app.py", default_timeout=120)
    at.run()
    radios = [r for r in at.radio if "Demo (archive)" in r.options]
    assert radios, "data mode toggle not found"
    if radios:
        radios[0].set_value("Demo (archive)").run()
    sel = [s for s in at.selectbox if any("College football" in o for o in s.options)]
    assert sel, "demo sport selector not found"
    sel[0].set_value("ncaaf").run()
    [b for b in at.button if b.label == "Run sims"][0].click().run()
    return at


@pytest.mark.parametrize("strip", [False, True])
def test_app_renders_cfb(monkeypatch, tmp_path, strip):
    from edgesim.cli import DEMO_DATES
    try:
        pipeline.demo_context(DEMO_DATES["ncaaf"], sport="ncaaf")
    except FileNotFoundError:
        pytest.skip("historical data not downloaded")
    at = _run_app(monkeypatch, tmp_path, strip)
    assert not at.exception, [e.value for e in at.exception]
