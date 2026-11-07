import pandas as pd

from edgesim.plays import UNIT_SIZES, units
from edgesim.providers.odds_api import consensus
from edgesim.watch import Alert, WatchState, find_alerts

STEAM = {"spread": 1.5, "total": 2.5, "ml": 0.05}


def test_units_only_use_the_scale():
    for tier in ("A: model + sharp", "B: model edge", "C: sharp lean"):
        for ev in (-0.04, 0.0, 0.01, 0.03, 0.06, 0.2):
            for k in (0, 0.001, 0.007, 0.01, 0.02, 0.029, 0.03):
                assert units(tier, ev, k) in UNIT_SIZES
    assert units(None, 0.1, 0.03) is None


def test_unit_tiers():
    assert units("C: sharp lean", -0.02, 0.0) == 1
    assert units("B: model edge", 0.02, 0.005) == 1
    assert units("B: model edge", 0.03, 0.01) == 2
    assert units("B: model edge", 0.04, 0.02) == 3
    assert units("B: model edge", 0.08, 0.03) == 5          # max without sharp agreement
    assert units("A: model + sharp", 0.04, 0.03) == 5       # EV too small for super max
    assert units("A: model + sharp", 0.06, 0.03) == 10      # super max


def _cand(**kw):
    base = dict(game="B @ A", market="spread", side="home", pick="A -3", price=-110,
                model_prob=0.52, market_prob=0.50, ev=-0.01, tier=None, units=None,
                move=0.0, sharp_side="-", sharp_score=0.0, sharp_reasons="")
    base.update(kw)
    return base


def test_find_alerts_kinds():
    c = pd.DataFrame([
        _cand(tier="B: model edge", units=2, ev=0.03),                       # new play
        _cand(side="away", pick="B +3", model_prob=0.62, market_prob=0.50),   # early lean
        _cand(market="total", side="over", pick="Over 220", move=3.0),         # steam
        _cand(market="total", side="under", pick="Under 220", move=-3.0,
              sharp_side="this side", sharp_score=55, sharp_reasons="Reverse line move"),
    ])
    alerts, early = find_alerts(c, "nba", STEAM)
    kinds = sorted(a.kind for a in alerts)
    assert kinds == ["EARLY", "NEW PLAY", "SHARP", "STEAM"]
    assert len(early) == 1 and early.iloc[0]["tier"] == "W: early lean"
    assert len({a.key for a in alerts}) == len(alerts)


def test_early_needs_an_unmoved_line():
    c = pd.DataFrame([_cand(model_prob=0.62, market_prob=0.50, move=1.0)])
    alerts, early = find_alerts(c, "nba", STEAM)
    assert not any(a.kind == "EARLY" for a in alerts) and early.empty


def test_watch_state_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    WatchState({"a", "b"}).save(p)
    assert WatchState.load(p).seen == {"a", "b"}
    assert Alert("STEAM", "nfl", "g", "pick", -110, "d").line().startswith("STEAM")


def test_consensus_prefers_my_books():
    rows = []
    for book, pt, price in (("pinnacle", -3.0, -105), ("betmgm", -3.0, -108),
                            ("fanduel", -2.5, -115), ("draftkings", -2.5, -112)):
        rows.append(dict(event_id="e", home="A", away="B", market="spread", side="home",
                         point=pt, price=price, book=book))
    odds = pd.DataFrame(rows)
    allb = consensus(odds).iloc[0]
    assert allb.best_book == "pinnacle"
    mine = consensus(odds, ["fanduel", "draftkings"]).iloc[0]
    assert mine.point == -2.5 and mine.best_book == "draftkings" and mine.best_price == -112
    # a market none of my books list falls back to everyone
    only_other = consensus(odds[odds.book.isin(["pinnacle"])], ["fanduel"]).iloc[0]
    assert only_other.best_book == "pinnacle"


def test_scan_end_to_end_offline(tmp_path, monkeypatch):
    """A full scan on an archived NFL date, with live data swapped for the archive."""
    import pytest
    from datetime import date
    from edgesim import pipeline, tracker, watch
    try:
        ctx = pipeline.demo_context("2021-11-14", sport="nfl")
    except FileNotFoundError:
        pytest.skip("historical data not downloaded")
    ctx.source = "live"
    monkeypatch.setattr(pipeline, "live_context", lambda *a, **k: ctx)
    monkeypatch.setattr(pipeline, "REPORTS", tmp_path)
    monkeypatch.setattr(watch, "REPORTS", tmp_path)
    monkeypatch.setattr(watch, "STATE_DIR", tmp_path / "state")
    log = tmp_path / "log.csv"
    monkeypatch.setattr(tracker, "LOG", log)
    monkeypatch.setattr(tracker.log_plays, "__defaults__", (log, "nba"))
    day = date(2021, 11, 14)
    first, _ = watch.scan(day, "nfl")
    again, _ = watch.scan(day, "nfl")
    assert first and not again                 # alerts fire once
    assert (tmp_path / "alerts_2021-11-14.md").exists()
    logged = pd.read_csv(log)
    assert set(logged.stake_u.dropna()) <= set(UNIT_SIZES)


def test_units_ignore_missing_tier_values():
    # pandas 3 stores a mixed None/str column as strings with NaN for the gaps
    assert units(float("nan"), 0.05, 0.02) is None
