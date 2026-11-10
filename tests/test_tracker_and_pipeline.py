import pandas as pd
import pytest

from edgesim import tracker
from edgesim.data.historical import ROOT


def test_settle():
    assert tracker.settle("spread", "home", -4.5, 110, 100) == 1
    assert tracker.settle("spread", "away", 4.5, 110, 100) == -1
    assert tracker.settle("spread", "home", -10, 110, 100) == 0
    assert tracker.settle("total", "over", 209.5, 110, 100) == 1
    assert tracker.settle("total", "under", 209.5, 110, 100) == -1
    assert tracker.settle("ml", "away", None, 99, 100) == 1
    assert tracker.payout(1, -110) == pytest.approx(0.909, abs=1e-3)
    assert tracker.payout(1, 150) == 1.5


def test_log_and_grade(tmp_path, monkeypatch):
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    monkeypatch.setattr(tracker, "SNAPS", snaps)
    pd.DataFrame([
        dict(home="BOS", away="MIA", market="spread", side="home", point=-4.5, price=-110, pulled_at="1"),
        dict(home="BOS", away="MIA", market="spread", side="away", point=4.5, price=-110, pulled_at="1"),
        dict(home="BOS", away="MIA", market="spread", side="home", point=-6.0, price=-110, pulled_at="2"),
        dict(home="BOS", away="MIA", market="spread", side="away", point=6.0, price=-110, pulled_at="2"),
    ]).to_csv(snaps / "nba_2026-11-04.csv", index=False)
    plays = pd.DataFrame([dict(game="MIA @ BOS", home="BOS", away="MIA", market="spread",
                               side="home", line=-4.5, pick="BOS -4.5", price=-110, book="dk",
                               tier="A: model + sharp", model_prob=.6, blended_prob=.55,
                               market_prob=.5, ev=.05, kelly=.01, sharp_score=50)])
    log = tmp_path / "log.csv"
    assert tracker.log_plays(plays, "2026-11-04", log) == 1
    tracker.log_plays(plays, "2026-11-04", log)       # duplicate is ignored
    graded = tracker.grade(log, scores={("2026-11-04", "BOS", "MIA"): (112, 100)})
    r = graded.iloc[0]
    assert len(graded) == 1
    assert r.result == 1 and r.units == pytest.approx(0.909, abs=1e-3)
    assert r.close_line == -6.0 and r.clv_pts == 1.5     # we got -4.5, it closed -6
    s = tracker.summary(graded)
    assert s.iloc[0].bets == 1


@pytest.mark.skipif(not (ROOT / "data/raw/nba_archive_10Y.json").exists(), reason="no data")
def test_demo_day_end_to_end():
    from edgesim.pipeline import demo_context, run_day
    ctx = demo_context("2022-01-17", with_props=False)
    res = run_day(ctx)
    assert len(res.sims) == 12
    assert len(res.candidates) == 12 * 6
    assert res.sims.home_win.between(0, 1).all()
    # walk-forward: ratings must not include the demo day's own games
    assert all(ctx.ratings.games.get(t, 0) < 82 for t in ctx.ratings.games)


@pytest.mark.skipif(not (ROOT / "data/raw/nba_archive_10Y.json").exists(), reason="no data")
def test_backtest_walk_forward_has_no_lookahead():
    from edgesim.backtest.games import walk_forward
    from edgesim.data.historical import load_games
    from edgesim.gamesim import SimConfig
    g = load_games()
    g = g[g.season == 2016].head(300)
    a = walk_forward(g, sim=SimConfig(n_sims=500))
    g2 = g.copy()
    g2.loc[g2.index[-1], "home_pts"] += 80            # tamper with the final game only
    b = walk_forward(g2, sim=SimConfig(n_sims=500))
    assert a.exp_home.iloc[-1] == pytest.approx(b.exp_home.iloc[-1])


def test_data_notes_name_the_season_and_new_teams():
    from datetime import date
    import pandas as pd
    from edgesim.pipeline import data_notes
    r = pd.DataFrame(dict(date=pd.to_datetime(["2026-03-10", "2026-11-08"]), season=[2026, 2027],
                          home=["1101", "1102"], away=["1103", "1104"]))
    notes = data_notes(r, [dict(home="1101", away="Zeta U")], date(2026, 11, 10), "ncaab")
    assert "1 games so far this season (2026-27)" in notes[0]
    assert "Zeta U" in notes[1]
    assert "(2026)" in data_notes(r.assign(season=[2025, 2026]), [], date(2026, 11, 10), "ncaaf")[0]
