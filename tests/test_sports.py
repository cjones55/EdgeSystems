"""Multi-sport: simulators, data loaders, team resolution, card and backtest plumbing."""
import numpy as np
import pandas as pd
import pytest

from edgesim import gamesim_nfl, gamesim_nhl, sports
from edgesim.data.historical import ROOT
from edgesim.plays import GameMarket, PlayConfig, daily_card
from edgesim.ratings import RatingParams, TeamRatings
from edgesim.resolve import resolve
from edgesim.teams import to_code

RAW = ROOT / "data" / "raw"


# ------------------------------------------------------------------ simulators
def test_nfl_sim_mean_and_key_numbers():
    r = gamesim_nfl.simulate_game(24, 21, gamesim_nfl.NFLSimConfig(n_sims=40000, seed=1))
    assert r.home.mean() == pytest.approx(24, abs=0.6)
    assert r.away.mean() == pytest.approx(21, abs=0.6)
    m = np.abs(r.margin)
    # 3 and 7 are the most common margins, 5 is rare: football's key numbers
    assert (m == 3).mean() > (m == 5).mean() * 1.8
    assert (m == 7).mean() > (m == 5).mean() * 1.5
    assert 0 < (r.margin == 0).mean() < 0.01        # NFL ties are possible but rare


def test_college_football_has_no_ties():
    cfg = sports.get("ncaaf").sim_config(n_sims=20000, seed=2)
    r = gamesim_nfl.simulate_game(28, 28, cfg)
    assert (r.margin == 0).sum() == 0


def test_nhl_sim_shape():
    r = gamesim_nhl.simulate_game(3.1, 2.8, gamesim_nhl.NHLSimConfig(n_sims=40000, seed=3))
    assert r.home.mean() + r.away.mean() == pytest.approx(5.9, abs=0.35)
    assert (r.margin != 0).all()                      # OT / shootout always decides
    assert 0.15 < r.went_ot.mean() < 0.28             # ~22% of NHL games go past regulation
    one_goal = (np.abs(r.margin) == 1).mean()
    assert 0.38 < one_goal < 0.55                     # actual NHL: ~45%
    pl, _ = r.p_cover(-1.5)
    assert 0.25 < pl < 0.45                           # favourites rarely cover -1.5


def test_ncaab_sim_uses_40_minutes():
    sp = sports.get("ncaab")
    r = sp.sim(72, 68, n_sims=20000, seed=4)
    assert r.home.mean() == pytest.approx(72, abs=1)
    assert 9 < r.margin.std() < 14


@pytest.mark.parametrize("key", list(sports.SPORTS))
def test_every_sport_sim_runs(key):
    sp = sports.get(key)
    r = TeamRatings(sp.rating)
    eh, ea = r.expected("A", "B")
    res = sp.sim(eh, ea, n_sims=2000, seed=1)
    assert 0.4 < res.p_home_win() < 0.75               # home edge only
    assert res.total.mean() == pytest.approx(sp.rating.league_avg, rel=0.08)


def test_neutral_site_removes_home_edge():
    r = TeamRatings(RatingParams(hca=4.0))
    eh, ea = r.expected("A", "B", neutral=True)
    assert eh == pytest.approx(ea)


def test_new_team_prior_and_error_cap():
    r = TeamRatings(RatingParams(new_team_off=-5, new_team_def=-5, err_cap=10, recentre=False))
    r.off["FBS"], r.deff["FBS"] = 0.0, 0.0            # an established, average team
    eh, ea = r.expected("FBS", "FCS")                  # vs a team never seen before
    assert eh - ea == pytest.approx(10 + r.hca, abs=1e-9)
    before = r.off.get("FBS", 0)
    r.update("FBS", "FCS", 90, 0)                     # blowout counts as at most +10
    assert r.off["FBS"] - before <= r.params.k_early * 10 + 1e-9


# ------------------------------------------------------------------ names
def test_team_resolution():
    assert to_code("St.Louis", "nfl") == "LAR" and to_code("St.Louis", "nhl") == "STL"
    assert to_code("Montréal Canadiens", "nhl") == "MTL"
    assert to_code("Arizona Coyotes", "nhl") == "UTA"
    assert to_code("Washington Commanders", "nfl") == "WSH"


@pytest.mark.skipif(not (RAW / "ncaaf").exists(), reason="no college football data")
def test_college_football_names():
    assert resolve("Ohio State Buckeyes", "ncaaf") == "Ohio State"
    assert resolve("Miami (FL) Hurricanes", "ncaaf") == "Miami"
    assert resolve("North Carolina State Wolfpack", "ncaaf") == "NC State"
    assert resolve("Some Brand New School", "ncaaf") == "Some Brand New School"


@pytest.mark.skipif(not (RAW / "ncaab").exists(), reason="no college basketball data")
def test_college_basketball_names():
    a, b = resolve("Kansas Jayhawks", "ncaab"), resolve("Kansas", "ncaab")
    assert a == b and a.isdigit()
    assert resolve("Saint Mary's Gaels", "ncaab") == resolve("St Mary's CA", "ncaab")


# ------------------------------------------------------------------ data
@pytest.mark.parametrize("key,min_rows", [("nfl", 2500), ("nhl", 13000), ("ncaaf", 9000),
                                          ("ncaab", 50000)])
def test_history_loads(key, min_rows):
    path = {"nfl": RAW / "nfl_archive_10Y.json", "nhl": RAW / "nhl_archive_10Y.json",
            "ncaaf": RAW / "ncaaf", "ncaab": RAW / "ncaab"}[key]
    if not path.exists():
        pytest.skip("no data")
    h = sports.load_history(key)
    assert len(h) >= min_rows
    need = {"date", "season", "home", "away", "home_pts", "away_pts", "close_spread",
            "close_total", "home_ml", "away_ml", "neutral", "over_price", "under_price"}
    assert need <= set(h.columns)
    lined = h.dropna(subset=["close_spread"])
    if key != "nhl":
        resid = lined.home_pts - lined.away_pts + lined.close_spread
        assert abs(resid.mean()) < 1.0               # spreads are signed from the home side
    if key == "nhl":
        t = h.dropna(subset=["close_total"])
        t = t[t.close_total % 1 == 0.5]
        from edgesim.odds import implied_prob
        heavy_over = t[t.over_price.map(implied_prob) > 0.56]
        heavy_under = t[t.under_price.map(implied_prob) > 0.56]
        # the side priced as the favourite should hit more often
        assert (heavy_over.home_pts + heavy_over.away_pts > heavy_over.close_total).mean() > \
               (heavy_under.home_pts + heavy_under.away_pts > heavy_under.close_total).mean()


# ------------------------------------------------------------------ card
def test_nhl_card_prices_puck_line_and_ties_push_in_nfl():
    r = TeamRatings(sports.get("nhl").rating)
    g = GameMarket("BOS", "TOR", spread=-1.5, spread_home_price=185, spread_away_price=-225,
                   total=6.0, over_price=-105, under_price=-115, home_ml=-130, away_ml=110)
    plays, cands, sims = daily_card([g], r, cfg=PlayConfig.for_sport("nhl", n_sims=4000))
    assert len(cands) == 6 and set(cands.market) == {"spread", "total", "ml"}
    r = TeamRatings(sports.get("nfl").rating)
    g = GameMarket("KC", "BUF", spread=-2.5, total=47.5, home_ml=-140, away_ml=120)
    _, cands, _ = daily_card([g], r, cfg=PlayConfig.for_sport("nfl", n_sims=4000))
    ml = cands[cands.market == "ml"]
    assert ml.model_prob.sum() == pytest.approx(1.0, abs=1e-6)   # decided-game probabilities


@pytest.mark.parametrize("key", ["nfl", "nhl", "ncaaf", "ncaab"])
def test_walk_forward_each_sport(key):
    path = {"nfl": RAW / "nfl_archive_10Y.json", "nhl": RAW / "nhl_archive_10Y.json",
            "ncaaf": RAW / "ncaaf", "ncaab": RAW / "ncaab"}[key]
    if not path.exists():
        pytest.skip("no data")
    from edgesim.backtest.games import moneyline_scores, walk_forward
    h = sports.load_history(key)
    lined = h[h[["close_spread", "close_total"]].notna().all(axis=1)]
    season = lined.season.iloc[len(lined) // 2]
    g = h[h.season == season]
    pred = walk_forward(g, key, n_sims=500)
    assert len(pred) > 50
    assert pred.p_home_win.between(0, 1).all()
    s = moneyline_scores(pred)
    assert s["model_logloss"] < 0.69                   # better than a coin flip


@pytest.mark.skipif(not (RAW / "ncaab").exists(), reason="no data")
def test_demo_context_college():
    from edgesim.pipeline import demo_context, run_day
    res = run_day(demo_context("2022-02-12", sport="ncaab"))
    assert len(res.sims) > 100
    assert not res.sims.game.str.contains(r"^\d+ @", regex=True).any()   # names, not ids
