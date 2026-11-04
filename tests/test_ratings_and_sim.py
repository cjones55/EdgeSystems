import numpy as np
import pytest

from edgesim.gamesim import SimConfig, possession_pmf, simulate_game
from edgesim.ratings import RatingParams, TeamRatings


def test_ratings_learn_direction():
    r = TeamRatings()
    for _ in range(30):
        r.update("BOS", "DET", 125, 100)
        r.update("DET", "BOS", 100, 120)
    eh, ea = r.expected("BOS", "DET")
    assert eh - ea > 10
    assert r.table()[0]["team"] == "BOS"


def test_ratings_home_court_and_rest():
    r = TeamRatings(RatingParams(hca=3.0, b2b=2.0))
    eh, ea = r.expected("A", "B")
    assert eh - ea == pytest.approx(3.0)
    eh2, ea2 = r.expected("A", "B", home_rest=1, away_rest=2)
    assert (eh2 - ea2) == pytest.approx(3.0 - 2.0)


def test_new_season_regresses():
    r = TeamRatings(RatingParams(season_regress=0.5))
    r.off, r.deff = {"A": 4.0, "B": -4.0}, {"A": 2.0, "B": -2.0}
    r.update("A", "B", 110, 100, season=2020)
    assert abs(r.off["A"]) < 4.0


def test_possession_pmf_mean():
    for ppp in (0.95, 1.12, 1.25):
        pmf = possession_pmf(ppp)
        assert pmf.sum() == pytest.approx(1)
        assert pmf @ np.arange(4) == pytest.approx(ppp)


def test_sim_matches_targets():
    res = simulate_game(115, 108, SimConfig(n_sims=40000, seed=1))
    s = res.summary()
    assert s["home_pts"] == pytest.approx(115, abs=0.5)
    assert s["away_pts"] == pytest.approx(108, abs=0.5)
    # outcome spread = calibrated game noise + model uncertainty
    cfg = SimConfig()
    assert s["margin_sd"] == pytest.approx(np.hypot(cfg.margin_sd, cfg.rating_sd), rel=0.05)
    assert s["total_sd"] == pytest.approx(np.hypot(cfg.total_sd, cfg.rating_total_sd), rel=0.08)
    assert (res.home != res.away).all()           # overtime breaks every tie
    assert 0.01 < s["ot_rate"] < 0.08


def test_sim_probabilities_consistent():
    res = simulate_game(110, 110, SimConfig(n_sims=40000, seed=2))
    assert res.p_home_win() == pytest.approx(0.5, abs=0.02)
    c, p = res.p_cover(-3)
    c2, p2 = res.p_cover(3)
    assert c < 0.5 < c2 and p > 0
    o, _ = res.p_over(220.5)
    assert o == pytest.approx(0.5, abs=0.03)


def test_sim_seeded_is_reproducible():
    a = simulate_game(112, 109, SimConfig(n_sims=2000, seed=5))
    b = simulate_game(112, 109, SimConfig(n_sims=2000, seed=5))
    assert (a.home == b.home).all()
