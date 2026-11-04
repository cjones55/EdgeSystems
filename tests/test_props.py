import numpy as np
import pandas as pd
import pytest

from edgesim.props import PropParams, build_features, prob_over, prob_push, simulate_player


def _games(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        d = pd.Timestamp("2023-01-01") + pd.Timedelta(days=2 * i)
        for pid, team, opp, mins, ppm in (("1", "BOS", "MIA", 34, 0.8), ("2", "MIA", "BOS", 30, 0.5)):
            m = mins + rng.normal(0, 3)
            rows.append(dict(date=d, game_id=f"g{i}", player_id=pid, player=f"P{pid}", team=team,
                             opp=opp, min=m, pts=rng.poisson(ppm * m), reb=rng.poisson(0.2 * m),
                             ast=rng.poisson(0.1 * m), fg3m=rng.poisson(0.05 * m),
                             stl=rng.poisson(0.03 * m), blk=rng.poisson(0.02 * m),
                             tov=rng.poisson(0.05 * m)))
    return pd.DataFrame(rows)


def test_features_have_no_lookahead():
    g = _games()
    f1 = build_features(g)
    g2 = g.copy()
    g2.loc[g2.date == g2.date.max(), "pts"] = 500       # change only the final game
    f2 = build_features(g2)
    last = f1.date == f1.date.max()
    # the final game's own projection must not see its own result
    assert np.allclose(f1.loc[last, "rate_pts"], f2.loc[last, "rate_pts"])


def test_features_learn_player_rates():
    f = build_features(_games(80))
    late = f[f.date == f.date.max()].set_index("player_id")
    assert late.loc["1", "rate_pts"] > late.loc["2", "rate_pts"]
    assert late.loc["1", "proj_min"] == pytest.approx(34, abs=3)


def test_prob_over_monotone_and_matches_sim():
    lines = np.array([10.5, 15.5, 20.5, 25.5, 30.5])
    p = prob_over(np.full(5, 0.6), np.full(5, 33.0), np.full(5, 4.0), lines, 12.0)
    assert (np.diff(p) < 0).all()
    sim = simulate_player({"pts": 0.6, "reb": .2, "ast": .1, "fg3m": .05, "stl": .03, "blk": .02,
                           "tov": .05}, 33.0, 4.0, n=60000, params=PropParams(), seed=1)
    for ln, pr in zip(lines, p):
        assert (sim.pts > ln).mean() == pytest.approx(pr, abs=0.03)


def test_push_only_on_whole_lines():
    assert prob_push(np.array([0.5]), np.array([30.]), np.array([4.]), np.array([15.5]), 10)[0] == 0
    assert prob_push(np.array([0.5]), np.array([30.]), np.array([4.]), np.array([15.0]), 10)[0] > 0


def test_combo_stats_correlated():
    sim = simulate_player({"pts": 0.6, "reb": .25, "ast": .15, "fg3m": .05, "stl": .03, "blk": .02,
                           "tov": .05}, 30.0, 6.0, n=20000, seed=3)
    assert np.corrcoef(sim.pts, sim.reb)[0, 1] > 0.1
    assert (sim.pra == sim.pts + sim.reb + sim.ast).all()


def test_calibration_shrinks():
    p = PropParams(calib={"pts": 0.8})
    assert p.calibrate("pts", 0.7) < 0.7
    assert p.calibrate("pts", 0.5) == pytest.approx(0.5)
    assert p.calibrate("reb", 0.7) == pytest.approx(0.7)
