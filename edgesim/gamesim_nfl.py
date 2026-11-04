"""Monte Carlo NFL game simulator (drive based).

Each team gets ~11-12 possessions. A drive ends in a touchdown (usually 7,
sometimes 6 or 8), a field goal (3) or nothing. Because scores are built
from 3s and 7s, the simulated margins land on the real key numbers (3, 7,
10, 14) at realistic rates, which matters a lot when pricing NFL spreads.
Ties after regulation go to a simplified overtime that can still end tied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gamesim import SimResult

TD_SHARE = 0.58          # share of scoring drives that are touchdowns
TD_VALUES = np.array([6, 7, 8])
TD_PROBS = np.array([0.04, 0.91, 0.05])   # missed PAT / PAT / two-point
PTS_PER_SCORE = TD_SHARE * float(TD_VALUES @ TD_PROBS) + (1 - TD_SHARE) * 3


@dataclass
class NFLSimConfig:
    n_sims: int = 20000
    drives_mean: float = 11.3
    drives_sd: float = 1.0
    rating_sd: float = 2.0        # model uncertainty on the margin (points)
    rating_total_sd: float = 3.0
    ot_tie_prob: float = 0.08     # NFL regular season games can end tied; college can't
    ot_fg_share: float = 0.55     # share of overtime winners decided by a field goal
    seed: int | None = None


def _team_points(rng, drives: np.ndarray, ppd: np.ndarray) -> np.ndarray:
    p_score = np.clip(ppd / PTS_PER_SCORE, 0.02, 0.85)
    scores = rng.binomial(drives, p_score)
    tds = rng.binomial(scores, TD_SHARE)
    fgs = scores - tds
    # touchdown values: count of 6s and 8s among the TDs
    n6 = rng.binomial(tds, TD_PROBS[0])
    n8 = rng.binomial(tds - n6, TD_PROBS[2] / (1 - TD_PROBS[0]))
    return 7 * tds - n6 + n8 + 3 * fgs


def simulate_game(exp_home: float, exp_away: float, cfg: NFLSimConfig | None = None) -> SimResult:
    cfg = cfg or NFLSimConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_sims
    margin = exp_home - exp_away + rng.normal(0, cfg.rating_sd, n)
    total = exp_home + exp_away + rng.normal(0, cfg.rating_total_sd, n)
    th, ta = np.maximum((total + margin) / 2, 1), np.maximum((total - margin) / 2, 1)
    drives = np.maximum(7, np.round(rng.normal(cfg.drives_mean, cfg.drives_sd, n))).astype(int)
    extra = rng.integers(-1, 2, n)          # one team can get an extra possession
    dh, da = drives + np.maximum(extra, 0), drives + np.maximum(-extra, 0)
    # points per drive such that expected points match the target on average
    home = _team_points(rng, dh, th / (drives + 1 / 3))
    away = _team_points(rng, da, ta / (drives + 1 / 3))
    # overtime
    tied = home == away
    went_ot = tied.copy()
    k = int(tied.sum())
    if k:
        p_home = 1 / (1 + np.exp(-(th[tied] - ta[tied]) / 8))
        decided = rng.random(k) >= cfg.ot_tie_prob
        home_wins = rng.random(k) < p_home
        pts = np.where(rng.random(k) < cfg.ot_fg_share, 3, 7)
        h_add = np.where(decided & home_wins, pts, 0)
        a_add = np.where(decided & ~home_wins, pts, 0)
        home[tied] += h_add
        away[tied] += a_add
    return SimResult(home=home, away=away, went_ot=went_ot)
