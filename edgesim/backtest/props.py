"""Backtest player prop projections on held-out seasons.

Historical sportsbook prop lines are not freely available, so this test
measures what we *can* measure honestly:
  1. accuracy of the projected mean vs a naive season-average baseline
  2. calibration of P(over) at a proxy line (the player's last-10-game
     average rounded to x.5, which is roughly how books open many props)
Real books shade lines with injury/lineup info we don't have, so treat ROI at
proxy lines as an upper bound, not an expected return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from ..props import COMBOS, STATS, PropParams, _GH_W, _GH_X, build_features, prob_over

TRAIN = ["2018-19", "2019-20", "2020-21", "2021-22"]
TEST = ["2022-23", "2023-24"]
EVAL_STATS = ["pts", "reb", "ast", "fg3m", "pra"]


def _mix_loglik(y, rate, pm, sd, opp, size):
    mins = np.clip(pm[:, None] + sd[:, None] * _GH_X, 0.5, 48)
    mu = np.maximum(rate[:, None] * mins * opp[:, None], 1e-6)
    pmf = st.nbinom.pmf(y[:, None], size, size / (size + mu)) @ _GH_W
    return float(np.log(np.maximum(pmf, 1e-12)).sum())


def fit_dispersion(feat: pd.DataFrame, stats=None, sample: int = 40000, seed: int = 0) -> dict:
    stats = stats or STATS + list(COMBOS)
    d = feat.sample(min(sample, len(feat)), random_state=seed)
    grid = np.array([2, 3, 4, 6, 8, 10, 14, 20, 30, 50, 100])
    out = {}
    for s in stats:
        args = (d[s].values, d[f"rate_{s}"].values, d.proj_min.values, d.min_sd.values,
                d[f"opp_{s}"].values)
        ll = [_mix_loglik(*args, g) for g in grid]
        out[s] = float(grid[int(np.argmax(ll))])
    return out


def fit_calibration(train: pd.DataFrame, disp: dict, sample: int = 40000) -> dict:
    """Fit a logit-scale shrink per stat so P(over) is calibrated on training seasons."""
    out = {}
    d = train.sample(min(sample, len(train)), random_state=1)
    for s in EVAL_STATS:
        t = d.dropna(subset=[f"line_{s}"])
        p = np.clip(prob_over(t[f"rate_{s}"].values, t.proj_min.values, t.min_sd.values,
                              t[f"line_{s}"].values, disp[s], t[f"opp_{s}"].values), 1e-6, 1 - 1e-6)
        y = (t[s].values > t[f"line_{s}"].values).astype(float)
        lg = np.log(p / (1 - p))
        best = min(np.arange(0.5, 1.31, 0.05),
                   key=lambda a: -np.mean(y * np.log(1 / (1 + np.exp(-a * lg)))
                                          + (1 - y) * np.log(1 - 1 / (1 + np.exp(-a * lg)))))
        out[s] = round(float(best), 2)
    return out


def add_baselines(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.sort_values(["player_id", "date"]).copy()
    g = f.groupby(["player_id", "season"], sort=False)
    for s in EVAL_STATS:
        f[f"season_avg_{s}"] = g[s].transform(lambda x: x.shift().expanding().mean())
        l10 = f.groupby("player_id", sort=False)[s].transform(lambda x: x.shift().rolling(10).mean())
        f[f"line_{s}"] = np.floor(l10) + 0.5
    return f


def run(games: pd.DataFrame, params: PropParams | None = None, min_prev: int = 15,
        min_proj_minutes: float = 20.0) -> dict:
    params = params or PropParams()
    games = games[games.season.isin(TRAIN + TEST)]
    feat = add_baselines(build_features(games, params))
    elig = feat[(feat.n_prev >= min_prev) & (feat.proj_min >= min_proj_minutes)]
    train, test = elig[elig.season.isin(TRAIN)], elig[elig.season.isin(TEST)]
    disp = fit_dispersion(train)
    params.dispersion.update(disp)
    params.calib.update(fit_calibration(train, disp))

    acc, cal, bets = [], [], []
    for s in EVAL_STATS:
        t = test.dropna(subset=[f"season_avg_{s}", f"line_{s}"])
        proj = t[f"rate_{s}"] * t.proj_min * t[f"opp_{s}"]
        acc.append(dict(stat=s, rows=len(t), model_mae=float((t[s] - proj).abs().mean()),
                        season_avg_mae=float((t[s] - t[f"season_avg_{s}"]).abs().mean()),
                        model_bias=float((t[s] - proj).mean())))
        p = params.calibrate(s, prob_over(t[f"rate_{s}"].values, t.proj_min.values,
                                          t.min_sd.values, t[f"line_{s}"].values, disp[s],
                                          t[f"opp_{s}"].values))
        y = (t[s].values > t[f"line_{s}"].values).astype(float)
        for lo in np.arange(0.2, 0.8, 0.1):
            m = (p >= lo) & (p < lo + 0.1)
            if m.sum() >= 50:
                cal.append(dict(stat=s, bucket=f"{lo:.1f}-{lo+0.1:.1f}", n=int(m.sum()),
                                predicted=float(p[m].mean()), actual=float(y[m].mean())))
        brier = float(((p - y) ** 2).mean())
        for thr in (0.55, 0.58, 0.62):
            pick_over, pick_under = p >= thr, p <= 1 - thr
            wins = y[pick_over].sum() + (1 - y[pick_under]).sum()
            n = pick_over.sum() + pick_under.sum()
            bets.append(dict(stat=s, threshold=thr, bets=int(n),
                             hit_rate=float(wins / n) if n else np.nan,
                             roi_at_minus115=float((wins * 100 / 115 - (n - wins)) / n) if n else np.nan,
                             brier=brier, brier_coinflip=0.25))
    return dict(dispersion=disp, calib=dict(params.calib), accuracy=pd.DataFrame(acc), calibration=pd.DataFrame(cal),
                proxy_bets=pd.DataFrame(bets))
