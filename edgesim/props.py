"""Player prop projections.

For each player we project, using only games before the one being predicted:
  * minutes: exponentially weighted mean and spread of recent minutes
  * per-minute production for each stat, shrunk toward the league rate
    (players with little history get pulled to average)
  * opponent factor: how much of that stat the opponent has allowed lately
    vs league average
The stat's distribution is a negative binomial whose mean is
rate x minutes x opponent x pace, mixed over the minutes uncertainty.
Dispersion per stat is fitted on historical seasons (see backtest/props.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as st

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
COMBOS = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"],
          "ra": ["reb", "ast"]}


@dataclass
class PropParams:
    min_halflife: float = 6.0
    rate_halflife: float = 20.0
    prior_minutes: float = 150.0      # shrinkage weight (minutes) toward league rate
    opp_halflife: float = 15.0
    opp_weight: float = 0.5           # how much of the opponent factor to apply
    # negative binomial "size"; larger = closer to Poisson. Fitted by backtest.
    dispersion: dict = field(default_factory=lambda: {
        "pts": 12.0, "reb": 14.0, "ast": 9.0, "fg3m": 8.0, "stl": 20.0, "blk": 6.0,
        "tov": 20.0, "pra": 18.0, "pr": 16.0, "pa": 14.0, "ra": 14.0})
    # probability shrink toward 50% (logit scale), fitted by backtest; 1.0 = none
    calib: dict = field(default_factory=dict)

    def calibrate(self, stat: str, p):
        a = self.calib.get(stat, 1.0)
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-a * np.log(p / (1 - p))))


def _ewm_prev(g: pd.Series, hl: float) -> pd.Series:
    return g.shift().ewm(halflife=hl, min_periods=1).mean()


def build_features(games: pd.DataFrame, params: PropParams | None = None) -> pd.DataFrame:
    """Walk-forward player projections for every played game in ``games``.

    ``games`` needs: date, game_id, player_id, team, opp, min and the STATS
    columns (only rows where the player played). Rows are returned with
    proj_min, min_sd, rate_<stat>, opp_<stat>, n_prev columns added.
    """
    p = params or PropParams()
    df = games[games["min"] > 0].sort_values(["player_id", "date"]).copy()
    for c in COMBOS:
        df[c] = df[COMBOS[c]].sum(axis=1)
    allstats = STATS + list(COMBOS)
    g = df.groupby("player_id", sort=False)
    df["n_prev"] = g.cumcount()
    df["proj_min"] = g["min"].transform(lambda s: _ewm_prev(s, p.min_halflife))
    sq = g["min"].transform(lambda s: _ewm_prev(s ** 2, p.min_halflife))
    df["min_sd"] = np.sqrt(np.maximum(sq - df["proj_min"] ** 2, 0)).clip(lower=2.5)

    # league per-minute rates, expanding by date (no look-ahead)
    daily = df.groupby("date")[allstats + ["min"]].sum().cumsum().shift()
    league_rate = daily[allstats].div(daily["min"], axis=0).bfill()
    lr = league_rate.reindex(df["date"]).to_numpy()

    ew_min = g["min"].transform(lambda s: _ewm_prev(s, p.rate_halflife))
    cnt = df["n_prev"].clip(upper=3 * p.rate_halflife)  # effective sample size proxy
    eff_min = ew_min * cnt
    for i, s in enumerate(allstats):
        ew_s = g[s].transform(lambda x: _ewm_prev(x, p.rate_halflife))
        raw = (ew_s / ew_min).fillna(0)
        df[f"rate_{s}"] = (raw * eff_min + lr[:, i] * p.prior_minutes) / (eff_min.fillna(0) + p.prior_minutes)

    # opponent factor: stat allowed per minute by opponent vs league rate
    team_games = df.groupby(["date", "game_id", "opp"])[allstats + ["min"]].sum().reset_index()
    team_games = team_games.sort_values(["opp", "date"])
    tg = team_games.groupby("opp", sort=False)
    lr_t = league_rate.reindex(team_games["date"]).to_numpy()
    for i, s in enumerate(allstats):
        allowed = tg[s].transform(lambda x: _ewm_prev(x, p.opp_halflife))
        mins = tg["min"].transform(lambda x: _ewm_prev(x, p.opp_halflife))
        f = (allowed / mins) / lr_t[:, i]
        team_games[f"opp_{s}"] = 1 + p.opp_weight * (f.fillna(1).clip(0.7, 1.3) - 1)
    df = df.merge(team_games[["game_id", "opp"] + [f"opp_{s}" for s in allstats]],
                  on=["game_id", "opp"], how="left")
    return df


def projected_mean(row, stat: str, pace: float = 1.0) -> float:
    return float(row[f"rate_{stat}"] * row["proj_min"] * row[f"opp_{stat}"] * pace)


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
_GH_W = _GH_W / _GH_W.sum()


def prob_over(rate, proj_min, min_sd, line, size, factor=1.0):
    """P(stat > line), mixing a negative binomial over Normal minutes.

    Vectorised: every argument may be an array. Returns P(over), and since
    lines are usually x.5, P(under) = 1 - P(over) (P(push) for whole lines
    is returned separately by ``prob_push``).
    """
    rate, proj_min, min_sd, line, factor = map(np.asarray, (rate, proj_min, min_sd, line, factor))
    mins = np.clip(proj_min[..., None] + min_sd[..., None] * _GH_X, 0.5, 48)
    mu = np.maximum(rate[..., None] * mins * factor[..., None], 1e-6)
    pr = size / (size + mu)
    p_le = st.nbinom.cdf(np.floor(line)[..., None], size, pr)
    return (1 - p_le) @ _GH_W


def prob_push(rate, proj_min, min_sd, line, size, factor=1.0):
    rate, proj_min, min_sd, line, factor = map(np.asarray, (rate, proj_min, min_sd, line, factor))
    mins = np.clip(proj_min[..., None] + min_sd[..., None] * _GH_X, 0.5, 48)
    mu = np.maximum(rate[..., None] * mins * factor[..., None], 1e-6)
    pr = size / (size + mu)
    whole = (np.asarray(line) == np.floor(line))[..., None]
    return (st.nbinom.pmf(np.asarray(line)[..., None], size, pr) * whole) @ _GH_W


def simulate_player(rate: dict, proj_min: float, min_sd: float, n: int = 10000,
                    params: PropParams | None = None, pace: float = 1.0,
                    opp: dict | None = None, seed: int | None = None) -> pd.DataFrame:
    """Joint simulation of a stat line (stats share the minutes draw, so
    points/rebounds/assists are positively correlated as in real games)."""
    p = params or PropParams()
    rng = np.random.default_rng(seed)
    mins = np.clip(rng.normal(proj_min, min_sd, n), 0, 48)
    out = {"min": mins}
    for s in STATS:
        mu = np.maximum(rate[s] * mins * (opp or {}).get(s, 1.0) * pace, 1e-6)
        r = p.dispersion[s]
        out[s] = rng.negative_binomial(r, r / (r + mu))
    df = pd.DataFrame(out)
    for c, parts in COMBOS.items():
        df[c] = df[parts].sum(axis=1)
    return df
