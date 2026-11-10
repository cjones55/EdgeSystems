"""Projected box scores for the game drill-in view.

Team level (every sport): the full game simulation, so score distribution,
most likely final scores, win/cover/over chances at the market numbers.
Player level: NBA (minutes x per-minute rates x opponent) and NFL (the props
projections), with a "rest of roster" row so the players add up to the team's
simulated score where that applies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NBA_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]


def game_sim(g, ratings, sport: str, n_sims: int = 20000, seed: int = 7):
    """Re-run the card's simulation for one GameMarket and keep the raw draws."""
    from .sports import get
    eh, ea = ratings.expected(g.home, g.away, g.home_rest, g.away_rest, g.neutral)
    eh, ea = max(eh + g.home_adj, 1.0), max(ea + g.away_adj, 1.0)
    return get(sport).sim_matchup(ratings, g.home, g.away, eh, ea, n_sims=n_sims, seed=seed)


def team_summary(sim, g) -> dict:
    s = sim.summary()
    out = dict(home_pts=s["home_pts"], away_pts=s["away_pts"], home_win=s["home_win"],
               fair_spread=s["fair_spread"], fair_total=s["fair_total"])
    if g.spread is not None:
        out["home_cover"], out["spread_push"] = sim.p_cover(g.spread)
    if g.total is not None:
        out["over"], out["total_push"] = sim.p_over(g.total)
    return out


def likely_scores(sim, top: int = 10) -> pd.DataFrame:
    """Most common exact final scores in the simulation."""
    home = np.round(sim.home if hasattr(sim, "home") else (sim.total + sim.margin) / 2).astype(int)
    away = np.round(sim.away if hasattr(sim, "away") else (sim.total - sim.margin) / 2).astype(int)
    df = pd.DataFrame(dict(away=away, home=home)).value_counts(normalize=True).head(top)
    return df.rename("chance").reset_index()


def nba_box(feat: pd.DataFrame, team: str, team_pts: float | None = None) -> pd.DataFrame:
    """Expected line for each projected player on ``team``."""
    d = feat[feat.team == team]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for r in d.itertuples(index=False):
        row = dict(player=r.player, min=r.proj_min)
        for s in NBA_STATS:
            row[s] = getattr(r, f"rate_{s}") * r.proj_min * getattr(r, f"opp_{s}", 1.0)
        rows.append(row)
    box = pd.DataFrame(rows).sort_values("min", ascending=False)
    rest_min = max(0.0, 240 - box["min"].sum())
    rest = {s: BENCH_RATE[s] * rest_min for s in NBA_STATS}
    if team_pts is not None and box.pts.sum() > 0:
        # make the players add up to the simulated team score (bench at a typical rate)
        scale = float(np.clip((team_pts - rest["pts"]) / box.pts.sum(), 0.8, 1.2))
        box["pts"] *= scale
        rest["pts"] = max(0.0, team_pts - box.pts.sum())
    if rest_min > 0.5:
        box = pd.concat([box, pd.DataFrame([dict(player="Rest of roster", min=rest_min, **rest)])],
                        ignore_index=True)
    total = box[["min", *NBA_STATS]].sum().to_dict()
    return pd.concat([box, pd.DataFrame([dict(player="TEAM", **total)])],
                     ignore_index=True).astype({c: float for c in ["min", *NBA_STATS]})


# per-minute output of a typical bench player
BENCH_RATE = {"pts": 0.40, "reb": 0.18, "ast": 0.09, "fg3m": 0.04, "stl": 0.03, "blk": 0.02, "tov": 0.05}


NFL_COLS = {"QB": ["pass_cmp", "pass_att", "pass_yds", "pass_tds", "pass_ints", "rush_att",
                   "rush_yds", "anytime_td"],
            "RB": ["rush_att", "rush_yds", "rec", "rec_yds", "anytime_td"],
            "WR": ["rec", "rec_yds", "rush_yds", "anytime_td"],
            "TE": ["rec", "rec_yds", "anytime_td"]}


def nfl_box(feat: pd.DataFrame, team: str) -> dict[str, pd.DataFrame]:
    """Projected passing / rushing / receiving tables for ``team``.

    Players need an established role in a stat (the props eligibility rules) to
    get a number for it. ``anytime_td`` is converted to a probability."""
    from .nfl_props import eligible
    d = feat[feat.team == team].copy()
    if d.empty:
        return {}
    d["td_chance"] = 1 - np.exp(-d.proj_anytime_td.fillna(0).clip(lower=0))
    nice = {"proj_pass_cmp": "cmp", "proj_pass_att": "att", "proj_pass_yds": "yds",
            "proj_pass_tds": "TD", "proj_pass_ints": "INT", "proj_rush_att": "car",
            "proj_rush_yds": "yds", "proj_rec": "rec", "proj_rec_yds": "yds"}
    out = {}
    # who shows up in each table follows the props role rules; the numbers are the raw projections
    passing = d[eligible(d, "pass_att")].sort_values("proj_pass_att", ascending=False).head(1)
    if not passing.empty:
        out["Passing"] = passing[["player", "proj_pass_cmp", "proj_pass_att", "proj_pass_yds",
                                  "proj_pass_tds", "proj_pass_ints"]].rename(columns=nice)
    rushing = d[eligible(d, "rush_yds") | eligible(d, "rush_att")].sort_values("proj_rush_yds", ascending=False)
    if not rushing.empty:
        out["Rushing"] = rushing[["player", "position", "proj_rush_att", "proj_rush_yds",
                                  "td_chance"]].rename(columns=nice)
    receiving = d[eligible(d, "rec_yds") | eligible(d, "rec")].sort_values("proj_rec_yds", ascending=False)
    if not receiving.empty:
        out["Receiving"] = receiving[["player", "position", "proj_rec", "proj_rec_yds",
                                      "td_chance"]].rename(columns=nice)
    return out
