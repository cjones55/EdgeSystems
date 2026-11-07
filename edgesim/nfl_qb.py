"""Quarterback availability for the NFL game sim.

Team ratings are learned from scores, so they quietly assume the usual starter
plays. When he is out, the team's expected points should drop by the gap
between him and whoever replaces him.

  QB value   expected points added per game (passing + rushing EPA, nflverse),
             exponentially weighted over his recent starts and shrunk toward
             the level of a typical backup who is forced to start
  regular    the QB who started most of the team's last 4 games
  backup     the team's next most used QB over the last two seasons
             (a typical replacement if there is none)
  adjustment POINT_SCALE x (backup value - regular value), added to the team's
             expected points (POINT_SCALE fitted on 2017-2021 starter-out games)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HALFLIFE = 16.0     # starts
SHRINK_GAMES = 30.0 # starts of prior weight toward replacement level (tuned 2016-2021)
POINT_SCALE = 2.0   # fitted in backtest_qb_adjustment (shrunk EPA gap -> points)


def starters(pg: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: the QB with the most pass attempts and his EPA."""
    qb = pg[(pg.position == "QB")].copy()
    qb["epa"] = qb.passing_epa.fillna(0) + qb.rushing_epa.fillna(0)
    qb = qb.sort_values("attempts", ascending=False).drop_duplicates(["team", "game_id"])
    return qb[["date", "season", "game_id", "team", "player_id", "player", "attempts", "epa"]] \
        .sort_values("date").reset_index(drop=True)


def replacement_level(st: pd.DataFrame) -> float:
    """Average EPA/game of QBs starting for a team that is not their regular starter."""
    st = mark_regulars(st)
    backups = st[(st.player_id != st.regular_id) & st.regular_id.notna()]
    return float(backups.epa.mean()) if len(backups) > 30 else -4.0


def mark_regulars(st: pd.DataFrame) -> pd.DataFrame:
    """Add regular_id: who started most of the team's previous 4 games."""
    st = st.sort_values("date").copy()
    reg = []
    for _, g in st.groupby("team", sort=False):
        ids = g.player_id.tolist()
        r = []
        for i in range(len(ids)):
            prev = ids[max(0, i - 4):i]
            r.append(max(set(prev), key=prev.count) if prev else None)
        reg.append(pd.Series(r, index=g.index))
    st["regular_id"] = pd.concat(reg)
    return st


class QBValues:
    """EPA/game value of every QB as of any date (only earlier starts count)."""

    def __init__(self, st: pd.DataFrame, repl: float):
        self.repl = repl
        self.by = {}
        for pid, g in st.sort_values("date").groupby("player_id"):
            dates = g.date.values
            epa = g.epa.values
            vals = []
            num = den = 0.0
            decay = 0.5 ** (1 / HALFLIFE)
            for e in epa:            # value after each start
                num = num * decay + e
                den = den * decay + 1
                vals.append(num / den)
            n = np.arange(1, len(epa) + 1)
            shrunk = (n * np.array(vals) + SHRINK_GAMES * repl) / (n + SHRINK_GAMES)
            self.by[pid] = (dates, shrunk)

    def at(self, pid, day) -> float:
        if pid not in self.by:
            return self.repl
        dates, vals = self.by[pid]
        i = np.searchsorted(dates, np.datetime64(day), side="left")
        return float(vals[i - 1]) if i > 0 else self.repl


BASE_DECAY = 0.92   # per game: roughly how fast the team rating forgets old games


def baseline(st: pd.DataFrame, team: str, day, vals: QBValues, n: int = 16) -> float:
    """QB quality already baked into the team's rating: a decayed average of the
    values of the QBs who started its recent games."""
    tg = st[(st.team == team) & (st.date < day)].tail(n)
    if tg.empty:
        return vals.repl
    w = BASE_DECAY ** np.arange(len(tg))[::-1]
    v = np.array([vals.at(p, day) for p in tg.player_id])
    return float(np.sum(w * v) / np.sum(w))


@dataclass
class QBStatus:
    team: str
    starter: str
    starter_value: float
    baseline: float
    adjustment: float           # points added to the team's expected score
    out: list

    def note(self) -> str:
        why = f"{', '.join(self.out)} out; " if self.out else ""
        return (f"{self.team} QB: {why}{self.starter} expected to start "
                f"({self.adjustment:+.1f} pts vs the QB play in the team's rating)")


def team_qb_status(pg: pd.DataFrame, team: str, day: pd.Timestamp, out_names: set[str],
                   st: pd.DataFrame | None = None, vals: QBValues | None = None) -> QBStatus | None:
    """Expected starter = the team's most recent starter not listed out (else its
    most used other QB over two seasons). Returns the points adjustment."""
    st = starters(pg) if st is None else st
    vals = vals or QBValues(st, replacement_level(st))
    tg = st[(st.team == team) & (st.date < day)]
    if tg.empty:
        return None
    out_qbs = sorted({n for n in tg.tail(8).player if str(n).lower() in out_names})
    avail = tg[~tg.player.str.lower().isin(out_names)]
    if not avail.empty and avail.date.max() >= tg.tail(4).date.min():
        pid, name = avail.player_id.iloc[-1], avail.player.iloc[-1]
    else:
        recent = pg[(pg.team == team) & (pg.position == "QB") & (pg.date < day)
                    & (pg.date >= day - pd.Timedelta(days=730))
                    & ~pg.player.str.lower().isin(out_names)]
        if recent.empty:
            pid, name = None, "a backup"
        else:
            pid, name = recent.groupby(["player_id", "player"]).attempts.sum().sort_values().index[-1]
    sv = vals.at(pid, day) if pid else vals.repl
    base = baseline(st, team, day, vals)
    return QBStatus(team, name, sv, base, POINT_SCALE * (sv - base), out_qbs)


def backtest_qb_adjustment(pred: pd.DataFrame, pg: pd.DataFrame,
                           schedule: str = "data/raw/nfl_players/games.csv") -> tuple[float, str]:
    """Fit POINT_SCALE from the walk-forward NFL predictions (exp_home/exp_away),
    using who actually started. Returns (fitted scale, markdown)."""
    from .nfl_props import _fix
    st = starters(pg)
    vals = QBValues(st, replacement_level(st))
    sched = pd.read_csv(schedule)
    sched["home_team"] = sched.home_team.map(_fix)
    sched["away_team"] = sched.away_team.map(_fix)
    sched["date"] = pd.to_datetime(sched.gameday)
    key = sched.set_index(["date", "home_team", "away_team"]).game_id.to_dict()
    sidx = st.set_index(["game_id", "team"]).player_id.to_dict()
    rows = []
    for r in pred.itertuples():
        gid = key.get((r.date, r.home, r.away))
        if gid is None:
            continue
        x = {}
        for team in (r.home, r.away):
            pid = sidx.get((gid, team))
            x[team] = (vals.at(pid, r.date) - baseline(st, team, r.date, vals)) if pid else 0.0
        rows.append(dict(season=r.season, resid=(r.home_pts - r.away_pts) - (r.exp_home - r.exp_away),
                         x=x[r.home] - x[r.away]))
    d = pd.DataFrame(rows)
    train, test = d[d.season <= 2018], d[d.season > 2018]
    scale = float(np.clip((train.x * train.resid).sum() / (train.x ** 2).sum(), 0, 2.0))
    big = test[test.x.abs() >= 4]

    def mae(df, k):
        return np.sqrt(((df.resid - k * df.x) ** 2).mean())
    md = "\n".join([
        "## Quarterback changes", "",
        "Each game's expected points are adjusted by the gap between the expected starter's "
        "value (EPA per game, shrunk toward a typical backup) and the QB play already baked into "
        "the team's rating. The scale is fitted on 2016-2018 and tested on 2019-2021; the "
        "half-life and shrinkage settings were chosen looking at 2016-2021.", "",
        f"Fitted scale: {scale:.2f} points per point of EPA gap.", "",
        "| test games | margin error (RMSE) without | with adjustment | change |", "|---|---|---|---|",
        f"| all {len(test)} | {mae(test, 0):.2f} | {mae(test, scale):.2f} | "
        f"{1 - mae(test, scale) / mae(test, 0):+.1%} |",
        f"| {len(big)} with a QB gap of 4+ EPA | {mae(big, 0):.2f} | {mae(big, scale):.2f} | "
        f"{1 - mae(big, scale) / mae(big, 0):+.1%} |",
    ])
    return scale, md
