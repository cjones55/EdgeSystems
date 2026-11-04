"""Forward test: log every live play, then grade it against the closing line
and the final score. Closing line value (CLV) is the fastest honest read on
whether the plays (and especially the sharp signals, which have no free
historical data) have real edge: you need hundreds of bets to trust a win
rate, but CLV stabilises much sooner.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .odds import no_vig_probs

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "plays_log.csv"
SNAPS = ROOT / "data" / "snapshots"


def log_plays(plays: pd.DataFrame, day: date, path: Path = LOG) -> int:
    if plays.empty:
        return 0
    cols = ["game", "home", "away", "market", "side", "line", "pick", "price", "book", "tier",
            "model_prob", "blended_prob", "market_prob", "ev", "kelly", "sharp_score"]
    df = plays[[c for c in cols if c in plays.columns]].copy()
    df.insert(0, "day", str(day))
    df.insert(1, "logged_at", datetime.now().isoformat(timespec="seconds"))
    if path.exists():
        old = pd.read_csv(path)
        # one entry per day/game/market/side: keep the first time it was flagged
        key = ["day", "game", "market", "side"]
        new_keys = df[key].astype(str).agg("|".join, axis=1)
        old_keys = set(old[key].astype(str).agg("|".join, axis=1))
        df = df[~new_keys.isin(old_keys)]
        df = pd.concat([old, df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


def closing_lines(day: str) -> pd.DataFrame:
    """Consensus from the last odds snapshot of the day (run the daily command
    again close to tip-off to capture a real closing number)."""
    p = SNAPS / f"{day}.csv"
    if not p.exists():
        return pd.DataFrame()
    s = pd.read_csv(p)
    last = s[s.pulled_at == s.pulled_at.max()]
    return last.groupby(["home", "away", "market", "side"]).agg(
        close_point=("point", "median"), close_price=("price", "median")).reset_index()


def settle(market: str, side: str, line: float, home_pts: int, away_pts: int) -> int:
    """+1 win, 0 push, -1 loss."""
    margin = home_pts - away_pts
    if market == "spread":
        r = (margin if side == "home" else -margin) + line
    elif market == "total":
        r = (home_pts + away_pts - line) * (1 if side == "over" else -1)
    else:
        r = margin if side == "home" else -margin
    return int(np.sign(r))


def payout(result: int, price: float) -> float:
    if result > 0:
        return price / 100 if price > 0 else 100 / -price
    return 0.0 if result == 0 else -1.0


def grade(path: Path = LOG, scores: dict | None = None) -> pd.DataFrame:
    """Fill in result, units and CLV for logged plays whose games are final.

    ``scores`` maps (day, home, away) -> (home_pts, away_pts); fetched from
    ESPN when not given.
    """
    if not path.exists():
        return pd.DataFrame()
    log = pd.read_csv(path)
    for c in ("result", "units", "close_line", "clv_pts", "clv_prob"):
        if c not in log.columns:
            log[c] = np.nan
    todo = log[log.result.isna()]
    if scores is None:
        from .providers import espn
        scores = {}
        for day in todo.day.unique():
            for g in espn.scoreboard(date.fromisoformat(day)):
                if g["completed"]:
                    scores[(day, g["home"], g["away"])] = (g["home_pts"], g["away_pts"])
    closes = {d: closing_lines(d) for d in todo.day.unique()}
    for i, r in todo.iterrows():
        sc = scores.get((r.day, r.home, r.away))
        if sc is not None:
            res = settle(r.market, r.side, r.line, *sc)
            log.loc[i, "result"] = res
            log.loc[i, "units"] = payout(res, r.price)
        c = closes.get(r.day)
        if c is not None and not c.empty:
            m = c[(c.home == r.home) & (c.away == r.away) & (c.market == r.market)]
            mine, other = m[m.side == r.side], m[m.side != r.side]
            if not mine.empty:
                if r.market in ("spread", "total") and not pd.isna(r.line):
                    cl = mine.close_point.iloc[0]
                    log.loc[i, "close_line"] = cl
                    # spread: getting more points than the close is good;
                    # over: a lower number than the close is good, under: higher
                    log.loc[i, "clv_pts"] = {"spread": r.line - cl, "over": cl - r.line,
                                             "under": r.line - cl}["spread" if r.market == "spread" else r.side]
                if not other.empty:
                    close_p = no_vig_probs([mine.close_price.iloc[0], other.close_price.iloc[0]])[0]
                    log.loc[i, "clv_prob"] = close_p - (1 / (1 + (r.price / 100 if r.price > 0 else 100 / -r.price)))
    log.to_csv(path, index=False)
    return log


def summary(log: pd.DataFrame) -> pd.DataFrame:
    done = log.dropna(subset=["result"])
    if done.empty:
        return pd.DataFrame()
    g = done.groupby("tier")
    return pd.DataFrame(dict(
        bets=g.size(), wins=g.result.apply(lambda s: (s > 0).sum()),
        losses=g.result.apply(lambda s: (s < 0).sum()), units=g.units.sum(),
        roi=g.units.mean(), avg_clv_pts=g.clv_pts.mean(),
        beat_close_pct=g.clv_pts.apply(lambda s: (s.dropna() > 0).mean() if s.notna().any() else np.nan),
    )).reset_index()
