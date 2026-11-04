"""Walk-forward backtest of the game model against real open/closing lines.

For every game the ratings only know games played before that date; the
sport's Monte Carlo simulator then prices the spread (or puck line), total
and moneyline, and we compare against what the market offered and what
actually happened. Bets settle at the real prices in the archive when it has
them (NHL, college basketball) and at -110 otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import sports
from ..odds import american_to_decimal, expected_value, implied_prob, no_vig_probs
from ..ratings import TeamRatings

# kept for backwards compatibility (NBA)
TRAIN_SEASONS = range(2011, 2016)
TEST_SEASONS = range(2016, 2022)
MIN_GAMES = {"nba": 5, "nfl": 2, "nhl": 5, "ncaaf": 2, "ncaab": 4}


def walk_forward(games: pd.DataFrame, sport: str = "nba", params=None, sim=None,
                 n_sims: int = 5000, min_games: int | None = None) -> pd.DataFrame:
    sp = sports.get(sport)
    r = TeamRatings(params or sp.rating)
    min_games = MIN_GAMES[sport] if min_games is None else min_games
    out = []
    has = games[["close_spread", "close_total", "home_ml"]].notna().any(axis=1).values
    for i, g in enumerate(games.itertuples()):
        if g.season != r.season:
            r.new_season(g.season)
        ready = r.games.get(g.home, 0) >= min_games and r.games.get(g.away, 0) >= min_games
        neutral = bool(getattr(g, "neutral", False))
        if ready and has[i]:
            eh, ea = r.expected(g.home, g.away, g.home_rest, g.away_rest, neutral)
            if sim is not None:   # explicit config object (tests)
                cfg = type(sim)(**{**sim.__dict__, "seed": int(g.Index)})
                res = sp.simulate(eh, ea, cfg)
            else:
                res = sp.sim(eh, ea, n_sims=n_sims, seed=int(g.Index))
            tie = float((res.margin == 0).mean())
            row = dict(idx=g.Index, exp_home=eh, exp_away=ea, p_home_win=res.p_home_win(),
                       p_tie=tie, fair_spread=res.fair_spread(), fair_total=res.fair_total())
            for when in ("open", "close"):
                s, t = getattr(g, f"{when}_spread"), getattr(g, f"{when}_total")
                if not np.isnan(s):
                    row[f"p_cover_{when}"], row[f"p_cpush_{when}"] = res.p_cover(s)
                if not np.isnan(t):
                    row[f"p_over_{when}"], row[f"p_opush_{when}"] = res.p_over(t)
            out.append(row)
        r.update(g.home, g.away, g.home_pts, g.away_pts, g.home_rest, g.away_rest,
                 season=g.season, neutral=neutral)
    pred = pd.DataFrame(out).set_index("idx")
    return games.join(pred, how="inner")


# ---------------------------------------------------------------------------
def _settle(result: np.ndarray, price=-110.0) -> np.ndarray:
    """Units won per 1 unit risked given (result - line) from the bettor's side."""
    price = np.broadcast_to(np.asarray(price, dtype=float), np.shape(result))
    win = np.where(price > 0, price / 100, 100 / np.abs(price))
    return np.where(result > 0, win, np.where(result < 0, -1.0, 0.0))


def _ev(p_win, p_push, price):
    dec = np.vectorize(american_to_decimal)(price)
    return p_win * (dec - 1) - (1 - p_win - p_push)


def _two_sided(pred, p_a, push, price_a, price_b, result_a):
    """Pick the side with the higher EV; return bets frame."""
    ev_a = _ev(p_a, push, price_a)
    ev_b = _ev(1 - p_a - push, push, price_b)
    side = np.where(ev_a >= ev_b, 1, -1)
    ev = np.maximum(ev_a, ev_b)
    price = np.where(side == 1, price_a, price_b)
    units = _settle(side * result_a, price)
    return pd.DataFrame(dict(season=pred.season.values, edge=ev, side=side, units=units,
                             price=price), index=pred.index)


def spread_bets(pred: pd.DataFrame, when: str = "close", min_edge: float = -1.0) -> pd.DataFrame:
    p = pred.dropna(subset=[f"p_cover_{when}", f"{when}_spread"])
    margin = (p.home_pts - p.away_pts).values
    b = _two_sided(p, p[f"p_cover_{when}"].values, p[f"p_cpush_{when}"].values,
                   p.spread_home_price.values, p.spread_away_price.values,
                   margin + p[f"{when}_spread"].values)
    return b[b.edge >= min_edge]


def total_bets(pred: pd.DataFrame, when: str = "close", min_edge: float = -1.0) -> pd.DataFrame:
    p = pred.dropna(subset=[f"p_over_{when}", f"{when}_total"])
    tot = (p.home_pts + p.away_pts).values
    b = _two_sided(p, p[f"p_over_{when}"].values, p[f"p_opush_{when}"].values,
                   p.over_price.values, p.under_price.values, tot - p[f"{when}_total"].values)
    return b[b.edge >= min_edge]


def ml_bets(pred: pd.DataFrame, min_edge: float = -1.0, which: str = "close") -> pd.DataFrame:
    hcol, acol = ("home_ml", "away_ml") if which == "close" else ("open_home_ml", "open_away_ml")
    p = pred.dropna(subset=[hcol, acol])
    margin = (p.home_pts - p.away_pts).values
    b = _two_sided(p, p.p_home_win.values, p.p_tie.values, p[hcol].values, p[acol].values, margin)
    return b[b.edge >= min_edge]


def record(bets: pd.DataFrame) -> dict:
    n = len(bets)
    w, l = int((bets.units > 0).sum()), int((bets.units < 0).sum())
    roi = bets.units.sum() / n if n else float("nan")
    se = bets.units.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return dict(bets=n, wins=w, losses=l, pushes=n - w - l,
                win_pct=w / (w + l) if w + l else float("nan"), roi=roi, roi_se=se,
                z=roi / se if se and se > 0 else float("nan"))


def edge_table(bets: pd.DataFrame, cuts=(-1.0, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10)) -> pd.DataFrame:
    rows = []
    for c in cuts:
        rows.append({"min_ev": "all games" if c <= -1 else c, **record(bets[bets.edge >= c])})
    return pd.DataFrame(rows)


def moneyline_scores(pred: pd.DataFrame) -> dict:
    df = pred.dropna(subset=["home_ml", "away_ml"])
    df = df[df.home_pts != df.away_pts]
    mkt = np.array([no_vig_probs([h, a])[0] for h, a in zip(df.home_ml, df.away_ml)])
    y = (df.home_pts > df.away_pts).astype(float).values
    model = (df.p_home_win / (1 - df.p_tie)).clip(0.001, 0.999).values

    def ll(p):
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def brier(p):
        return float(((p - y) ** 2).mean())

    return dict(games=len(df), model_logloss=ll(model), market_logloss=ll(mkt),
                model_brier=brier(model), market_brier=brier(mkt),
                coinflip_logloss=float(np.log(2)))


def calibration(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append(dict(bucket=f"{edges[b]:.1f}-{edges[b+1]:.1f}", n=int(m.sum()),
                             predicted=float(prob[m].mean()), actual=float(outcome[m].mean())))
    return pd.DataFrame(rows)


def clv(pred: pd.DataFrame, min_disagree: float = 1.0) -> dict:
    """Bet at the OPEN spread on the model's side when model and open disagree;
    did the market move our way by the close? (closing line value)"""
    df = pred.dropna(subset=["open_spread", "close_spread"])
    diff = df.fair_spread - df.open_spread
    pick = df[np.abs(diff) >= min_disagree]
    side = np.where(pick.fair_spread < pick.open_spread, 1, -1)
    move = side * (pick.open_spread - pick.close_spread)
    margin = pick.home_pts - pick.away_pts
    return dict(
        bets=len(pick), beat_close_pct=float((move > 0).mean()) if len(pick) else np.nan,
        lost_to_close_pct=float((move < 0).mean()) if len(pick) else np.nan,
        avg_clv_pts=float(move.mean()) if len(pick) else np.nan,
        open_line_ats=record(pd.DataFrame(dict(units=_settle((side * (margin + pick.open_spread)).values)))),
    )


def clv_ml(pred: pd.DataFrame, min_edge: float = 0.02) -> dict:
    """Moneyline version (NHL): model side at the opening price vs closing no-vig price."""
    df = pred.dropna(subset=["open_home_ml", "open_away_ml", "home_ml", "away_ml"])
    o = np.array([no_vig_probs([h, a])[0] for h, a in zip(df.open_home_ml, df.open_away_ml)])
    c = np.array([no_vig_probs([h, a])[0] for h, a in zip(df.home_ml, df.away_ml)])
    m = (df.p_home_win / (1 - df.p_tie)).values
    pick = np.abs(m - o) >= min_edge
    side = np.where(m > o, 1, -1)[pick]
    move = side * (c - o)[pick]
    return dict(bets=int(pick.sum()), beat_close_pct=float((move > 0).mean()),
                lost_to_close_pct=float((move < 0).mean()), avg_clv_prob=float(move.mean()))


def steam_follow(pred: pd.DataFrame, min_move: float = 1.5) -> dict:
    df = pred.dropna(subset=["open_spread", "close_spread"])
    move = df.open_spread - df.close_spread
    big = df[np.abs(move) >= min_move]
    side = np.sign(move[np.abs(move) >= min_move])
    margin = big.home_pts - big.away_pts
    return dict(
        games=len(big),
        at_close=record(pd.DataFrame(dict(units=_settle((side * (margin + big.close_spread)).values)))),
        at_open=record(pd.DataFrame(dict(units=_settle((side * (margin + big.open_spread)).values)))),
    )


def steam_follow_ml(pred: pd.DataFrame, min_move: float = 0.03) -> dict:
    df = pred.dropna(subset=["open_home_ml", "open_away_ml", "home_ml", "away_ml"])
    o = np.array([implied_prob(h) for h in df.open_home_ml])
    c = np.array([implied_prob(h) for h in df.home_ml])
    mv = c - o
    big = np.abs(mv) >= min_move
    d = df[big]
    side = np.sign(mv[big])
    margin = (d.home_pts - d.away_pts).values
    close_price = np.where(side > 0, d.home_ml, d.away_ml)
    open_price = np.where(side > 0, d.open_home_ml, d.open_away_ml)
    return dict(games=int(big.sum()),
                at_close=record(pd.DataFrame(dict(units=_settle(side * margin, close_price)))),
                at_open=record(pd.DataFrame(dict(units=_settle(side * margin, open_price)))))


def card_replay(pred: pd.DataFrame, cfg=None, sport: str = "nba") -> pd.DataFrame:
    """Replay the daily card's 'B: model edge' rule (market blend + min EV)."""
    from ..plays import PlayConfig
    sp = sports.get(sport)
    cfg = cfg or PlayConfig(blend=dict(sp.blend))
    rows = []

    def run(market, when, p_a, push, pa, pb, result_a):
        units = []
        for side, p_m, price, other in ((1, p_a, pa, pb), (-1, 1 - p_a - push, pb, pa)):
            mkt = np.array([no_vig_probs([x, y])[0] for x, y in zip(price, other)])
            decided = (1 - push).clip(1e-9)
            pb_ = mkt + cfg.blend[market] * (p_m / decided - mkt)
            ev = _ev(pb_ * decided, push, price)
            take = ev >= cfg.min_ev
            units.append(_settle(side * result_a[take], price[take]))
        u = np.concatenate(units)
        rows.append(dict(market=market, line=when, plays=len(u),
                         plays_per_100_games=100 * len(u) / max(len(p_a), 1),
                         **{k: v for k, v in record(pd.DataFrame(dict(units=u))).items()
                            if k in ("win_pct", "roi", "z")}))

    for when in ("open", "close"):
        d = pred.dropna(subset=[f"{when}_spread", f"p_cover_{when}"]) \
            if f"p_cover_{when}" in pred else pred.iloc[:0]
        if len(d):
            run("spread", when, d[f"p_cover_{when}"].values, d[f"p_cpush_{when}"].values,
                d.spread_home_price.values, d.spread_away_price.values,
                (d.home_pts - d.away_pts + d[f"{when}_spread"]).values)
        d = pred.dropna(subset=[f"{when}_total", f"p_over_{when}"]) \
            if f"p_over_{when}" in pred else pred.iloc[:0]
        if len(d):
            run("total", when, d[f"p_over_{when}"].values, d[f"p_opush_{when}"].values,
                d.over_price.values, d.under_price.values,
                (d.home_pts + d.away_pts - d[f"{when}_total"]).values)
    d = pred.dropna(subset=["home_ml", "away_ml"])
    if len(d):
        run("ml", "close", d.p_home_win.values, d.p_tie.values, d.home_ml.values,
            d.away_ml.values, (d.home_pts - d.away_pts).values)
    return pd.DataFrame(rows)
