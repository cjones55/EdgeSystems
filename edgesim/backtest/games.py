"""Walk-forward backtest of the game model against real open/closing lines.

For every game the ratings only know games played before that date; the
Monte Carlo simulator then prices the spread, total and moneyline, and we
compare against what the market offered and what actually happened.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..gamesim import SimConfig, simulate_game
from ..odds import no_vig_probs
from ..ratings import RatingParams, TeamRatings

TRAIN_SEASONS = range(2011, 2016)   # used for tuning parameters
TEST_SEASONS = range(2016, 2022)    # never touched while tuning
JUICE = -110


def walk_forward(games: pd.DataFrame, params: RatingParams | None = None,
                 sim: SimConfig | None = None, min_games: int = 5) -> pd.DataFrame:
    sim = sim or SimConfig(n_sims=5000)
    r = TeamRatings(params or RatingParams())
    out = []
    for g in games.itertuples():
        if g.season != r.season:
            r.new_season(g.season)
        ready = r.games.get(g.home, 0) >= min_games and r.games.get(g.away, 0) >= min_games
        if ready and not np.isnan(g.close_spread):
            eh, ea = r.expected(g.home, g.away, g.home_rest, g.away_rest)
            res = simulate_game(eh, ea, SimConfig(**{**sim.__dict__, "seed": int(g.Index)}))
            row = dict(idx=g.Index, exp_home=eh, exp_away=ea, p_home_win=res.p_home_win(),
                       fair_spread=res.fair_spread(), fair_total=res.fair_total())
            for when in ("open", "close"):
                sp, tot = getattr(g, f"{when}_spread"), getattr(g, f"{when}_total")
                if not np.isnan(sp):
                    row[f"p_cover_{when}"], row[f"p_cpush_{when}"] = res.p_cover(sp)
                if not np.isnan(tot):
                    row[f"p_over_{when}"], row[f"p_opush_{when}"] = res.p_over(tot)
            out.append(row)
        r.update(g.home, g.away, g.home_pts, g.away_pts, g.home_rest, g.away_rest)
    pred = pd.DataFrame(out).set_index("idx")
    return games.join(pred, how="inner")


# ---------------------------------------------------------------------------
def _settle(margin_vs_line: np.ndarray) -> np.ndarray:
    """Units won per 1 unit risked at -110 given (result - line) from the bettor's side."""
    win = 100 / 110
    return np.where(margin_vs_line > 0, win, np.where(margin_vs_line < 0, -1.0, 0.0))


def spread_bets(pred: pd.DataFrame, when: str = "close", min_edge: float = 0.0) -> pd.DataFrame:
    p = pred[f"p_cover_{when}"]
    push = pred[f"p_cpush_{when}"]
    p_home = p / (1 - push).clip(lower=1e-9)
    side = np.where(p_home >= 0.5, 1, -1)              # 1 = bet home
    edge = np.abs(p_home - 0.5)
    margin = pred.home_pts - pred.away_pts
    res = side * (margin + pred[f"{when}_spread"])
    df = pd.DataFrame(dict(season=pred.season, edge=edge, side=side, units=_settle(res.values),
                           prob=np.maximum(p_home, 1 - p_home)), index=pred.index)
    return df[df.edge >= min_edge]


def total_bets(pred: pd.DataFrame, when: str = "close", min_edge: float = 0.0) -> pd.DataFrame:
    p = pred[f"p_over_{when}"]
    push = pred[f"p_opush_{when}"]
    p_over = p / (1 - push).clip(lower=1e-9)
    side = np.where(p_over >= 0.5, 1, -1)
    edge = np.abs(p_over - 0.5)
    res = side * (pred.home_pts + pred.away_pts - pred[f"{when}_total"])
    df = pd.DataFrame(dict(season=pred.season, edge=edge, side=side, units=_settle(res.values),
                           prob=np.maximum(p_over, 1 - p_over)), index=pred.index)
    return df[df.edge >= min_edge]


def record(bets: pd.DataFrame) -> dict:
    n = len(bets)
    w, l = int((bets.units > 0).sum()), int((bets.units < 0).sum())
    roi = bets.units.sum() / n if n else float("nan")
    # standard error of ROI, for an honest "is this just noise?" read
    se = bets.units.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return dict(bets=n, wins=w, losses=l, pushes=n - w - l,
                win_pct=w / (w + l) if w + l else float("nan"), roi=roi, roi_se=se,
                z=roi / se if se and se > 0 else float("nan"))


def edge_table(bets: pd.DataFrame, cuts=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10)) -> pd.DataFrame:
    rows = []
    for c in cuts:
        rows.append({"min_edge": c, **record(bets[bets.edge >= c])})
    return pd.DataFrame(rows)


def moneyline_scores(pred: pd.DataFrame) -> dict:
    df = pred.dropna(subset=["home_ml", "away_ml"])
    mkt = np.array([no_vig_probs([h, a])[0] for h, a in zip(df.home_ml, df.away_ml)])
    y = (df.home_pts > df.away_pts).astype(float).values
    model = df.p_home_win.clip(0.001, 0.999).values

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
    """Bet at the OPEN line on the model's side when model and open disagree.

    Closing line value (did the market move our way by close?) is the
    standard test of whether a betting model has real information, because
    closing lines are the sharpest number available and far less noisy than
    win/loss results.
    """
    df = pred.dropna(subset=["open_spread", "close_spread"])
    diff = df.fair_spread - df.open_spread          # negative -> model likes home more
    pick = df[np.abs(diff) >= min_disagree]
    side = np.where(pick.fair_spread < pick.open_spread, 1, -1)  # 1 = home
    move_toward_pick = side * (pick.open_spread - pick.close_spread)
    margin = pick.home_pts - pick.away_pts
    return dict(
        bets=len(pick),
        beat_close_pct=float((move_toward_pick > 0).mean()),
        lost_to_close_pct=float((move_toward_pick < 0).mean()),
        avg_clv_pts=float(move_toward_pick.mean()),
        open_line_ats=record(pd.DataFrame(dict(units=_settle((side * (margin + pick.open_spread)).values)))),
    )


def steam_follow(pred: pd.DataFrame, min_move: float = 1.5) -> dict:
    """Proxy test for 'follow the sharp move': when the spread moves a lot from
    open to close, how does the side it moved toward do?"""
    df = pred.dropna(subset=["open_spread", "close_spread"])
    move = df.open_spread - df.close_spread          # + means toward home
    big = df[np.abs(move) >= min_move]
    side = np.sign(move[np.abs(move) >= min_move])
    margin = big.home_pts - big.away_pts
    return dict(
        games=len(big),
        at_close=record(pd.DataFrame(dict(units=_settle((side * (margin + big.close_spread)).values)))),
        at_open=record(pd.DataFrame(dict(units=_settle((side * (margin + big.open_spread)).values)))),
    )


def card_replay(pred: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Replay the daily card's 'B: model edge' rule (blend + min EV) on history,
    at the open (-110) and at the close (-110 for spread/total, real closing ML)."""
    from ..odds import expected_value
    from ..plays import PlayConfig
    cfg = cfg or PlayConfig()
    rows = []
    for when in ("open", "close"):
        for market, pcol, line in (("spread", "p_cover", "spread"), ("total", "p_over", "total")):
            df = pred.dropna(subset=[f"{when}_{line}", f"{pcol}_{when}"])
            push = df[f"{'p_cpush' if market == 'spread' else 'p_opush'}_{when}"]
            pa = df[f"{pcol}_{when}"] / (1 - push).clip(lower=1e-9)
            units = []
            for side in (1, -1):
                pm = pa if side == 1 else 1 - pa
                pb = 0.5 + cfg.blend[market] * (pm - 0.5)
                ev = pb * (1 - push) * (100 / 110) - (1 - pb) * (1 - push)
                take = ev >= cfg.min_ev
                if market == "spread":
                    res = side * (df.home_pts - df.away_pts + df[f"{when}_spread"])
                else:
                    res = side * (df.home_pts + df.away_pts - df[f"{when}_total"])
                units.append(_settle(res[take].values))
            u = np.concatenate(units)
            rows.append(dict(market=market, line=when, plays=len(u),
                             plays_per_100_games=100 * len(u) / max(len(df), 1),
                             **{k: v for k, v in record(pd.DataFrame(dict(units=u))).items()
                                if k in ("win_pct", "roi", "z")}))
    df = pred.dropna(subset=["home_ml", "away_ml"])
    u = []
    for r in df.itertuples():
        mh = no_vig_probs([r.home_ml, r.away_ml])[0]
        for p_model, mkt, price, won in ((r.p_home_win, mh, r.home_ml, r.home_pts > r.away_pts),
                                         (1 - r.p_home_win, 1 - mh, r.away_ml, r.away_pts > r.home_pts)):
            pb = mkt + cfg.blend["ml"] * (p_model - mkt)
            if expected_value(pb, price) >= cfg.min_ev:
                u.append((price / 100 if price > 0 else 100 / -price) if won else -1.0)
    rows.append(dict(market="ml", line="close", plays=len(u), plays_per_100_games=100 * len(u) / len(df),
                     **{k: v for k, v in record(pd.DataFrame(dict(units=np.array(u)))).items()
                        if k in ("win_pct", "roi", "z")}))
    return pd.DataFrame(rows)
