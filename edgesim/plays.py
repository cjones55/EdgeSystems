"""Turn simulations, market prices and sharp signals into a daily card.

How a play is graded (all numbers come from the backtest, see BACKTEST.md):
  * The sim's probability is blended with the market's no-vig probability.
    Over 7,000 held-out games the model carried only ~8-15% extra weight
    beyond the opening line and ~0% beyond the close, so we lean on the
    market and only let the model move it a little (``blend``).
  * EV is computed from the blended probability at the best available price.
  * Sharp signals (reverse line movement, money/ticket splits, steam) are
    reported alongside and used to tier plays. They are NOT folded into the
    probability, because no free historical public-betting data exists to
    validate how much they are worth. Track them with ``edgesim grade``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .odds import expected_value, kelly_fraction, no_vig_probs
from .props import PropParams, prob_over
from .ratings import TeamRatings
from .sharp import MarketSplit, SharpConfig, detect


@dataclass
class GameMarket:
    home: str
    away: str
    home_rest: float = 2
    away_rest: float = 2
    start: str | None = None
    spread: float | None = None          # home spread
    spread_home_price: float = -110
    spread_away_price: float = -110
    total: float | None = None
    over_price: float = -110
    under_price: float = -110
    home_ml: float | None = None
    away_ml: float | None = None
    books: dict = field(default_factory=dict)   # (market, side) -> best book name
    open_spread: float | None = None
    open_total: float | None = None
    open_home_ml: float | None = None
    neutral: bool = False
    home_name: str | None = None         # display names (college teams are keyed by id)
    away_name: str | None = None


@dataclass
class PlayConfig:
    sport: str = "nba"
    blend: dict = field(default_factory=lambda: {"spread": 0.12, "total": 0.10, "ml": 0.04,
                                                 "prop": 0.5})
    min_ev: float = 0.01            # minimum EV (per unit) to list as a bet
    lean_ev: float = -0.05          # shown as a lean if sharp signal agrees (about the vig)
    sharp_min_score: float = 40.0
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.03
    n_sims: int = 20000
    seed: int = 7

    @classmethod
    def for_sport(cls, sport: str, **kw) -> "PlayConfig":
        from .sports import get
        sp = get(sport)
        return cls(sport=sport, blend=dict(sp.blend), sharp=sp.sharp, **kw)
    sharp: SharpConfig = field(default_factory=SharpConfig)


def _fmt_line(x: float) -> str:
    return f"{x:+g}" if x else "PK"


def _fmt_price(x: float) -> str:
    return f"{x:+.0f}"


def evaluate_game(g: GameMarket, ratings: TeamRatings, splits: dict | None = None,
                  cfg: PlayConfig | None = None) -> tuple[dict, list[dict]]:
    """Returns (sim summary row, candidate plays for every side of every market)."""
    cfg = cfg or PlayConfig()
    splits = splits or {}
    from .sports import get
    eh, ea = ratings.expected(g.home, g.away, g.home_rest, g.away_rest, g.neutral)
    res = get(cfg.sport).sim(eh, ea, n_sims=cfg.n_sims, seed=cfg.seed)
    s = res.summary()
    hn, an = g.home_name or g.home, g.away_name or g.away
    game = f"{an} {'vs' if g.neutral else '@'} {hn}"
    p_tie = float((res.margin == 0).mean())
    summary = dict(game=game, home=g.home, away=g.away, neutral=g.neutral, start=g.start, proj_home=s["home_pts"],
                   proj_away=s["away_pts"], home_win=s["home_win"], fair_spread=s["fair_spread"],
                   fair_total=s["fair_total"], market_spread=g.spread, market_total=g.total)
    cands = []

    def add(market, side_label, side_ab, p_model, p_push, price, other_price, line_desc,
            open_line, cur_line, book, side_key, line_val):
        mkt_p = no_vig_probs([price, other_price])[0]
        decided = 1 - p_push
        p_model_d = p_model / decided if decided > 0 else p_model
        p_blend_d = mkt_p + cfg.blend[market] * (p_model_d - mkt_p)
        p_win = p_blend_d * decided
        ev = expected_value(p_win, price, p_push)
        split = splits.get((g.home, g.away, market))
        if split is not None and split.open_line is None and open_line is not None:
            split = MarketSplit(market, open_line, split.current_line or cur_line,
                                split.tickets_a, split.money_a)
        if split is None and open_line is not None and cur_line is not None:
            split = MarketSplit(market, open_line, cur_line, None, None)  # movement only
        sig = detect(split, cfg.sharp) if split else None
        agrees = bool(sig and sig.side == side_ab and sig.score >= cfg.sharp_min_score)
        against = bool(sig and sig.side and sig.side != side_ab and sig.score >= cfg.sharp_min_score)
        cands.append(dict(
            game=game, home=g.home, away=g.away, market=market, side=side_key, line=line_val,
            pick=side_label + line_desc, price=price, book=book,
            model_prob=p_model_d, market_prob=mkt_p, blended_prob=p_blend_d, ev=ev,
            kelly=kelly_fraction(p_win, price, p_push, cfg.kelly_fraction, cfg.kelly_cap),
            sharp_side=("this side" if agrees else "other side" if against else "-"),
            sharp_score=sig.score if sig else 0.0,
            sharp_reasons="; ".join(sig.reasons) if sig else "",
            tickets_pct=(split.tickets_a if side_ab == "A" else 100 - split.tickets_a)
            if split and split.tickets_a is not None else None,
            money_pct=(split.money_a if side_ab == "A" else 100 - split.money_a)
            if split and split.money_a is not None else None,
            _agrees=agrees, _against=against))

    if g.spread is not None:
        ph, pp = res.p_cover(g.spread)
        add("spread", hn, "A", ph, pp, g.spread_home_price, g.spread_away_price,
            f" {_fmt_line(g.spread)}", g.open_spread, g.spread, g.books.get(("spread", "home")),
            "home", g.spread)
        add("spread", an, "B", 1 - ph - pp, pp, g.spread_away_price, g.spread_home_price,
            f" {_fmt_line(-g.spread)}", g.open_spread, g.spread, g.books.get(("spread", "away")),
            "away", -g.spread)
    if g.total is not None:
        po, pp = res.p_over(g.total)
        add("total", "Over", "A", po, pp, g.over_price, g.under_price, f" {g.total:g}",
            g.open_total, g.total, g.books.get(("total", "over")), "over", g.total)
        add("total", "Under", "B", 1 - po - pp, pp, g.under_price, g.over_price, f" {g.total:g}",
            g.open_total, g.total, g.books.get(("total", "under")), "under", g.total)
    if g.home_ml is not None and g.away_ml is not None:
        pw = res.p_home_win()      # NFL ties are graded as a push on the moneyline
        add("ml", hn, "A", pw, p_tie, g.home_ml, g.away_ml, " ML", g.open_home_ml, g.home_ml,
            g.books.get(("ml", "home")), "home", None)
        add("ml", an, "B", 1 - pw - p_tie, p_tie, g.away_ml, g.home_ml, " ML", None, None,
            g.books.get(("ml", "away")), "away", None)
    return summary, cands


def tier(c: dict, cfg: PlayConfig) -> str | None:
    if c["_against"]:
        return None
    if c["ev"] >= cfg.min_ev and c["_agrees"]:
        return "A: model + sharp"
    if c["ev"] >= cfg.min_ev:
        return "B: model edge"
    if c["_agrees"] and c["ev"] >= cfg.lean_ev and c["model_prob"] > 0.5:
        return "C: sharp lean"
    return None


def daily_card(games: list[GameMarket], ratings: TeamRatings, splits: dict | None = None,
               cfg: PlayConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (plays, every candidate, per-game sim summaries)."""
    cfg = cfg or PlayConfig()
    sums, cands = [], []
    for g in games:
        s, c = evaluate_game(g, ratings, splits, cfg)
        sums.append(s)
        cands.extend(c)
    allc = pd.DataFrame(cands)
    if allc.empty:
        return allc, allc, pd.DataFrame(sums)
    allc["tier"] = [tier(c, cfg) for c in cands]
    plays = allc[allc.tier.notna()].sort_values(["tier", "ev"], ascending=[True, False])
    drop = ["_agrees", "_against"]
    return plays.drop(columns=drop), allc.drop(columns=drop), pd.DataFrame(sums)


# ---------------------------------------------------------------------------
def prop_plays(lines: pd.DataFrame, features: pd.DataFrame, pace: dict | None = None,
               minutes_factor: dict | None = None, params: PropParams | None = None,
               cfg: PlayConfig | None = None) -> pd.DataFrame:
    """Price player props.

    ``lines``: player, team, stat, line, over_price, under_price, book
    ``features``: one row per player with rate_<stat>, proj_min, min_sd, opp_<stat>
    ``pace``: team -> pace multiplier from the game sim (expected total / league avg)
    ``minutes_factor``: team -> starter minutes multiplier for expected blowouts
    """
    params = params or PropParams()
    cfg = cfg or PlayConfig()
    pace, minutes_factor = pace or {}, minutes_factor or {}
    f = features.set_index("player")
    rows = []
    for r in lines.itertuples():
        if r.player not in f.index or f"rate_{r.stat}" not in f.columns:
            continue
        x = f.loc[r.player]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[-1]
        pm = x.proj_min * (minutes_factor.get(x.team, 1.0) if x.proj_min >= 28 else 1.0)
        mean_factor = x[f"opp_{r.stat}"] * pace.get(x.team, 1.0)
        size = params.dispersion.get(r.stat, 10)
        p_over_raw = float(prob_over(np.array([x[f"rate_{r.stat}"]]), np.array([pm]),
                                     np.array([x.min_sd]), np.array([r.line]), size,
                                     np.array([mean_factor]))[0])
        p_over_m = float(params.calibrate(r.stat, p_over_raw))
        proj = float(x[f"rate_{r.stat}"] * pm * mean_factor)
        mkt_over = no_vig_probs([r.over_price, r.under_price])[0]
        p_over_b = mkt_over + cfg.blend["prop"] * (p_over_m - mkt_over)
        for side, p, price, mp in (("Over", p_over_b, r.over_price, mkt_over),
                                   ("Under", 1 - p_over_b, r.under_price, 1 - mkt_over)):
            ev = expected_value(p, price)
            rows.append(dict(player=r.player, team=x.team, stat=r.stat, pick=f"{side} {r.line:g}",
                             price=price, book=getattr(r, "book", None), projection=proj,
                             proj_minutes=pm,
                             model_prob=p_over_m if side == "Over" else 1 - p_over_m,
                             market_prob=mp, blended_prob=p, ev=ev,
                             kelly=kelly_fraction(p, price, 0, cfg.kelly_fraction, cfg.kelly_cap)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["play"] = out.ev >= cfg.min_ev + 0.01   # props are noisier: need a bit more edge
    return out.sort_values("ev", ascending=False)
