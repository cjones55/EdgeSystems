"""Assemble everything for one day: ratings, markets, splits, props -> card.

Two sources of context:
  * ``live_context``  - today's real slate (ESPN + The Odds API + your splits)
  * ``demo_context``  - any date from the historical archive (2011-2022),
                        fully offline, so the whole system can be run and
                        inspected without API keys.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .plays import GameMarket, PlayConfig, daily_card, prop_plays
from .props import COMBOS, PropParams, build_features
from .ratings import TeamRatings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FITTED = DATA / "fitted_params.json"


def load_prop_params() -> PropParams:
    p = PropParams()
    if FITTED.exists():
        fit = json.loads(FITTED.read_text())
        p.dispersion.update(fit.get("dispersion", {}))
        p.calib.update(fit.get("calib", {}))
    return p


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    return re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s).strip().replace("  ", " ")


@dataclass
class DayContext:
    day: date
    ratings: TeamRatings
    games: list[GameMarket]
    splits: dict = field(default_factory=dict)
    prop_lines: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    injuries: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)
    source: str = "live"


@dataclass
class DayResult:
    ctx: DayContext
    plays: pd.DataFrame
    candidates: pd.DataFrame
    sims: pd.DataFrame
    props: pd.DataFrame


def run_day(ctx: DayContext, cfg: PlayConfig | None = None,
            prop_params: PropParams | None = None) -> DayResult:
    cfg = cfg or PlayConfig()
    prop_params = prop_params or load_prop_params()
    plays, cands, sims = daily_card(ctx.games, ctx.ratings, ctx.splits, cfg)
    props = pd.DataFrame()
    if not ctx.prop_lines.empty and not ctx.player_features.empty and not sims.empty:
        league_total = ctx.ratings.league_avg
        pace, mfac = {}, {}
        for s in sims.itertuples():
            t = (s.proj_home + s.proj_away) / league_total
            pace[s.home] = pace[s.away] = t
            blow = max(0.0, abs(s.proj_home - s.proj_away) - 8)
            mfac[s.home] = mfac[s.away] = 1 - 0.01 * blow
        props = prop_plays(ctx.prop_lines, ctx.player_features, pace, mfac, prop_params, cfg)
    return DayResult(ctx, plays, cands, sims, props)


# ---------------------------------------------------------------------------
def fit_ratings(results: pd.DataFrame, until: pd.Timestamp) -> TeamRatings:
    r = TeamRatings()
    for g in results[results.date < until].itertuples():
        r.update(g.home, g.away, g.home_pts, g.away_pts, getattr(g, "home_rest", 2),
                 getattr(g, "away_rest", 2), season=g.season)
    return r


def rest_days(results: pd.DataFrame, team: str, day: pd.Timestamp) -> float:
    played = results[((results.home == team) | (results.away == team)) & (results.date < day)]
    if played.empty:
        return 4
    return float(min((day - played.date.max()).days, 4))


def project_players(player_games: pd.DataFrame, upcoming: list[tuple[str, str]],
                    day: pd.Timestamp, params: PropParams | None = None,
                    lookback_days: int = 30) -> pd.DataFrame:
    """Project every recently active player on a team playing ``day``."""
    pg = player_games[(player_games.date < day) & (player_games["min"] > 0)]
    recent = pg[pg.date >= day - pd.Timedelta(days=lookback_days)]
    opp_of = {}
    for h, a in upcoming:
        opp_of[h], opp_of[a] = a, h
    roster = recent[recent.team.isin(opp_of)].groupby("player_id").agg(
        player=("player", "last"), team=("team", "last")).reset_index()
    roster = roster[roster.team.isin(opp_of)]
    stub = roster.assign(date=day, game_id=lambda d: "next_" + d.team, opp=roster.team.map(opp_of),
                         min=1.0)
    for s in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]:
        stub[s] = 0.0
    feat = build_features(pd.concat([pg, stub], ignore_index=True), params)
    out = feat[feat.game_id.astype(str).str.startswith("next_")].copy()
    out = out[out.n_prev >= 5]
    return out.drop(columns=["min", *["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"],
                             *COMBOS.keys()])


# ---------------------------------------------------------------------------
def demo_context(day: str | date, with_props: bool = True, top_players: int = 7) -> DayContext:
    from .data.historical import load_games
    from .providers.public_betting import load_csv, to_splits

    d = pd.Timestamp(day)
    hist = load_games()
    if not (hist.date == d).any():
        raise ValueError(f"No archived games on {d.date()} (archive covers 2011-12..2021-22)")
    ratings = fit_ratings(hist, d)
    todays = hist[hist.date == d]
    games = [GameMarket(home=g.home, away=g.away, home_rest=g.home_rest, away_rest=g.away_rest,
                        start=str(d.date()), spread=g.close_spread, total=g.close_total,
                        home_ml=None if np.isnan(g.home_ml) else g.home_ml,
                        away_ml=None if np.isnan(g.away_ml) else g.away_ml,
                        open_spread=None if np.isnan(g.open_spread) else g.open_spread,
                        open_total=None if np.isnan(g.open_total) else g.open_total)
             for g in todays.itertuples()]
    ctx = DayContext(day=d.date(), ratings=ratings, games=games, source="demo",
                     splits=to_splits(load_csv(d.date())))
    ctx.notes.append("Demo mode: lines are the archived open/close numbers; the 'current' "
                     "line is the close. Actual results are known, so this is for inspecting "
                     "the system, not for betting.")
    if ctx.splits:
        ctx.notes.append("Public betting % in demo mode come from a sample CSV and are illustrative.")
    if with_props:
        try:
            from .data.boxscores import load_player_games
            pg = load_player_games()
        except FileNotFoundError:
            pg = None
        if pg is not None and (pg.date == d).any():
            feat = project_players(pg, [(g.home, g.away) for g in games], d, load_prop_params())
            feat = feat.sort_values("proj_min", ascending=False).groupby("team").head(top_players)
            ctx.player_features = feat
            # proxy lines: last-10 average rounded to x.5 at -115 both ways
            hist_p = pg[(pg.date < d) & (pg["min"] > 0) & pg.player_id.isin(feat.player_id)].copy()
            hist_p["pra"] = hist_p[COMBOS["pra"]].sum(axis=1)
            l10 = hist_p.groupby("player_id").tail(10).groupby("player_id")[
                ["pts", "reb", "ast", "fg3m", "pra"]].mean()
            rows = []
            for r in feat.itertuples():
                if r.player_id in l10.index:
                    for s in ["pts", "reb", "ast", "fg3m", "pra"]:
                        if l10.loc[r.player_id, s] < 1:
                            continue
                        rows.append(dict(player=r.player, team=r.team, stat=s,
                                         line=float(np.floor(l10.loc[r.player_id, s]) + 0.5),
                                         over_price=-115, under_price=-115, book="proxy"))
            ctx.prop_lines = pd.DataFrame(rows)
            ctx.notes.append("Demo prop lines are proxies (player's last-10 average rounded to "
                             "x.5 at -115), not real sportsbook lines.")
    return ctx


# ---------------------------------------------------------------------------
def _snapshot(day: date, odds: pd.DataFrame) -> pd.DataFrame:
    """Store every odds pull; the first one of the day is the 'open' we compare against."""
    path = DATA / "snapshots" / f"{day:%Y-%m-%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    odds = odds.assign(pulled_at=datetime.now().isoformat(timespec="seconds"))
    if path.exists():
        allsnap = pd.concat([pd.read_csv(path), odds], ignore_index=True)
    else:
        allsnap = odds
    allsnap.to_csv(path, index=False)
    return allsnap


def live_context(day: date | None = None, with_props: bool = False,
                 action_network: bool = False, history_seasons: int = 2) -> DayContext:
    from .providers import espn, odds_api
    from .providers.public_betting import fetch_action_network, load_csv, to_splits

    day = day or date.today()
    d = pd.Timestamp(day)
    notes = []
    start = espn.season_start(day)
    start = date(start.year - (history_seasons - 1), 10, 1)
    results = espn.results(start, day - timedelta(days=1))
    if results.empty:
        raise RuntimeError("Could not load any results from ESPN")
    from .data.historical import add_rest
    results = add_rest(results)
    ratings = fit_ratings(results, d)
    slate = [g for g in espn.scoreboard(day, live=True) if g["home"] and g["away"]]
    if not slate:
        notes.append("No NBA games found for this date.")

    # --- markets
    cons = pd.DataFrame()
    if odds_api.api_key():
        try:
            raw = odds_api.game_odds()
            snaps = _snapshot(day, raw)
            cons = odds_api.consensus(raw)
            first = snaps[snaps.pulled_at == snaps.pulled_at.min()]
            open_cons = odds_api.consensus(first)
        except Exception as e:  # network / quota
            notes.append(f"Odds API failed ({e}); falling back to ESPN lines at -110.")
    else:
        notes.append("ODDS_API_KEY not set: using ESPN's consensus spread/total at -110, "
                     "no moneylines, no props, no line shopping.")

    games = []
    for s in slate:
        gm = GameMarket(home=s["home"], away=s["away"], start=s["start"],
                        home_rest=rest_days(results, s["home"], d),
                        away_rest=rest_days(results, s["away"], d))
        if not cons.empty:
            c = cons[(cons.home == s["home"]) & (cons.away == s["away"])]
            oc = open_cons[(open_cons.home == s["home"]) & (open_cons.away == s["away"])]

            def pick(df, market, side, col):
                v = df[(df.market == market) & (df.side == side)][col]
                return None if v.empty or pd.isna(v.iloc[0]) else float(v.iloc[0])

            gm.spread = pick(c, "spread", "home", "point")
            gm.spread_home_price = pick(c, "spread", "home", "best_price") or -110
            gm.spread_away_price = pick(c, "spread", "away", "best_price") or -110
            gm.total = pick(c, "total", "over", "point")
            gm.over_price = pick(c, "total", "over", "best_price") or -110
            gm.under_price = pick(c, "total", "under", "best_price") or -110
            gm.home_ml = pick(c, "ml", "home", "best_price")
            gm.away_ml = pick(c, "ml", "away", "best_price")
            gm.open_spread = pick(oc, "spread", "home", "point")
            gm.open_total = pick(oc, "total", "over", "point")
            for _, r in c.iterrows():
                gm.books[(r.market, r.side)] = r.best_book
            gm.books["event_id"] = c.event_id.iloc[0] if len(c) else None
        else:
            gm.spread = s.get("espn_spread")
            gm.total = s.get("espn_total")
        games.append(gm)

    # --- public betting splits
    split_df = load_csv(day)
    if split_df.empty and action_network:
        try:
            split_df = fetch_action_network(day)
            notes.append("Public betting % from Action Network (experimental source).")
        except Exception as e:
            notes.append(f"Action Network fetch failed ({e}).")
    if split_df.empty:
        notes.append("No public betting % loaded: sharp signals use line movement only. Add "
                     f"data/public_betting/{day:%Y-%m-%d}.csv or use the app's Splits tab.")

    ctx = DayContext(day=day, ratings=ratings, games=games, splits=to_splits(split_df),
                     notes=notes, source="live")

    # --- injuries + props
    try:
        ctx.injuries = espn.injuries()
    except Exception as e:
        notes.append(f"Injury report unavailable ({e}).")
    if with_props and slate:
        pg = espn.player_games(results[results.date >= d - pd.Timedelta(days=400)])
        feat = project_players(pg, [(g.home, g.away) for g in games], d, load_prop_params())
        if not ctx.injuries.empty:
            out = ctx.injuries[ctx.injuries.status.str.lower().isin(["out", "doubtful"])]
            out_names = set(out.player.map(norm_name))
            feat = feat[~feat.player.map(norm_name).isin(out_names)]
        ctx.player_features = feat
        if odds_api.api_key():
            lines = []
            for gm in games:
                ev = gm.books.get("event_id")
                if ev:
                    try:
                        lines.append(odds_api.player_props(ev))
                    except Exception as e:
                        notes.append(f"Props failed for {gm.away}@{gm.home}: {e}")
            if lines:
                raw = pd.concat(lines)
                ctx.prop_lines = props_to_lines(raw, feat)
        else:
            notes.append("Props need ODDS_API_KEY for real lines.")
    return ctx


def props_to_lines(raw: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Best over/under price per player+stat at the most common line."""
    if raw.empty:
        return raw
    name_map = {norm_name(p): p for p in feat.player}
    raw = raw.assign(player=raw.player.map(lambda n: name_map.get(norm_name(n))))
    raw = raw.dropna(subset=["player", "line"])
    rows = []
    for (player, stat), g in raw.groupby(["player", "stat"]):
        line = g.line.mode().iloc[0]
        at = g[g.line == line]
        ov, un = at[at.side == "over"], at[at.side == "under"]
        if ov.empty or un.empty:
            continue
        bo, bu = ov.loc[ov.price.idxmax()], un.loc[un.price.idxmax()]
        rows.append(dict(player=player, stat=stat, line=line, over_price=bo.price,
                         under_price=bu.price, book=f"{bo.book}/{bu.book}"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def write_report(res: DayResult, out_dir: Path = REPORTS) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    day = res.ctx.day
    lines = [f"# EdgeSim plays for {day} ({res.ctx.source})", ""]
    for n in res.ctx.notes:
        lines.append(f"> {n}")
    lines.append("")
    lines.append("## Game plays")
    if res.plays.empty:
        lines.append("No game plays cleared the bar today. That is normal: the backtest shows "
                     "NBA sides and totals are very efficient.")
    else:
        cols = ["tier", "game", "pick", "price", "book", "blended_prob", "market_prob", "ev",
                "kelly", "sharp_score", "sharp_reasons"]
        lines.append(_md(res.plays[cols]))
    lines += ["", "## Simulations", _md(res.sims.drop(columns=["home", "away"]))]
    if not res.props.empty:
        lines += ["", "## Prop plays",
                  _md(res.props[res.props.play].head(25)[
                      ["player", "stat", "pick", "price", "book", "projection", "blended_prob",
                       "market_prob", "ev"]]) if res.props.play.any() else "No prop plays today."]
    path = out_dir / f"{day}_{res.ctx.source}.md"
    path.write_text("\n".join(lines))
    res.plays.to_csv(out_dir / f"{day}_{res.ctx.source}_plays.csv", index=False)
    if not res.props.empty:
        res.props.to_csv(out_dir / f"{day}_{res.ctx.source}_props.csv", index=False)
    return path


def _md(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}")
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
