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
    sport: str = "nba"


@dataclass
class DayResult:
    ctx: DayContext
    plays: pd.DataFrame
    candidates: pd.DataFrame
    sims: pd.DataFrame
    props: pd.DataFrame


def run_day(ctx: DayContext, cfg: PlayConfig | None = None,
            prop_params: PropParams | None = None) -> DayResult:
    cfg = cfg or PlayConfig.for_sport(ctx.sport)
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
def fit_ratings(results: pd.DataFrame, until: pd.Timestamp, sport: str = "nba") -> TeamRatings:
    from .sports import get
    r = TeamRatings(get(sport).rating)
    for g in results[results.date < until].itertuples():
        r.update(g.home, g.away, g.home_pts, g.away_pts, getattr(g, "home_rest", 2),
                 getattr(g, "away_rest", 2), season=g.season,
                 neutral=bool(getattr(g, "neutral", False)))
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
def demo_context(day: str | date, with_props: bool = True, top_players: int = 7,
                 sport: str = "nba") -> DayContext:
    from .providers.public_betting import load_csv, to_splits
    from .resolve import display
    from .sports import get, load_history

    sp = get(sport)
    d = pd.Timestamp(day)
    hist = load_history(sport)
    if not (hist.date == d).any():
        dates = hist.dropna(subset=["close_total"]).date
        raise ValueError(f"No archived {sp.label} games on {d.date()} (archive: {sp.archive_note}; "
                         f"try {dates.iloc[len(dates) // 2].date()})")
    ratings = fit_ratings(hist, d, sport)
    todays = hist[(hist.date == d) & hist[["close_spread", "close_total", "home_ml"]].notna().any(axis=1)]

    def f(v):
        return None if pd.isna(v) else float(v)

    games = [GameMarket(home=g.home, away=g.away, home_rest=g.home_rest, away_rest=g.away_rest,
                        start=str(d.date()), spread=f(g.close_spread), total=f(g.close_total),
                        spread_home_price=f(g.spread_home_price) or -110,
                        spread_away_price=f(g.spread_away_price) or -110,
                        over_price=f(g.over_price) or -110, under_price=f(g.under_price) or -110,
                        home_ml=f(g.home_ml), away_ml=f(g.away_ml),
                        open_spread=f(g.open_spread), open_total=f(g.open_total),
                        open_home_ml=f(g.open_home_ml), neutral=bool(g.neutral),
                        home_name=display(g.home, sport), away_name=display(g.away, sport))
             for g in todays.itertuples()]
    ctx = DayContext(day=d.date(), ratings=ratings, games=games, source="demo", sport=sport,
                     splits=to_splits(load_csv(d.date(), sport)))
    ctx.notes.append("Demo mode: lines are the archived numbers (the 'current' line is the "
                     "close). Actual results are known, so this is for inspecting the system, "
                     "not for betting.")
    if ctx.splits:
        ctx.notes.append("Public betting % in demo mode come from a sample CSV and are illustrative.")
    if with_props and sp.has_props:
        try:
            from .data.boxscores import load_player_games
            pg = load_player_games()
        except FileNotFoundError:
            pg = None
        if pg is not None and (pg.date == d).any():
            feat = project_players(pg, [(g.home, g.away) for g in games], d, load_prop_params())
            feat = feat.sort_values("proj_min", ascending=False).groupby("team").head(top_players)
            ctx.player_features = feat
            hist_p = pg[(pg.date < d) & (pg["min"] > 0) & pg.player_id.isin(feat.player_id)].copy()
            hist_p["pra"] = hist_p[COMBOS["pra"]].sum(axis=1)
            l10 = hist_p.groupby("player_id").tail(10).groupby("player_id")[
                ["pts", "reb", "ast", "fg3m", "pra"]].mean()
            rows = []
            for r in feat.itertuples():
                if r.player_id in l10.index:
                    for st in ["pts", "reb", "ast", "fg3m", "pra"]:
                        if l10.loc[r.player_id, st] < 1:
                            continue
                        rows.append(dict(player=r.player, team=r.team, stat=st,
                                         line=float(np.floor(l10.loc[r.player_id, st]) + 0.5),
                                         over_price=-115, under_price=-115, book="proxy"))
            ctx.prop_lines = pd.DataFrame(rows)
            ctx.notes.append("Demo prop lines are proxies (player's last-10 average rounded to "
                             "x.5 at -115), not real sportsbook lines.")
    return ctx


# ---------------------------------------------------------------------------
def snapshot_path(day: date, sport: str = "nba") -> Path:
    return DATA / "snapshots" / f"{sport}_{day:%Y-%m-%d}.csv"


def _snapshot(day: date, odds: pd.DataFrame, sport: str = "nba") -> pd.DataFrame:
    """Store every odds pull; the first one of the day is the 'open' we compare against."""
    path = snapshot_path(day, sport)
    path.parent.mkdir(parents=True, exist_ok=True)
    odds = odds.assign(pulled_at=datetime.now().isoformat(timespec="seconds"))
    if path.exists():
        allsnap = pd.concat([pd.read_csv(path), odds], ignore_index=True)
    else:
        allsnap = odds
    allsnap.to_csv(path, index=False)
    return allsnap


def save_split_history(day: date, splits: pd.DataFrame, sport: str = "nba") -> Path | None:
    """Keep every public-betting pull, so ticket/money signals can be graded later
    (there is no free history of them anywhere else)."""
    if splits.empty:
        return None
    path = DATA / "splits_history" / f"{sport}_{day:%Y-%m-%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = splits.assign(pulled_at=datetime.now().isoformat(timespec="seconds"))
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def live_results(day: date, sport: str = "nba", history_seasons: int = 2) -> pd.DataFrame:
    """Completed games to rate teams on: ESPN for recent seasons. College sports
    start from the historical archive (same team keys) and add ESPN games after it."""
    from .data.historical import add_rest
    from .providers import espn
    from .sports import load_history

    start = espn.season_start(day, sport)
    start = date(start.year - (history_seasons - 1), start.month, 1)
    frames = []
    if get_sport(sport).college:
        try:
            hist = load_history(sport)
            hist = hist[hist.date < pd.Timestamp(day)]
            frames.append(hist[["date", "season", "home", "away", "home_pts", "away_pts", "neutral"]])
            start = max(start, (hist.date.max() + pd.Timedelta(days=1)).date())
        except FileNotFoundError:
            pass
    if start <= day - timedelta(days=1):
        espn_res = espn.results(start, day - timedelta(days=1), sport=sport)
        if not espn_res.empty:
            frames.append(espn_res[["date", "season", "home", "away", "home_pts", "away_pts", "neutral"]])
    if not frames:
        raise RuntimeError(f"Could not load any {sport.upper()} results")
    res = pd.concat(frames, ignore_index=True).dropna(subset=["home", "away"])
    res = res.sort_values("date").reset_index(drop=True)
    return add_rest(res)


def get_sport(sport: str):
    from .sports import get
    return get(sport)


def live_context(day: date | None = None, with_props: bool = False,
                 action_network: bool = False, history_seasons: int = 2,
                 sport: str = "nba") -> DayContext:
    from .providers import espn, odds_api
    from .providers.public_betting import fetch_action_network, load_csv, to_splits
    from .resolve import display

    sp = get_sport(sport)
    day = day or date.today()
    d = pd.Timestamp(day)
    notes = []
    results = live_results(day, sport, history_seasons)
    ratings = fit_ratings(results, d, sport)
    slate = [g for g in espn.scoreboard(day, live=True, sport=sport) if g["home"] and g["away"]]
    if not slate:
        notes.append(f"No {sp.label} games found for this date.")

    # --- markets
    cons = open_cons = pd.DataFrame()
    if odds_api.api_key():
        try:
            raw = odds_api.game_odds(sport=sport)
            snaps = _snapshot(day, raw, sport)
            if "book" in snaps.columns:
                snaps = snaps[snaps.book != "espn"]
            cons = odds_api.consensus(raw, odds_api.my_books())
            first = snaps[snaps.pulled_at == snaps.pulled_at.min()]
            open_cons = odds_api.consensus(first, odds_api.my_books())
            if odds_api.my_books():
                notes.append("Prices from your books: " + ", ".join(odds_api.my_books())
                             + " (EDGESIM_BOOKS in .env).")
        except Exception as e:  # network / quota
            notes.append(f"Odds API failed ({e}); falling back to ESPN lines at -110.")
    else:
        notes.append("ODDS_API_KEY not set: using ESPN's consensus spread/total at -110, "
                     "no moneylines, no props, no line shopping.")
    espn_open = {}
    if cons.empty and slate:
        # snapshot ESPN's lines too, so line movement (and steam alerts) work without a key
        rows = [dict(home=s["home"], away=s["away"], market=m, side=side,
                     point=sign * s[k], price=-110, book="espn")
                for s in slate for m, side, k, sign in (
                    ("spread", "home", "espn_spread", 1), ("spread", "away", "espn_spread", -1),
                    ("total", "over", "espn_total", 1), ("total", "under", "espn_total", 1))
                if s.get(k) is not None]
        if rows:
            snaps = _snapshot(day, pd.DataFrame(rows), sport)
            snaps = snaps[(snaps.get("book", pd.Series("", index=snaps.index)) == "espn")
                          & snaps.side.isin(["home", "over"])]
            first = snaps.sort_values("pulled_at").groupby(["home", "away", "market"]).first()
            espn_open = first.point.to_dict()

    def pick(df, market, side, col):
        v = df[(df.market == market) & (df.side == side)][col] if not df.empty else pd.Series(dtype=float)
        return None if v.empty or pd.isna(v.iloc[0]) else float(v.iloc[0])

    games = []
    for s in slate:
        gm = GameMarket(home=s["home"], away=s["away"], start=s["start"], neutral=s["neutral"],
                        home_rest=rest_days(results, s["home"], d),
                        away_rest=rest_days(results, s["away"], d),
                        home_name=s.get("home_name") or display(s["home"], sport),
                        away_name=s.get("away_name") or display(s["away"], sport))
        c = cons[(cons.home == s["home"]) & (cons.away == s["away"])] if not cons.empty else cons
        if not c.empty:
            oc = open_cons[(open_cons.home == s["home"]) & (open_cons.away == s["away"])]
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
            gm.open_home_ml = pick(oc, "ml", "home", "best_price")
            for _, r in c.iterrows():
                gm.books[(r.market, r.side)] = r.best_book
            gm.books["event_id"] = c.event_id.iloc[0]
        else:
            if not cons.empty:
                notes.append(f"No odds matched for {gm.away_name} at {gm.home_name}; using ESPN line.")
            gm.spread = s.get("espn_spread")
            gm.total = s.get("espn_total")
            gm.open_spread = espn_open.get((s["home"], s["away"], "spread"), gm.spread)
            gm.open_total = espn_open.get((s["home"], s["away"], "total"), gm.total)
        games.append(gm)

    # --- public betting splits
    split_df = load_csv(day, sport)
    if split_df.empty and action_network:
        try:
            split_df = fetch_action_network(day, sport)
            notes.append("Public betting % from Action Network (experimental source).")
            save_split_history(day, split_df, sport)
        except Exception as e:
            notes.append(f"Action Network fetch failed ({e}).")
    if split_df.empty:
        notes.append("No public betting % loaded: sharp signals use line movement only. Add "
                     f"data/public_betting/{sport}/{day:%Y-%m-%d}.csv or use the app's "
                     "Public betting % tab.")

    ctx = DayContext(day=day, ratings=ratings, games=games, splits=to_splits(split_df),
                     notes=notes, source="live", sport=sport)

    # --- injuries + props (NBA)
    if sp.has_props:
        try:
            ctx.injuries = espn.injuries(sport)
        except Exception as e:
            notes.append(f"Injury report unavailable ({e}).")
    elif with_props:
        notes.append(f"Player props are NBA-only for now; skipped for {sp.label}.")
    if with_props and sp.has_props and slate:
        pg = espn.player_games(pd.DataFrame([dict(event_id=e, date=dt) for e, dt in _event_ids(results, d, sport)]))
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
                ctx.prop_lines = props_to_lines(pd.concat(lines), feat)
        else:
            notes.append("Props need ODDS_API_KEY for real lines.")
    return ctx


def _event_ids(results: pd.DataFrame, d: pd.Timestamp, sport: str) -> list:
    """ESPN event ids for recent games (box scores for player projections)."""
    from .providers import espn
    out = []
    day = (d - pd.Timedelta(days=400)).date()
    end = (d - pd.Timedelta(days=1)).date()
    while day <= end:
        for g in espn.scoreboard(day, sport=sport):
            if g["completed"]:
                out.append((g["event_id"], pd.Timestamp(day)))
        day += timedelta(days=1)
    return out


def props_to_lines(raw: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Best over/under price per player+stat at the most common line."""
    if raw.empty:
        return raw
    name_map = {norm_name(p): p for p in feat.player}
    raw = raw.assign(player=raw.player.map(lambda n: name_map.get(norm_name(n))))
    raw = raw.dropna(subset=["player", "line"])
    rows = []
    from .providers.odds_api import my_books
    books = my_books()
    for (player, stat), g in raw.groupby(["player", "stat"]):
        if books and g.book.isin(books).any():
            g = g[g.book.isin(books)]      # only lines you can bet
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
    sp = get_sport(res.ctx.sport)
    lines = [f"# EdgeSim {sp.label} plays for {day} ({res.ctx.source})", ""]
    for n in res.ctx.notes:
        lines.append(f"> {n}")
    lines.append("")
    lines.append("## Game plays")
    if res.plays.empty:
        lines.append("No game plays cleared the bar today. That is normal: the backtest shows "
                     "NBA sides and totals are very efficient.")
    else:
        cols = ["units", "tier", "game", "pick", "price", "book", "blended_prob", "market_prob", "ev",
                "kelly", "sharp_score", "sharp_reasons"]
        lines.append(_md(res.plays[cols]))
    lines += ["", "## Simulations", _md(res.sims.drop(columns=["home", "away"], errors="ignore"))]
    if not res.props.empty:
        lines += ["", "## Prop plays",
                  _md(res.props[res.props.play].head(25)[
                      ["units", "player", "stat", "pick", "price", "book", "projection", "blended_prob",
                       "market_prob", "ev"]]) if res.props.play.any() else "No prop plays today."]
    stem = f"{day}_{res.ctx.sport}_{res.ctx.source}"
    path = out_dir / f"{stem}.md"
    path.write_text("\n".join(lines))
    res.plays.to_csv(out_dir / f"{stem}_plays.csv", index=False)
    if not res.props.empty:
        res.props.to_csv(out_dir / f"{stem}_props.csv", index=False)
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
