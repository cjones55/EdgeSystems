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
        if ctx.sport == "nfl":
            from .nfl_props import price_props
            props = price_props(ctx.prop_lines, ctx.player_features, cfg=cfg)
        else:
            props = prop_plays(ctx.prop_lines, ctx.player_features, pace, mfac, prop_params, cfg)
    return DayResult(ctx, plays, cands, sims, props)


# ---------------------------------------------------------------------------
def fit_ratings(results: pd.DataFrame, until: pd.Timestamp, sport: str = "nba") -> TeamRatings:
    from .sports import get, rating_points
    sp = get(sport)
    r = TeamRatings(sp.rating)
    for g in results[results.date < until].itertuples():
        hp, ap = rating_points(g, sp.market_weight)
        r.update(g.home, g.away, hp, ap, getattr(g, "home_rest", 2),
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
    if with_props and sport == "nfl":
        from . import nfl_props as N
        try:
            pg = N.load_player_games()
            feat = N.project_upcoming(pg, [(g.home, g.away, g.spread, g.total) for g in games], d)
            ctx.player_features = feat
            ctx.prop_lines = N.proxy_lines(feat, pg[pg.date < d])
            ctx.notes.append("Demo NFL prop lines are proxies (player's last-8 median rounded to "
                             "x.5 at -115; anytime TD from his recent scoring rate), not real sportsbook lines. "
                                 "Demo mode also has no injury news.")
        except FileNotFoundError as e:
            ctx.notes.append(str(e))
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
            frames.append(hist[["date", "season", "home", "away", "home_pts", "away_pts", "neutral",
                                "close_spread", "close_total"]])
            start = max(start, (hist.date.max() + pd.Timedelta(days=1)).date())
        except FileNotFoundError:
            pass
    if start <= day - timedelta(days=1):
        espn_res = espn.results(start, day - timedelta(days=1), sport=sport)
        if not espn_res.empty:
            # ESPN keeps the closing consensus line on finished games
            espn_res = espn_res.assign(close_spread=espn_res.get("espn_spread"),
                                       close_total=espn_res.get("espn_total"))
            frames.append(espn_res[["date", "season", "home", "away", "home_pts", "away_pts", "neutral",
                                    "close_spread", "close_total"]])
    if not frames:
        raise RuntimeError(f"Could not load any {sport.upper()} results")
    res = pd.concat(frames, ignore_index=True).dropna(subset=["home", "away"])
    if sport == "ncaab":
        # ESPN lists D1 games against non-D1 schools (Apprentice School, Ecclesia...). The
        # archive and the backtest only have D1 vs D1, and a 50-point win over a team rated
        # "average new team" makes the D1 side look like a juggernaut. Keep D1 vs D1 only.
        d1 = res.home.astype(str).str.isdigit() & res.away.astype(str).str.isdigit()
        res = res[d1]
    res = res.sort_values("date").reset_index(drop=True)
    return add_rest(res)


def data_notes(results: pd.DataFrame, slate: list[dict], day: date, sport: str) -> list[str]:
    """How current the ratings are, and which teams on the slate have no history."""
    from .providers.espn import season_label
    from .resolve import display
    if results.empty:
        return []
    season = season_label(day, sport)
    cur = results[results.season.astype(str) == str(season)]
    start = season - 1 if sport == "ncaab" else season      # college hoops is named for its end year
    name = str(start) if sport in ("nfl", "ncaaf") else f"{start}-{str(start + 1)[-2:]}"
    out = [f"Ratings use results through {results.date.max():%b %d, %Y}: "
           f"{len(cur)} games so far this season ({name})."]
    known = set(results.home) | set(results.away)
    new = sorted({display(t, sport) for g in slate for t in (g["home"], g["away"]) if t not in known})
    if new:
        out.append(f"No past games for {len(new)} team(s), so they start at an average new-team "
                   f"rating: {', '.join(new[:8])}{' ...' if len(new) > 8 else ''}.")
    return out


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
    if sport == "ncaab":
        n = len(slate)
        slate = [g for g in slate if str(g["home"]).isdigit() and str(g["away"]).isdigit()]
        if len(slate) < n:
            notes.append(f"Skipped {n - len(slate)} game(s) against non-Division I schools.")
    if not slate:
        notes.append(f"No {sp.label} games found for this date.")
    notes.extend(data_notes(results, slate, day, sport))

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
        notes.append("ODDS_API_KEY not set: using ESPN's consensus lines (spread, total and "
                     "moneyline when ESPN lists them). No props or FanDuel/DraftKings shopping.")
    espn_open = {}
    if cons.empty and slate:
        # snapshot ESPN's lines too, so line movement (and steam alerts) work without a key
        rows = [dict(home=s["home"], away=s["away"], market=m, side=side,
                     point=None if k is None else sign * s[k],
                     price=s.get(pk) or -110, book="espn")
                for s in slate for m, side, k, sign, pk in (
                    ("spread", "home", "espn_spread", 1, "espn_spread_home_price"),
                    ("spread", "away", "espn_spread", -1, "espn_spread_away_price"),
                    ("total", "over", "espn_total", 1, "espn_over_price"),
                    ("total", "under", "espn_total", 1, "espn_under_price"),
                    ("ml", "home", None, 1, "espn_home_ml"), ("ml", "away", None, 1, "espn_away_ml"))
                if (s.get(k) is not None if k else s.get(pk) is not None)]
        if rows:
            snaps = _snapshot(day, pd.DataFrame(rows), sport)
            snaps = snaps[(snaps.get("book", pd.Series("", index=snaps.index)) == "espn")
                          & snaps.side.isin(["home", "over"])]
            first = snaps.sort_values("pulled_at").groupby(["home", "away", "market"]).first()
            espn_open = {k: (r.price if k[2] == "ml" else r.point) for k, r in first.iterrows()}

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
            gm.spread_home_price = s.get("espn_spread_home_price") or -110
            gm.spread_away_price = s.get("espn_spread_away_price") or -110
            gm.over_price = s.get("espn_over_price") or -110
            gm.under_price = s.get("espn_under_price") or -110
            gm.home_ml, gm.away_ml = s.get("espn_home_ml"), s.get("espn_away_ml")
            k3 = (s["home"], s["away"])
            # ESPN's own opening line when it lists one, else our first snapshot of the day
            def first_of(*vals):
                return next((v for v in vals if v is not None), None)
            gm.open_spread = first_of(s.get("espn_open_spread"), espn_open.get((*k3, "spread")), gm.spread)
            gm.open_total = first_of(s.get("espn_open_total"), espn_open.get((*k3, "total")), gm.total)
            gm.open_home_ml = first_of(s.get("espn_open_home_ml"), espn_open.get((*k3, "ml")), gm.home_ml)
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
    elif sport == "nfl":
        # projections are free (box scores + props explorer); lines only when asked
        try:
            _nfl_props(ctx, day, games, odds_api, espn, fetch_lines=with_props)
        except Exception as e:
            notes.append(f"NFL props skipped ({type(e).__name__}: {e}).")
    elif with_props:
        notes.append(f"Player props cover the NBA and NFL; skipped for {sp.label}.")
    if with_props and sp.has_props and slate:
        try:
            _nba_props(ctx, day, d, results, games, odds_api, espn)
        except Exception as e:  # props are optional: never lose the game card over them
            notes.append(f"NBA props skipped ({type(e).__name__}: {e}).")
    return ctx


def _nba_props(ctx: DayContext, day: date, d: pd.Timestamp, results: pd.DataFrame,
               games: list, odds_api, espn) -> None:
    sport, notes = "nba", ctx.notes
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


def _nfl_props(ctx: DayContext, day: date, games: list, odds_api, espn,
               fetch_lines: bool = True) -> None:
    """Project NFL players for this slate and price their props."""
    from . import nfl_props as N
    notes = ctx.notes
    try:
        pg = N.load_player_games()
    except FileNotFoundError as e:
        notes.append(str(e))
        return
    out_names = set()
    try:
        inj = espn.injuries("nfl")
        bad = inj[inj.status.fillna("").str.lower().str.contains(
            r"\bout\b|doubtful|injured reserve|\bir\b|suspen|physically unable", regex=True)]
        out_names = set(bad.player.str.lower())
        ctx.injuries = inj
    except Exception as e:
        notes.append(f"NFL injury report unavailable ({e}); injured players may be projected.")
    # starting QB: adjust each team's expected points for who is expected to start
    try:
        from . import nfl_qb as Q
        st = Q.starters(pg)
        vals = Q.QBValues(st, Q.replacement_level(st))
        for g in games:
            parts = []
            for team, attr in ((g.home, "home_adj"), (g.away, "away_adj")):
                qs = Q.team_qb_status(pg, team, pd.Timestamp(day), out_names, st, vals)
                if qs and abs(qs.adjustment) >= 0.5:
                    setattr(g, attr, qs.adjustment)
                    parts.append(qs.note())
            if parts:
                g.adj_note = "; ".join(parts)
                notes.append(g.adj_note)
    except Exception as e:
        notes.append(f"QB adjustment skipped ({type(e).__name__}: {e}).")
    feat = N.project_upcoming(pg, [(g.home, g.away, g.spread, g.total) for g in games],
                              pd.Timestamp(day), out_players=out_names)
    ctx.player_features = feat
    stale = (pd.Timestamp(day) - pg.date.max()).days
    if stale > 9:
        notes.append(f"NFL player stats end {pg.date.max().date()}: run "
                     "`python scripts/fetch_data.py nfl --refresh-current` for the latest week.")
    if not fetch_lines:
        return
    if not odds_api.api_key():
        notes.append("NFL props need ODDS_API_KEY for real lines; projections are on the Props tab.")
        return
    raw = []
    for gm in games:
        ev = gm.books.get("event_id")
        if ev:
            try:
                raw.append(odds_api.player_props(ev, sport="nfl"))
            except Exception as e:
                notes.append(f"Props failed for {gm.away}@{gm.home}: {e}")
    if raw:
        ctx.prop_lines = N.lines_from_odds(pd.concat(raw), odds_api.my_books())


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
