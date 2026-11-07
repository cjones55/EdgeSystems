"""College football and college basketball history.

NCAAF: CollegeFootballData.com exports (github.com/matthewrichey3317/cfb-data, MIT),
       2015-2026 FBS games, closing spread/total, opening lines and moneylines
       from several books.
NCAAB: results for every D1 game from the Kaggle "March Machine Learning
       Mania" files, joined to near-closing odds snapshots from many US books
       (The Odds API, 2020-21 .. 2022-23 seasons, github.com/JDSB123/ncaam-historical-data).
College teams have no fixed short codes, so the team key is the school name
(football) or the Kaggle TeamID (basketball); ``college_resolver`` maps the
names ESPN and The Odds API use onto those keys.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .historical import add_rest

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def _std(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    for c, v in (("spread_home_price", -110.0), ("spread_away_price", -110.0),
                 ("over_price", -110.0), ("under_price", -110.0), ("open_home_ml", np.nan),
                 ("open_away_ml", np.nan), ("neutral", False)):
        if c not in df.columns:
            df[c] = v
    df["sport"] = sport
    df = df.sort_values(["date", "home"]).reset_index(drop=True)
    return add_rest(df)


# ---------------------------------------------------------------- football
def load_ncaaf(raw: Path = RAW / "ncaaf") -> pd.DataFrame:
    g = pd.concat(pd.read_csv(p) for p in sorted(raw.glob("games_compact_*.csv")))
    g = g[g.completed.astype(bool)].dropna(subset=["home_points", "away_points"])
    lines = pd.concat(pd.read_csv(p, usecols=["gameId", "provider", "spread", "spreadOpen",
                                                "overUnderOpen", "homeMoneyline", "awayMoneyline"])
                      for p in sorted(raw.glob("lines_*.csv")))
    lines = lines[lines.provider != "teamrankings"]   # model numbers, not a book
    agg = lines.groupby("gameId").agg(open_spread=("spreadOpen", "median"),
                                      open_total=("overUnderOpen", "median"),
                                      home_ml=("homeMoneyline", "median"),
                                      away_ml=("awayMoneyline", "median")).reset_index()
    g = g.merge(agg, left_on="game_id", right_on="gameId", how="left")
    df = pd.DataFrame(dict(
        date=pd.to_datetime(g.kickoff_utc).dt.tz_localize(None).dt.normalize()
        - pd.to_timedelta((pd.to_datetime(g.kickoff_utc).dt.hour < 9).astype(int), "D"),
        season=g.season.astype(int), home=g.home_team, away=g.away_team,
        home_pts=g.home_points.astype(int), away_pts=g.away_points.astype(int),
        neutral=g.neutral_site.astype(bool), close_spread=g.spread_close,
        close_total=g.ou_close, open_spread=g.open_spread, open_total=g.open_total,
        home_ml=g.home_ml.where(g.home_ml.abs() >= 100), away_ml=g.away_ml.where(g.away_ml.abs() >= 100),
        home_conf=g.home_conference, away_conf=g.away_conference))
    df.loc[df.close_spread.abs() > 60, "close_spread"] = np.nan
    df.loc[~df.close_total.between(25, 95), "close_total"] = np.nan
    return _std(df, "ncaaf")


# ---------------------------------------------------------------- basketball
def _norm(s: str) -> str:
    s = str(s).lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_EXTRA_NCAAB = {
    "fau": "florida atlantic", "southeast missouri st": "southeast missouri state",
    "louisiana monroe": "ul monroe", "cal st bakersfield": "cal state bakersfield",
    "texas a and m corpus christi": "texas a and m corpus christi", "queens": "queens nc",
    "mississippi valley st": "mississippi valley state", "arkansas pine bluff": "arkansas pine bluff",
    "bethune cookman": "bethune cookman", "texas a and m commerce": "east texas a and m",
}


@lru_cache(maxsize=1)
def ncaab_spellings(raw: Path = RAW / "ncaab") -> dict[str, int]:
    sp = pd.read_csv(raw / "MTeamSpellings.csv", encoding="latin-1")
    out = {_norm(n): int(t) for n, t in zip(sp.TeamNameSpelling, sp.TeamID)}
    teams = pd.read_csv(raw / "MTeams.csv")
    for n, t in zip(teams.TeamName, teams.TeamID):
        out.setdefault(_norm(n), int(t))
    return out


def ncaab_names(raw: Path = RAW / "ncaab") -> dict[str, str]:
    t = pd.read_csv(raw / "MTeams.csv")
    return {str(i): n for i, n in zip(t.TeamID, t.TeamName)}


def resolve_ncaab(name: str) -> str | None:
    """Kaggle TeamID (as str) for an ESPN / Odds API / Torvik style name.
    Tries the full name, then drops trailing words (mascots) one at a time."""
    m = ncaab_spellings()
    words = _norm(name).split()
    for k in range(len(words), 0, -1):
        cand = " ".join(words[:k])
        for c in (cand, _EXTRA_NCAAB.get(cand, ""), cand.replace(" st", " state"),
                  cand.replace(" state", " st")):
            if c and c in m:
                return str(m[c])
    return None


def load_ncaab(raw: Path = RAW / "ncaab", first_season: int = 2015) -> pd.DataFrame:
    seasons = pd.read_csv(raw / "MSeasons.csv")
    seasons["DayZero"] = pd.to_datetime(seasons.DayZero)
    res = pd.concat(pd.read_csv(raw / f) for f in (
        "MRegularSeasonCompactResults.csv", "MNCAATourneyCompactResults.csv",
        "MSecondaryTourneyCompactResults.csv") if (raw / f).exists())
    res = res[res.Season >= first_season].merge(seasons[["Season", "DayZero"]], on="Season")
    res["date"] = res.DayZero + pd.to_timedelta(res.DayNum, "D")
    # neutral sites: nominal home is the lower TeamID, so it carries no information about the winner
    w_home = (res.WLoc == "H") | ((res.WLoc == "N") & (res.WTeamID < res.LTeamID))
    df = pd.DataFrame(dict(
        date=res.date, season=res.Season.astype(int),
        home=np.where(w_home, res.WTeamID, res.LTeamID).astype(str),
        away=np.where(w_home, res.LTeamID, res.WTeamID).astype(str),
        home_pts=np.where(w_home, res.WScore, res.LScore),
        away_pts=np.where(w_home, res.LScore, res.WScore),
        neutral=(res.WLoc == "N").values))

    odds_path = raw / "odds_consolidated_canonical.csv"
    if odds_path.exists():
        o = pd.read_csv(odds_path, low_memory=False, usecols=[
            "event_id", "commence_time", "home_team", "away_team", "home_team_canonical",
            "away_team_canonical", "bookmaker", "spread", "spread_home_price",
            "spread_away_price", "total", "total_over_price", "total_under_price",
            "moneyline_home_price", "moneyline_away_price"])
        o = o[o.bookmaker != "draftkings"].dropna(subset=["spread"]) if (o.bookmaker != "draftkings").any() else o
        ev = o.groupby("event_id").agg(
            ct=("commence_time", "first"), h=("home_team_canonical", "first"),
            a=("away_team_canonical", "first"), close_spread=("spread", "median"),
            close_total=("total", "median"), home_ml=("moneyline_home_price", "median"),
            away_ml=("moneyline_away_price", "median"),
            spread_home_price=("spread_home_price", "median"),
            spread_away_price=("spread_away_price", "median"),
            over_price=("total_over_price", "median"),
            under_price=("total_under_price", "median")).reset_index()
        # tip-off times are UTC; US evening games fall on the previous date
        ev["date"] = (pd.to_datetime(ev.ct).dt.tz_localize(None) - pd.Timedelta(hours=7)).dt.normalize()
        ev["hid"] = ev.h.map(resolve_ncaab)
        ev["aid"] = ev.a.map(resolve_ncaab)
        ev = ev.dropna(subset=["hid", "aid"])
        ev = ev.drop_duplicates(["date", "hid", "aid"])
        key = ["date", "home", "away"]
        same = df.merge(ev.rename(columns={"hid": "home", "aid": "away"}), on=key, how="left")
        # results may list the teams the other way round (neutral sites)
        flip = ev.rename(columns={"hid": "away", "aid": "home"}).copy()
        flip["close_spread"] = -flip.close_spread
        flip[["home_ml", "away_ml"]] = flip[["away_ml", "home_ml"]].values
        flip[["spread_home_price", "spread_away_price"]] = flip[["spread_away_price", "spread_home_price"]].values
        cols = ["close_spread", "close_total", "home_ml", "away_ml", "spread_home_price",
                "spread_away_price", "over_price", "under_price"]
        other = df.merge(flip[key + cols], on=key, how="left")
        for c in cols:
            same[c] = same[c].fillna(other[c])
        df = same[list(df.columns) + cols]
    for c in ("open_spread", "open_total"):
        df[c] = np.nan
    for c in ("spread_home_price", "spread_away_price", "over_price", "under_price"):
        if c in df.columns:
            df[c] = df[c].where(df[c].abs() >= 100, -110.0).fillna(-110.0)
    for c in ("home_ml", "away_ml"):
        if c in df.columns:
            df[c] = df[c].where(df[c].abs() >= 100)
    return _std(df, "ncaab")
