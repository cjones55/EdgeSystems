"""The Odds API (https://the-odds-api.com) - game lines and player props.

Needs an API key in the ODDS_API_KEY environment variable (or .env file).
The free plan gives 500 credits/month: one game-lines pull costs 3 credits
(3 markets x 1 region); player props cost 1 credit per market per game.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from ..teams import to_code

BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
PROP_MARKETS = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
                "player_threes": "fg3m", "player_points_rebounds_assists": "pra"}


def api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY")


def _get(url: str, params: dict) -> list | dict:
    key = api_key()
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set")
    r = requests.get(url, params={"apiKey": key, **params}, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_game_odds(data: list) -> pd.DataFrame:
    """One row per (game, book, market, side)."""
    rows = []
    for ev in data:
        home, away = to_code(ev["home_team"]), to_code(ev["away_team"])
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    name = o["name"]
                    if mk["key"] in ("h2h", "spreads"):
                        side = "home" if to_code(name) == home else "away"
                    else:
                        side = name.lower()           # over / under
                    rows.append(dict(event_id=ev["id"], start=ev.get("commence_time"),
                                     home=home, away=away, book=bk["key"],
                                     market={"h2h": "ml", "spreads": "spread",
                                             "totals": "total"}[mk["key"]],
                                     side=side, point=o.get("point"), price=o["price"]))
    return pd.DataFrame(rows)


def game_odds(regions: str = "us") -> pd.DataFrame:
    data = _get(f"{BASE}/odds", {"regions": regions, "markets": "h2h,spreads,totals",
                                 "oddsFormat": "american"})
    return parse_game_odds(data)


def parse_props(data: dict) -> pd.DataFrame:
    rows = []
    for bk in data.get("bookmakers", []):
        for mk in bk.get("markets", []):
            stat = PROP_MARKETS.get(mk["key"])
            if not stat:
                continue
            for o in mk.get("outcomes", []):
                rows.append(dict(event_id=data.get("id"), book=bk["key"], stat=stat,
                                 player=o.get("description"), side=o["name"].lower(),
                                 line=o.get("point"), price=o["price"]))
    return pd.DataFrame(rows)


def player_props(event_id: str, markets=tuple(PROP_MARKETS), regions: str = "us") -> pd.DataFrame:
    data = _get(f"{BASE}/events/{event_id}/odds",
                {"regions": regions, "markets": ",".join(markets), "oddsFormat": "american"})
    return parse_props(data)


def consensus(odds: pd.DataFrame) -> pd.DataFrame:
    """Median line per game/market plus the best price available on each side."""
    if odds.empty:
        return odds
    g = odds.groupby(["event_id", "home", "away", "market", "side"])
    out = g.agg(point=("point", "median"), best_price=("price", "max"),
                n_books=("book", "nunique")).reset_index()
    best_book = odds.loc[g.price.idxmax(), ["event_id", "market", "side", "book"]]
    return out.merge(best_book.rename(columns={"book": "best_book"}),
                     on=["event_id", "market", "side"])
