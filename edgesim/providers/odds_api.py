"""The Odds API (https://the-odds-api.com) - game lines and player props.

Needs an API key in the ODDS_API_KEY environment variable (or .env file).
The free plan gives 500 credits/month: one game-lines pull costs 3 credits
(3 markets x 1 region); player props cost 1 credit per market per game.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from ..resolve import resolve

ROOT_URL = "https://api.the-odds-api.com/v4/sports"
BASE = f"{ROOT_URL}/basketball_nba"


def base(sport: str = "nba") -> str:
    from ..sports import get
    return f"{ROOT_URL}/{get(sport).odds_key}"


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


def parse_game_odds(data: list, sport: str = "nba") -> pd.DataFrame:
    """One row per (game, book, market, side)."""
    rows = []
    for ev in data:
        home, away = resolve(ev["home_team"], sport), resolve(ev["away_team"], sport)
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    name = o["name"]
                    if mk["key"] in ("h2h", "spreads"):
                        side = "home" if name == ev["home_team"] else "away"
                    else:
                        side = name.lower()           # over / under
                    rows.append(dict(event_id=ev["id"], start=ev.get("commence_time"),
                                     home=home, away=away, book=bk["key"],
                                     market={"h2h": "ml", "spreads": "spread",
                                             "totals": "total"}[mk["key"]],
                                     side=side, point=o.get("point"), price=o["price"]))
    return pd.DataFrame(rows)


def game_odds(regions: str = "us", sport: str = "nba") -> pd.DataFrame:
    data = _get(f"{base(sport)}/odds", {"regions": regions, "markets": "h2h,spreads,totals",
                                        "oddsFormat": "american"})
    return parse_game_odds(data, sport)


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
    """Most common line per game/market/side, and the best price offered AT that line
    (comparing prices across different lines would be meaningless)."""
    if odds.empty:
        return odds
    keys = ["event_id", "home", "away", "market", "side"]
    odds = odds.assign(_dec=odds.price.map(lambda p: p / 100 + 1 if p > 0 else 100 / -p + 1),
                       _pt=odds.point.fillna(0.0))
    mode = odds.groupby(keys)._pt.agg(lambda x: x.mode().iloc[0]).rename("_mode").reset_index()
    at = odds.merge(mode, on=keys)
    at = at[at._pt == at._mode]
    g = at.groupby(keys)
    best = at.loc[g._dec.idxmax()]
    out = best[keys + ["price", "book"]].rename(columns={"price": "best_price", "book": "best_book"})
    out = out.merge(mode.rename(columns={"_mode": "point"}), on=keys)
    out = out.merge(odds.groupby(keys).book.nunique().rename("n_books").reset_index(), on=keys)
    out.loc[out.market == "ml", "point"] = None
    return out.reset_index(drop=True)
