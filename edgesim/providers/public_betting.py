"""Public betting splits (% of bets and % of money on each side).

There is no free, official source for these numbers, so two inputs are
supported:
  1. CSV (always works): data/public_betting/<sport>/YYYY-MM-DD.csv with columns
       home,away,market,open_line,current_line,tickets_pct_a,money_pct_a
     market is spread|total|ml; side A is the home team (spread/ml) or the
     over (total). Copy the numbers from any splits site you trust.
  2. Action Network's public web endpoint (experimental): unofficial and
     undocumented; it may change or block requests at any time.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ..sharp import MarketSplit
from ..resolve import resolve

DIR = Path(__file__).resolve().parents[2] / "data" / "public_betting"
COLUMNS = ["home", "away", "market", "open_line", "current_line", "tickets_pct_a", "money_pct_a"]


def csv_path(day: date, sport: str = "nba") -> Path:
    return DIR / sport / f"{day:%Y-%m-%d}.csv"


def load_csv(day: date, sport: str = "nba") -> pd.DataFrame:
    p = csv_path(day, sport)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{p} is missing columns: {sorted(missing)}")
    df["home"] = df.home.map(lambda n: resolve(str(n), sport) or str(n))
    df["away"] = df.away.map(lambda n: resolve(str(n), sport) or str(n))
    return df


def save_csv(day: date, df: pd.DataFrame, sport: str = "nba") -> Path:
    path = csv_path(day, sport)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False)
    return path


def to_splits(df: pd.DataFrame) -> dict[tuple[str, str, str], MarketSplit]:
    out = {}
    for r in df.itertuples():
        def f(v):
            return None if pd.isna(v) else float(v)
        out[(r.home, r.away, r.market)] = MarketSplit(
            r.market, f(r.open_line), f(r.current_line), f(r.tickets_pct_a), f(r.money_pct_a))
    return out


# ---------------------------------------------------------------------------
AN_URL = "https://api.actionnetwork.com/web/v2/scoreboard/{league}"
AN_LEAGUE = {"nba": "nba", "nfl": "nfl", "nhl": "nhl", "ncaaf": "ncaaf", "ncaab": "ncaab"}


def _pct(info: dict | None, kind: str) -> float | None:
    try:
        return float(info[kind]["percent"])
    except (TypeError, KeyError, ValueError):
        return None


def parse_action_network(data: dict, sport: str = "nba") -> pd.DataFrame:
    """Defensive parser: skips anything that does not match the expected shape."""
    rows = []
    for g in data.get("games", []):
        try:
            teams = {t["id"]: t for t in g["teams"]}
            home = resolve(teams[g["home_team_id"]].get("full_name", ""), sport)
            away = resolve(teams[g["away_team_id"]].get("full_name", ""), sport)
        except (KeyError, TypeError):
            continue
        markets = g.get("markets") or {}
        book = next(iter(markets.values()), {}) if isinstance(markets, dict) else {}
        event = book.get("event", {}) if isinstance(book, dict) else {}
        for key, market, side_a in (("spread", "spread", "home"), ("total", "total", "over"),
                                    ("moneyline", "ml", "home")):
            for o in event.get(key, []) or []:
                if o.get("side") != side_a:
                    continue
                bi = o.get("bet_info") or {}
                rows.append(dict(home=home, away=away, market=market, open_line=None,
                                 current_line=o.get("value") if market != "ml" else o.get("odds"),
                                 tickets_pct_a=_pct(bi, "tickets"), money_pct_a=_pct(bi, "money")))
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_action_network(day: date, sport: str = "nba") -> pd.DataFrame:
    r = requests.get(AN_URL.format(league=AN_LEAGUE[sport]), params={"bookIds": "15", "date": f"{day:%Y%m%d}", "periods": "event"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    return parse_action_network(r.json(), sport)
