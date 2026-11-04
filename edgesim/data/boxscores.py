"""Player box scores (NocturneBear/NBA-Data-2010-2024, MIT)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "player_games.parquet"

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
_RENAME = {
    "points": "pts", "reboundsTotal": "reb", "assists": "ast", "threePointersMade": "fg3m",
    "steals": "stl", "blocks": "blk", "turnovers": "tov", "personId": "player_id",
    "personName": "player", "teamTricode": "team", "gameId": "game_id", "game_date": "date",
    "season_year": "season",
}


def _minutes(s: str) -> float:
    if not isinstance(s, str) or ":" not in s:
        return 0.0
    m, sec = s.split(":")
    return int(m) + int(sec) / 60


def load_player_games(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and CACHE.exists():
        return pd.read_parquet(CACHE)
    parts = sorted(RAW_DIR.glob("regular_season_box_scores_*.csv"))
    if not parts:
        raise FileNotFoundError("Run `python scripts/fetch_data.py` first")
    df = pd.concat(pd.read_csv(p, low_memory=False) for p in parts)
    df["min"] = df["minutes"].map(_minutes)
    df["home"] = df["matchup"].str.contains(" vs. ")
    df["opp"] = df["matchup"].str[-3:]
    df = df.rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    keep = ["season", "date", "game_id", "team", "opp", "home", "player_id", "player",
            "min", *STATS]
    df = df[keep].sort_values(["player_id", "date"]).reset_index(drop=True)
    try:
        df.to_parquet(CACHE)
    except Exception:  # pyarrow missing: fine, just no cache
        pass
    return df
