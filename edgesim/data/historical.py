"""Load and clean the historical NBA results + betting lines archive.

Source: sportsbookreview-scraper (MIT), seasons 2011-12 .. 2021-22.
The raw scrape has known defects that we repair or drop:
  * spread and total columns swapped on some rows (spread=178, total=4)
  * open spread recorded as 0 when it was missing
  * a handful of rows with team=0
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..teams import to_code

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "nba_archive_10Y.json"


def _fix_pair(spread: float, total: float, home_ml: float) -> tuple[float, float]:
    """Repair a (home spread, total) pair. Returns NaNs if unrecoverable."""
    if abs(spread) > 60 and 0 <= total <= 30:
        mag, total = total, abs(spread)
        # sign from the moneyline: favourite has negative ML
        spread = -mag if home_ml < 0 else mag
    if not (150 <= total <= 290) or abs(spread) > 30:
        return np.nan, np.nan
    return spread, total


def load_games(path: Path | str = RAW) -> pd.DataFrame:
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        h, a = to_code(r["home_team"]), to_code(r["away_team"])
        if not h or not a or h == a:
            continue
        try:
            hs, as_ = int(r["home_final"]), int(r["away_final"])
        except (TypeError, ValueError):
            continue
        hml, aml = r["home_close_ml"], r["away_close_ml"]
        cs, ct = _fix_pair(float(r["home_close_spread"]), float(r["close_over_under"]), hml)
        os_, ot = _fix_pair(float(r["home_open_spread"]), float(r["open_over_under"]), hml)
        if os_ == 0 and not np.isnan(cs) and abs(cs) >= 3:
            os_ = np.nan  # 0 used as "missing"
        d = str(int(r["date"]))
        out.append(dict(
            date=pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}"), season=int(r["season"]),
            home=h, away=a, home_pts=hs, away_pts=as_,
            home_ml=hml if isinstance(hml, (int, float)) and abs(hml) >= 100 else np.nan,
            away_ml=aml if isinstance(aml, (int, float)) and abs(aml) >= 100 else np.nan,
            open_spread=os_, close_spread=cs, open_total=ot, close_total=ct,
        ))
    df = pd.DataFrame(out).sort_values(["date", "home"]).reset_index(drop=True)
    # sanity: favourite on ML should match favourite on spread; otherwise the spread sign is bad
    bad = (df.home_ml < -150) & (df.close_spread > 2) | (df.home_ml > 150) & (df.close_spread < -2)
    df.loc[bad, ["close_spread", "open_spread"]] *= -1
    df = add_rest(df)
    return df


def add_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game (capped at 4); 1 means back-to-back."""
    long = pd.concat([
        df[["date"]].assign(team=df.home, idx=df.index, side="home"),
        df[["date"]].assign(team=df.away, idx=df.index, side="away"),
    ]).sort_values(["team", "date"])
    long["rest"] = long.groupby("team").date.diff().dt.days.clip(upper=4).fillna(4)
    df = df.copy()
    for side in ("home", "away"):
        m = long[long.side == side].set_index("idx").rest
        df[f"{side}_rest"] = m.reindex(df.index).values
    return df
