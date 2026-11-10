"""Load and clean the historical results + betting lines archives (NBA, NFL, NHL).

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
# (swap threshold for |spread|, valid total range, max |spread|)
LIMITS = {"nba": (60, (150, 290), 30), "nfl": (25, (28, 70), 25)}


def raw_path(sport: str) -> Path:
    return ROOT / "data" / "raw" / f"{sport}_archive_10Y.json"


def _fix_pair(spread: float, total: float, home_ml: float, sport: str = "nba") -> tuple[float, float]:
    """Repair a (home spread, total) pair. Returns NaNs if unrecoverable."""
    swap_at, (tmin, tmax), smax = LIMITS[sport]
    if abs(spread) > swap_at and 0 <= total <= smax:
        mag, total = total, abs(spread)
        # sign from the moneyline: favourite has negative ML
        spread = -mag if home_ml < 0 else mag
    if not (tmin <= total <= tmax) or abs(spread) > smax:
        return np.nan, np.nan
    return spread, total


def _other_side(price: float, overround: float = 1.045) -> float:
    """Price of the opposite side when the archive only stores one side."""
    from ..odds import implied_prob, prob_to_american
    p = overround - implied_prob(price)
    return float(np.round(prob_to_american(min(max(p, 0.02), 0.98))))


def load_games(path: Path | str | None = None, sport: str = "nba") -> pd.DataFrame:
    if sport == "nhl":
        return load_nhl(path or raw_path("nhl"))
    path = path or (RAW if sport == "nba" else raw_path(sport))
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        h, a = to_code(r["home_team"], sport), to_code(r["away_team"], sport)
        if not h or not a or h == a:
            continue
        try:
            hs, as_ = int(r["home_final"]), int(r["away_final"])
        except (TypeError, ValueError):
            continue
        hml, aml = r["home_close_ml"], r["away_close_ml"]
        cs, ct = _fix_pair(float(r["home_close_spread"]), float(r["close_over_under"]), hml, sport)
        os_, ot = _fix_pair(float(r["home_open_spread"]), float(r["open_over_under"]), hml, sport)
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
    for c in ("spread_home_price", "spread_away_price", "over_price", "under_price"):
        df[c] = -110.0
    df["open_home_ml"] = np.nan
    df["open_away_ml"] = np.nan
    df["sport"] = sport
    df["neutral"] = False
    df = add_rest(df)
    return df


def load_nhl(path: Path | str) -> pd.DataFrame:
    """NHL rows: moneylines (open + close), puck line with price, total with over price."""
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        h, a = to_code(r["home_team"], "nhl"), to_code(r["away_team"], "nhl")
        if not h or not a or h == a:
            continue
        try:
            hs, as_ = int(r["home_final"]), int(r["away_final"])
        except (TypeError, ValueError):
            continue
        d = str(int(r["date"]))

        def ml(v):
            return float(v) if isinstance(v, (int, float)) and abs(v) >= 100 else np.nan

        pl, pl_price = float(r["home_close_spread"]), float(r["home_close_spread_odds"])
        ok_pl = abs(pl) == 1.5 and abs(pl_price) >= 100
        tot, tot_price = float(r["close_over_under"]), float(r["close_over_under_odds"])
        ok_t = 4 <= tot <= 8 and abs(tot_price) >= 100
        otot = float(r["open_over_under"])
        out.append(dict(
            date=pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}"), season=int(r["season"]),
            home=h, away=a, home_pts=hs, away_pts=as_,
            home_ml=ml(r["home_close_ml"]), away_ml=ml(r["away_close_ml"]),
            open_home_ml=ml(r["home_open_ml"]), open_away_ml=ml(r["away_open_ml"]),
            open_spread=np.nan, close_spread=pl if ok_pl else np.nan,
            spread_home_price=pl_price if ok_pl else np.nan,
            spread_away_price=_other_side(pl_price) if ok_pl else np.nan,
            # opening total has no matching price in the archive, so it is not used
            open_total=np.nan, close_total=tot if ok_t else np.nan,
            # the archive's total price is the UNDER's (checked: higher implied
            # probability -> fewer overs), so the over price is derived
            under_price=tot_price if ok_t else np.nan,
            over_price=_other_side(tot_price) if ok_t else np.nan,
        ))
    df = pd.DataFrame(out).sort_values(["date", "home"]).reset_index(drop=True)
    df["sport"] = "nhl"
    df["neutral"] = False
    return add_rest(df)


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
