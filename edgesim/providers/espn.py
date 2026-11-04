"""ESPN public site API (no key needed): schedule, scores, box scores, injuries.

These endpoints are unofficial and undocumented. Responses are cached on disk
so a season of history is only downloaded once.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from ..resolve import resolve
from ..teams import to_code

ROOT_URL = "https://site.api.espn.com/apis/site/v2/sports"
BASE = f"{ROOT_URL}/basketball/nba"


def base(sport: str = "nba") -> str:
    from ..sports import get
    return f"{ROOT_URL}/{get(sport).espn_path}"


def _team_key(team: dict, sport: str) -> str | None:
    if sport in ("ncaaf", "ncaab"):
        return resolve(team.get("location") or team.get("displayName", ""), sport, team.get("id"))
    return to_code(team.get("abbreviation", ""), sport) or to_code(team.get("displayName", ""), sport)
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "espn"
HEADERS = {"User-Agent": "edgesim/1.0"}


def _get(url: str, params: dict | None = None, cache_key: str | None = None,
         ttl_hours: float | None = None) -> dict:
    path = CACHE / f"{cache_key}.json" if cache_key else None
    if path and path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if ttl_hours is None or age_h < ttl_hours:
            return json.loads(path.read_text())
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
def parse_scoreboard(data: dict, sport: str = "nba") -> list[dict]:
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        h, a = teams.get("home"), teams.get("away")
        if not h or not a:
            continue
        status = comp.get("status", ev.get("status", {})).get("type", {})
        g = dict(event_id=ev["id"], start=ev.get("date"),
                 home=_team_key(h["team"], sport), away=_team_key(a["team"], sport),
                 home_name=h["team"].get("displayName"), away_name=a["team"].get("displayName"),
                 neutral=bool(comp.get("neutralSite", False)),
                 completed=bool(status.get("completed")),
                 home_pts=int(h["score"]) if status.get("completed") else None,
                 away_pts=int(a["score"]) if status.get("completed") else None,
                 season_type=ev.get("season", {}).get("type"))
        g.update(parse_odds((comp.get("odds") or [{}])[0]))
        games.append(g)
    return games


def _american(v) -> float | None:
    """ESPN prices come as numbers or strings like "-180", "+150", "EVEN"."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v != 0 else None
    t = str(v).strip().upper()
    if t in ("EVEN", "EV", "PK"):
        return 100.0
    try:
        x = float(t.replace("+", ""))
        # ESPN posts placeholders like -100000 on huge mismatches: treat as no line
        return x if 100 <= abs(x) <= 5000 else None
    except ValueError:
        return None


def _line(v) -> float | None:
    """Lines like "-3.5", "o49.5", "u49.5", "PK"."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().lower().lstrip("ou")
    if t in ("pk", "even"):
        return 0.0
    try:
        return float(t)
    except ValueError:
        return None


def _dig(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def parse_odds(odds: dict) -> dict:
    """ESPN's consensus odds for one game: spread, total, moneyline, their prices,
    and opening numbers when ESPN lists them. Missing pieces come back as None."""
    h, a = odds.get("homeTeamOdds") or {}, odds.get("awayTeamOdds") or {}
    out = dict(
        espn_spread=_line(odds.get("spread")),
        espn_total=_line(odds.get("overUnder")),
        espn_home_ml=_american(h.get("moneyLine")) or _american(_dig(odds, "moneyline", "home", "close", "odds")),
        espn_away_ml=_american(a.get("moneyLine")) or _american(_dig(odds, "moneyline", "away", "close", "odds")),
        espn_spread_home_price=_american(h.get("spreadOdds")) or _american(_dig(odds, "pointSpread", "home", "close", "odds")),
        espn_spread_away_price=_american(a.get("spreadOdds")) or _american(_dig(odds, "pointSpread", "away", "close", "odds")),
        espn_over_price=_american(odds.get("overOdds")) or _american(_dig(odds, "total", "over", "close", "odds")),
        espn_under_price=_american(odds.get("underOdds")) or _american(_dig(odds, "total", "under", "close", "odds")),
        espn_open_spread=_line(_dig(odds, "pointSpread", "home", "open", "line")),
        espn_open_total=_line(_dig(odds, "total", "over", "open", "line")),
        espn_open_home_ml=_american(_dig(odds, "moneyline", "home", "open", "odds")),
    )
    if out["espn_spread"] is None:
        out["espn_spread"] = _line(_dig(odds, "pointSpread", "home", "close", "line"))
    if out["espn_total"] is None:
        out["espn_total"] = _line(_dig(odds, "total", "over", "close", "line"))
    return out


def scoreboard(day: date, live: bool = False, sport: str = "nba") -> list[dict]:
    from ..sports import get
    key = None if live else f"{sport}_scoreboard_{day:%Y%m%d}"
    ttl = None if day < date.today() - timedelta(days=1) else 1.0
    params = {"dates": f"{day:%Y%m%d}", **get(sport).espn_params}
    data = _get(f"{base(sport)}/scoreboard", params, cache_key=key, ttl_hours=ttl)
    return parse_scoreboard(data, sport)


def season_start(day: date, sport: str = "nba") -> date:
    """First day of the season containing ``day``. Every supported league starts
    between August and November, so July is the cut-over."""
    from ..sports import get
    y = day.year if day.month >= 7 else day.year - 1
    return date(y, get(sport).season_month, 1)


def results(start: date, end: date, workers: int = 8, sport: str = "nba") -> pd.DataFrame:
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    with ThreadPoolExecutor(workers) as ex:
        chunks = list(ex.map(lambda d: scoreboard(d, sport=sport), days))
    rows = [dict(g, date=pd.Timestamp(d)) for d, gs in zip(days, chunks) for g in gs]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df.completed & df.home.notna() & df.away.notna()]
    # skip preseason (type 1) and all-star games
    df = df[df.season_type.fillna(2).isin([2, 3, 5])]
    df["season"] = df.date.map(lambda d: season_label(d.date(), sport))
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
_LABELS = {"MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast", "STL": "stl",
           "BLK": "blk", "TO": "tov"}


def parse_boxscore(data: dict, event_id: str, day: pd.Timestamp) -> list[dict]:
    rows = []
    box = data.get("boxscore", {})
    teams = [to_code(t["team"].get("abbreviation")) for t in box.get("players", [])]
    for i, team_block in enumerate(box.get("players", [])):
        team = teams[i]
        opp = teams[1 - i] if len(teams) == 2 else None
        for stat_block in team_block.get("statistics", []):
            labels = stat_block.get("labels") or []
            for ath in stat_block.get("athletes", []):
                vals = ath.get("stats") or []
                if ath.get("didNotPlay") or len(vals) != len(labels):
                    continue
                rec = dict(date=day, game_id=str(event_id), team=team, opp=opp,
                           player_id=str(ath["athlete"]["id"]),
                           player=ath["athlete"].get("displayName"))
                for lab, v in zip(labels, vals):
                    if lab in _LABELS:
                        rec[_LABELS[lab]] = _num(v)
                    elif lab == "3PT":
                        rec["fg3m"] = _num(str(v).split("-")[0])
                if rec.get("min", 0) > 0:
                    rows.append(rec)
    return rows


def _num(v) -> float:
    try:
        return float(str(v).split(":")[0]) if v not in ("", "--", None) else 0.0
    except ValueError:
        return 0.0


def season_label(day: date, sport: str = "nba") -> int:
    """Season id matching the historical data: the start year, except college
    basketball, which (like the Kaggle data) is named for the year it ends."""
    y = season_start(day, sport).year
    return y + 1 if sport == "ncaab" else y


def boxscore(event_id: str, day: pd.Timestamp) -> list[dict]:
    data = _get(f"{BASE}/summary", {"event": event_id}, cache_key=f"summary_{event_id}")
    return parse_boxscore(data, event_id, day)


def player_games(games: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        chunks = list(ex.map(lambda r: boxscore(r.event_id, r.date), games.itertuples()))
    df = pd.DataFrame([r for c in chunks for r in c])
    if not df.empty:
        df["season"] = df.date.map(lambda d: str(season_start(d.date()).year))
    return df


def parse_injuries(data: dict) -> pd.DataFrame:
    rows = []
    for team in data.get("injuries", []):
        code = to_code(team.get("displayName", "")) or to_code(team.get("abbreviation", ""))
        for inj in team.get("injuries", []):
            rows.append(dict(team=code, player=inj.get("athlete", {}).get("displayName"),
                             status=inj.get("status"), detail=inj.get("shortComment")))
    return pd.DataFrame(rows, columns=["team", "player", "status", "detail"])


def injuries(sport: str = "nba") -> pd.DataFrame:
    df = parse_injuries(_get(f"{base(sport)}/injuries"))
    if sport != "nba":
        df["team"] = None   # parse_injuries maps NBA names; other sports match by player
    return df
