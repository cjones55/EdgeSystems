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

from ..teams import to_code

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
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
def parse_scoreboard(data: dict) -> list[dict]:
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        h, a = teams.get("home"), teams.get("away")
        if not h or not a:
            continue
        status = comp.get("status", ev.get("status", {})).get("type", {})
        g = dict(event_id=ev["id"], start=ev.get("date"),
                 home=to_code(h["team"].get("abbreviation")) or to_code(h["team"].get("displayName")),
                 away=to_code(a["team"].get("abbreviation")) or to_code(a["team"].get("displayName")),
                 completed=bool(status.get("completed")),
                 home_pts=int(h["score"]) if status.get("completed") else None,
                 away_pts=int(a["score"]) if status.get("completed") else None,
                 season_type=ev.get("season", {}).get("type"))
        odds = (comp.get("odds") or [{}])[0]
        g["espn_spread"] = odds.get("spread")          # home spread
        g["espn_total"] = odds.get("overUnder")
        games.append(g)
    return games


def scoreboard(day: date, live: bool = False) -> list[dict]:
    key = None if live else f"scoreboard_{day:%Y%m%d}"
    ttl = None if day < date.today() - timedelta(days=1) else 1.0
    data = _get(f"{BASE}/scoreboard", {"dates": f"{day:%Y%m%d}"}, cache_key=key, ttl_hours=ttl)
    return parse_scoreboard(data)


def season_start(day: date) -> date:
    """NBA regular seasons start in October."""
    y = day.year if day.month >= 9 else day.year - 1
    return date(y, 10, 1)


def results(start: date, end: date, workers: int = 8) -> pd.DataFrame:
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    with ThreadPoolExecutor(workers) as ex:
        chunks = list(ex.map(scoreboard, days))
    rows = [dict(g, date=pd.Timestamp(d)) for d, gs in zip(days, chunks) for g in gs]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df.completed & df.home.notna() & df.away.notna()]
    # skip preseason (type 1) and all-star games
    df = df[df.season_type.fillna(2).isin([2, 3, 5])]
    df["season"] = df.date.map(lambda d: season_start(d.date()).year)
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


def injuries() -> pd.DataFrame:
    return parse_injuries(_get(f"{BASE}/injuries"))
