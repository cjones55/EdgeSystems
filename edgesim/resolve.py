"""Map the team names each data source uses onto one key per team.

Pro leagues use short codes (BOS, KC, TOR...). College teams use the key of
the historical data: the CollegeFootballData school name for football (these
share ESPN's team ids) and the Kaggle TeamID for basketball. A team that can't
be resolved keeps a cleaned version of its own name, so it is still rated
(starting from the new-team prior) instead of being dropped.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .teams import to_code

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _norm(s: str) -> str:
    s = str(s).lower().replace("&", "and").replace("é", "e")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _ncaaf_tables() -> tuple[dict, dict]:
    by_id, by_name = {}, {}
    files = sorted((RAW / "ncaaf").glob("lines_*.csv"))
    if files:
        cols = ["homeTeam", "homeTeamId", "awayTeam", "awayTeamId"]
        l = pd.concat(pd.read_csv(p, usecols=cols) for p in files)
        for n, i in list(zip(l.homeTeam, l.homeTeamId)) + list(zip(l.awayTeam, l.awayTeamId)):
            by_id[str(int(i))] = n
            by_name[_norm(n)] = n
    for alias, n in (("miami fl", "Miami"), ("miami florida", "Miami"), ("ole miss", "Ole Miss"),
                     ("mississippi", "Ole Miss"), ("uconn", "UConn"), ("connecticut", "UConn"),
                     ("usc", "USC"), ("southern california", "USC"), ("lsu", "LSU"),
                     ("byu", "BYU"), ("ucf", "UCF"), ("smu", "SMU"), ("tcu", "TCU"),
                     ("utsa", "UTSA"), ("utep", "UTEP"), ("unlv", "UNLV"),
                     ("hawaii rainbow warriors", "Hawai'i"), ("hawaii", "Hawai'i"),
                     ("san jose state", "San José State"), ("app state", "App State"),
                     ("appalachian state", "App State"), ("north carolina state", "NC State"),
                     ("nc state", "NC State"), ("umass", "Massachusetts"), ("fiu", "Florida International"),
                     ("louisiana lafayette", "Louisiana"), ("louisiana monroe", "UL Monroe"),
                     ("southern mississippi", "Southern Miss"), ("texas san antonio", "UTSA"),
                     ("miami ohio", "Miami (OH)"), ("central florida", "UCF")):
        by_name.setdefault(alias, n)
    return by_id, by_name


def resolve_ncaaf(name: str, espn_id: str | None = None) -> str:
    by_id, by_name = _ncaaf_tables()
    if espn_id is not None and str(espn_id) in by_id:
        return by_id[str(espn_id)]
    words = _norm(name).split()
    for k in range(len(words), 0, -1):          # drop mascot words from the end
        cand = " ".join(words[:k])
        for c in (cand, cand.replace(" st", " state")):
            if c in by_name:
                return by_name[c]
    return str(name).strip()


def resolve_ncaab(name: str) -> str:
    from .data.college import resolve_ncaab as r
    try:
        out = r(name)
    except FileNotFoundError:
        out = None
    return out or str(name).strip()


def resolve(name: str, sport: str, espn_id: str | None = None) -> str | None:
    if sport == "ncaaf":
        return resolve_ncaaf(name, espn_id)
    if sport == "ncaab":
        return resolve_ncaab(name)
    return to_code(name, sport)


def display(key: str, sport: str) -> str:
    """Human-readable team name for a key."""
    if sport == "ncaab" and str(key).isdigit():
        try:
            from .data.college import ncaab_names
            return _ncaab_names().get(str(key), str(key))
        except FileNotFoundError:
            return str(key)
    return str(key)


@lru_cache(maxsize=1)
def _ncaab_names() -> dict:
    from .data.college import ncaab_names
    return ncaab_names()
