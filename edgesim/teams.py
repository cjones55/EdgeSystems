"""Canonical NBA team identities and the aliases different data sources use."""
from __future__ import annotations

# tricode: (full name, aliases)
TEAMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ATL": ("Atlanta Hawks", ("Hawks", "Atlanta")),
    "BOS": ("Boston Celtics", ("Celtics", "Boston")),
    "BKN": ("Brooklyn Nets", ("Nets", "Brooklyn", "NewJersey", "New Jersey", "NJN", "BRK")),
    "CHA": ("Charlotte Hornets", ("Hornets", "Charlotte", "Bobcats", "CHO")),
    "CHI": ("Chicago Bulls", ("Bulls", "Chicago")),
    "CLE": ("Cleveland Cavaliers", ("Cavaliers", "Cleveland", "Cavs")),
    "DAL": ("Dallas Mavericks", ("Mavericks", "Dallas", "Mavs")),
    "DEN": ("Denver Nuggets", ("Nuggets", "Denver")),
    "DET": ("Detroit Pistons", ("Pistons", "Detroit")),
    "GSW": ("Golden State Warriors", ("Warriors", "Golden State", "GS")),
    "HOU": ("Houston Rockets", ("Rockets", "Houston")),
    "IND": ("Indiana Pacers", ("Pacers", "Indiana")),
    "LAC": ("LA Clippers", ("Clippers", "Los Angeles Clippers", "LA Clippers", "LAClippers")),
    "LAL": ("Los Angeles Lakers", ("Lakers", "LA Lakers", "LALakers")),
    "MEM": ("Memphis Grizzlies", ("Grizzlies", "Memphis")),
    "MIA": ("Miami Heat", ("Heat", "Miami")),
    "MIL": ("Milwaukee Bucks", ("Bucks", "Milwaukee")),
    "MIN": ("Minnesota Timberwolves", ("Timberwolves", "Minnesota", "Wolves")),
    "NOP": ("New Orleans Pelicans", ("Pelicans", "New Orleans", "NOH", "NO")),
    "NYK": ("New York Knicks", ("Knicks", "New York", "NY")),
    "OKC": ("Oklahoma City Thunder", ("Thunder", "Oklahoma City", "OklahomaCity")),
    "ORL": ("Orlando Magic", ("Magic", "Orlando")),
    "PHI": ("Philadelphia 76ers", ("76ers", "Sixers", "Seventysixers", "Philadelphia")),
    "PHX": ("Phoenix Suns", ("Suns", "Phoenix", "PHO")),
    "POR": ("Portland Trail Blazers", ("Trail Blazers", "Trailblazers", "Blazers", "Portland")),
    "SAC": ("Sacramento Kings", ("Kings", "Sacramento")),
    "SAS": ("San Antonio Spurs", ("Spurs", "San Antonio", "SA")),
    "TOR": ("Toronto Raptors", ("Raptors", "Toronto")),
    "UTA": ("Utah Jazz", ("Jazz", "Utah", "UTAH")),
    "WAS": ("Washington Wizards", ("Wizards", "Washington", "WSH")),
}

_LOOKUP: dict[str, str] = {}
for code, (full, aliases) in TEAMS.items():
    for key in (code, full, *aliases):
        _LOOKUP[key.lower().replace(" ", "")] = code


def to_code(name: str) -> str | None:
    """Map any known spelling to a tricode, or None if unknown."""
    if not isinstance(name, str):
        return None
    return _LOOKUP.get(name.lower().replace(" ", "").replace(".", ""))


def full_name(code: str) -> str:
    return TEAMS[code][0]
