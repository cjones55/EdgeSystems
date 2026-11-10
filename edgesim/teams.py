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

NFL_TEAMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ARI": ("Arizona Cardinals", ("Cardinals", "Arizona")),
    "ATL": ("Atlanta Falcons", ("Falcons", "Atlanta")),
    "BAL": ("Baltimore Ravens", ("Ravens", "Baltimore")),
    "BUF": ("Buffalo Bills", ("Bills", "Buffalo")),
    "CAR": ("Carolina Panthers", ("Panthers", "Carolina")),
    "CHI": ("Chicago Bears", ("Bears", "Chicago")),
    "CIN": ("Cincinnati Bengals", ("Bengals", "Cincinnati")),
    "CLE": ("Cleveland Browns", ("Browns", "Cleveland")),
    "DAL": ("Dallas Cowboys", ("Cowboys", "Dallas")),
    "DEN": ("Denver Broncos", ("Broncos", "Denver")),
    "DET": ("Detroit Lions", ("Lions", "Detroit")),
    "GB": ("Green Bay Packers", ("Packers", "Green Bay", "GNB")),
    "HOU": ("Houston Texans", ("Texans", "Houston")),
    "IND": ("Indianapolis Colts", ("Colts", "Indianapolis")),
    "JAX": ("Jacksonville Jaguars", ("Jaguars", "Jacksonville", "JAC")),
    "KC": ("Kansas City Chiefs", ("Chiefs", "Kansas City", "KCChiefs", "Kansas", "KAN")),
    "LV": ("Las Vegas Raiders", ("Raiders", "Oakland", "Oakland Raiders", "Las Vegas", "OAK", "LVR")),
    "LAC": ("Los Angeles Chargers", ("Chargers", "SanDiego", "San Diego", "San Diego Chargers", "SD")),
    "LAR": ("Los Angeles Rams", ("Rams", "St.Louis", "St Louis", "St. Louis Rams", "STL", "LA")),
    "MIA": ("Miami Dolphins", ("Dolphins", "Miami")),
    "MIN": ("Minnesota Vikings", ("Vikings", "Minnesota")),
    "NE": ("New England Patriots", ("Patriots", "New England", "NWE")),
    "NO": ("New Orleans Saints", ("Saints", "New Orleans", "NOR")),
    "NYG": ("New York Giants", ("Giants",)),
    "NYJ": ("New York Jets", ("Jets",)),
    "PHI": ("Philadelphia Eagles", ("Eagles", "Philadelphia")),
    "PIT": ("Pittsburgh Steelers", ("Steelers", "Pittsburgh")),
    "SF": ("San Francisco 49ers", ("49ers", "Fortyniners", "San Francisco", "SFO")),
    "SEA": ("Seattle Seahawks", ("Seahawks", "Seattle")),
    "TB": ("Tampa Bay Buccaneers", ("Buccaneers", "Tampa Bay", "Bucs", "TAM")),
    "TEN": ("Tennessee Titans", ("Titans", "Tennessee")),
    "WSH": ("Washington Commanders", ("Commanders", "Redskins", "Washington", "Football Team",
                                      "Washington Football Team", "WAS")),
}

NHL_TEAMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ANA": ("Anaheim Ducks", ("Ducks", "Anaheim")),
    "BOS": ("Boston Bruins", ("Bruins", "Boston")),
    "BUF": ("Buffalo Sabres", ("Sabres", "Buffalo")),
    "CGY": ("Calgary Flames", ("Flames", "Calgary")),
    "CAR": ("Carolina Hurricanes", ("Hurricanes", "Carolina")),
    "CHI": ("Chicago Blackhawks", ("Blackhawks", "Chicago")),
    "COL": ("Colorado Avalanche", ("Avalanche", "Colorado")),
    "CBJ": ("Columbus Blue Jackets", ("Blue Jackets", "Columbus")),
    "DAL": ("Dallas Stars", ("Stars", "Dallas")),
    "DET": ("Detroit Red Wings", ("Red Wings", "Detroit")),
    "EDM": ("Edmonton Oilers", ("Oilers", "Edmonton")),
    "FLA": ("Florida Panthers", ("Panthers", "Florida")),
    "LA": ("Los Angeles Kings", ("Kings", "LA Kings", "LAK", "Los Angeles")),
    "MIN": ("Minnesota Wild", ("Wild", "Minnesota")),
    "MTL": ("Montreal Canadiens", ("Canadiens", "Montreal", "Montréal Canadiens")),
    "NSH": ("Nashville Predators", ("Predators", "Nashville")),
    "NJ": ("New Jersey Devils", ("Devils", "New Jersey", "NJD")),
    "NYI": ("New York Islanders", ("Islanders", "NY Islanders")),
    "NYR": ("New York Rangers", ("Rangers", "NY Rangers")),
    "OTT": ("Ottawa Senators", ("Senators", "Ottawa")),
    "PHI": ("Philadelphia Flyers", ("Flyers", "Philadelphia")),
    "PIT": ("Pittsburgh Penguins", ("Penguins", "Pittsburgh")),
    "SJ": ("San Jose Sharks", ("Sharks", "San Jose", "SJS")),
    "SEA": ("Seattle Kraken", ("Kraken", "SeattleKraken", "Seattle")),
    "STL": ("St. Louis Blues", ("Blues", "St.Louis", "St Louis", "St Louis Blues")),
    "TB": ("Tampa Bay Lightning", ("Lightning", "Tampa Bay", "Tampa", "TBL")),
    "TOR": ("Toronto Maple Leafs", ("Maple Leafs", "Toronto")),
    # the Coyotes franchise moved to Utah in 2024; one rating history
    "UTA": ("Utah Mammoth", ("Utah Hockey Club", "Utah", "UTAH", "Mammoth", "Coyotes",
                             "Arizona Coyotes", "Phoenix", "Arizona", "Arizonas", "ARI")),
    "VAN": ("Vancouver Canucks", ("Canucks", "Vancouver")),
    "VGK": ("Vegas Golden Knights", ("Golden Knights", "Vegas", "VEG")),
    "WSH": ("Washington Capitals", ("Capitals", "Washington", "WAS")),
    "WPG": ("Winnipeg Jets", ("Jets", "Winnipeg", "WinnipegJets")),
}

TEAMS_BY_SPORT = {"nba": TEAMS, "nfl": NFL_TEAMS, "nhl": NHL_TEAMS}


def _key(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return s.lower().replace(" ", "").replace(".", "")


_LOOKUP: dict[str, dict[str, str]] = {}
for _sport, _table in TEAMS_BY_SPORT.items():
    _LOOKUP[_sport] = {}
    for code, (full, aliases) in _table.items():
        for key in (code, full, *aliases):
            _LOOKUP[_sport][_key(key)] = code


def to_code(name: str, sport: str = "nba") -> str | None:
    """Map any known spelling to a team code for ``sport``, or None if unknown."""
    if not isinstance(name, str):
        return None
    return _LOOKUP[sport].get(_key(name))


def teams(sport: str = "nba") -> dict:
    return TEAMS_BY_SPORT[sport]


def full_name(code: str, sport: str = "nba") -> str:
    return TEAMS_BY_SPORT[sport][code][0]
