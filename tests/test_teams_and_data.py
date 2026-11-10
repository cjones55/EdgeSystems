import math

import pytest

from edgesim.data.historical import ROOT, _fix_pair, add_rest
from edgesim.teams import TEAMS, to_code


@pytest.mark.parametrize("name,code", [("Seventysixers", "PHI"), ("Golden State", "GSW"),
                                       ("NewJersey", "BKN"), ("LA Clippers", "LAC"), ("GS", "GSW"),
                                       ("UTAH", "UTA"), ("Los Angeles Lakers", "LAL"), ("WSH", "WAS")])
def test_aliases(name, code):
    assert to_code(name) == code


def test_thirty_teams():
    assert len(TEAMS) == 30
    assert to_code("Not A Team") is None


def test_fix_swapped_spread_and_total():
    # raw scrape swapped them: spread=178, total=4, home is an underdog (+155)
    s, t = _fix_pair(178.0, 4.0, 155)
    assert (s, t) == (4.0, 178.0)
    s, t = _fix_pair(197.0, 1.5, -200)
    assert (s, t) == (-1.5, 197.0)
    s, t = _fix_pair(-5.0, 210.0, -200)
    assert (s, t) == (-5.0, 210.0)
    s, t = _fix_pair(-5.0, 20.0, -200)        # garbage -> NaN
    assert math.isnan(s) and math.isnan(t)


def test_rest_days():
    import pandas as pd
    df = pd.DataFrame(dict(date=pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-05"]),
                           home=["BOS", "BOS", "MIA"], away=["MIA", "NYK", "BOS"]))
    r = add_rest(df)
    assert r.loc[1, "home_rest"] == 1      # BOS back to back
    assert r.loc[2, "away_rest"] == 3      # BOS 3 days later
    assert r.loc[2, "home_rest"] == 4      # MIA capped at 4


@pytest.mark.skipif(not (ROOT / "data/raw/nba_archive_10Y.json").exists(), reason="no data")
def test_archive_loads_clean():
    from edgesim.data.historical import load_games
    df = load_games()
    assert len(df) > 13000
    assert df.close_spread.abs().max() <= 30
    assert df.close_total.between(150, 290).all()
    # favourites on the moneyline are favourites on the spread
    assert ((df.home_ml < -200) & (df.close_spread > 0)).sum() == 0


def test_ncaab_names_never_fall_back_to_another_school():
    import pytest
    from edgesim.data.college import resolve_ncaab
    from edgesim.providers.espn import _team_key
    try:
        assert resolve_ncaab("Kentucky Wildcats") == "1246"
    except FileNotFoundError:
        pytest.skip("historical data not downloaded")
    for n in ("Kentucky Christian Knights", "Kentucky State Thorobreds"):
        assert resolve_ncaab(n) is None
    assert resolve_ncaab("Lehigh Mountain Hawks") is not None
    assert resolve_ncaab("San José State Spartans") == resolve_ncaab("San Jose State")
    kc = dict(location="Kentucky Christian", displayName="Kentucky Christian Knights", name="Knights")
    assert not str(_team_key(kc, "ncaab")).isdigit()
    uk = dict(location="Kentucky", displayName="Kentucky Wildcats", name="Wildcats")
    assert _team_key(uk, "ncaab") == "1246"
    for loc, disp, mascot in (("UAlbany", "UAlbany Great Danes", "Great Danes"),
                              ("UT Rio Grande Valley", "UT Rio Grande Valley Vaqueros", "Vaqueros"),
                              ("Queens University", "Queens University Royals", "Royals"),
                              ("LSU New Orleans", "LSU New Orleans Privateers", "Privateers")):
        assert str(_team_key(dict(location=loc, displayName=disp, name=mascot), "ncaab")).isdigit(), disp
