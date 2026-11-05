"""Parsers are tested against hand-written payloads in the documented shapes.
Live endpoints are checked with `python -m edgesim doctor` on your machine."""
import pandas as pd

from edgesim.providers.espn import parse_boxscore, parse_injuries, parse_scoreboard
from edgesim.providers.odds_api import consensus, parse_game_odds, parse_props
from edgesim.providers.public_betting import parse_action_network, to_splits

SCOREBOARD = {"events": [{
    "id": "401", "date": "2026-11-04T23:30Z", "season": {"type": 2},
    "competitions": [{
        "status": {"type": {"completed": True}},
        "competitors": [
            {"homeAway": "home", "score": "112", "team": {"abbreviation": "GS", "displayName": "Golden State Warriors"}},
            {"homeAway": "away", "score": "104", "team": {"abbreviation": "NY", "displayName": "New York Knicks"}}],
        "odds": [{"details": "GS -4.5", "overUnder": 228.5, "spread": -4.5}]}]}]}

BOX = {"boxscore": {"players": [
    {"team": {"abbreviation": "GS"}, "statistics": [{
        "labels": ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "+/-", "PTS"],
        "athletes": [
            {"athlete": {"id": "1", "displayName": "Steph"}, "stats": ["34", "10-20", "5-11", "4-4", "0", "5", "5", "7", "1", "0", "3", "2", "+9", "29"]},
            {"athlete": {"id": "2", "displayName": "Bench"}, "didNotPlay": True, "stats": []}]}]},
    {"team": {"abbreviation": "NY"}, "statistics": [{
        "labels": ["MIN", "PTS", "REB", "AST", "3PT", "STL", "BLK", "TO"],
        "athletes": [{"athlete": {"id": "3", "displayName": "Jalen"}, "stats": ["36", "30", "4", "8", "3-7", "2", "0", "2"]}]}]}]}}

ODDS = [{"id": "ev1", "home_team": "Golden State Warriors", "away_team": "New York Knicks",
         "commence_time": "2026-11-05T03:00:00Z", "bookmakers": [
    {"key": "dk", "markets": [
        {"key": "spreads", "outcomes": [{"name": "Golden State Warriors", "price": -110, "point": -4.5},
                                        {"name": "New York Knicks", "price": -110, "point": 4.5}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": -112, "point": 228.5},
                                       {"name": "Under", "price": -108, "point": 228.5}]},
        {"key": "h2h", "outcomes": [{"name": "Golden State Warriors", "price": -190},
                                    {"name": "New York Knicks", "price": 160}]}]},
    {"key": "fd", "markets": [
        {"key": "spreads", "outcomes": [{"name": "Golden State Warriors", "price": -105, "point": -4.5},
                                        {"name": "New York Knicks", "price": -115, "point": 4.5}]}]}]}]


def test_espn_scoreboard():
    g = parse_scoreboard(SCOREBOARD)[0]
    assert (g["home"], g["away"], g["home_pts"], g["away_pts"]) == ("GSW", "NYK", 112, 104)
    assert g["espn_spread"] == -4.5 and g["espn_total"] == 228.5


def test_espn_boxscore_handles_dnp_and_label_order():
    rows = parse_boxscore(BOX, "401", pd.Timestamp("2026-11-04"))
    assert len(rows) == 2
    steph = next(r for r in rows if r["player"] == "Steph")
    assert (steph["pts"], steph["ast"], steph["fg3m"], steph["opp"]) == (29, 7, 5, "NYK")
    jalen = next(r for r in rows if r["player"] == "Jalen")
    assert (jalen["pts"], jalen["fg3m"], jalen["team"]) == (30, 3, "NYK")


def test_espn_injuries():
    df = parse_injuries({"injuries": [{"displayName": "Boston Celtics", "injuries": [
        {"athlete": {"displayName": "Jayson Tatum"}, "status": "Out", "shortComment": "Achilles"}]}]})
    assert df.iloc[0].team == "BOS" and df.iloc[0].status == "Out"


def test_odds_api_parse_and_consensus():
    df = parse_game_odds(ODDS)
    assert set(df.market) == {"spread", "total", "ml"}
    c = consensus(df)
    home_spread = c[(c.market == "spread") & (c.side == "home")].iloc[0]
    assert home_spread.point == -4.5
    assert home_spread.best_price == -105 and home_spread.best_book == "fd"


def test_odds_api_props():
    df = parse_props({"id": "ev1", "bookmakers": [{"key": "dk", "markets": [
        {"key": "player_points", "outcomes": [
            {"name": "Over", "description": "Stephen Curry", "price": -115, "point": 27.5},
            {"name": "Under", "description": "Stephen Curry", "price": -105, "point": 27.5}]}]}]})
    assert list(df.stat.unique()) == ["pts"] and len(df) == 2


def test_action_network_parser_is_defensive():
    assert parse_action_network({"games": [{"bad": 1}]}).empty
    data = {"games": [{"home_team_id": 1, "away_team_id": 2,
                       "teams": [{"id": 1, "full_name": "Boston Celtics"}, {"id": 2, "full_name": "Miami Heat"}],
                       "markets": {"15": {"event": {"spread": [
                           {"side": "home", "value": -5.5, "odds": -110,
                            "bet_info": {"tickets": {"percent": 70}, "money": {"percent": 45}}}]}}}}]}
    df = parse_action_network(data)
    assert df.iloc[0].home == "BOS" and df.iloc[0].tickets_pct_a == 70
    sp = to_splits(df)[("BOS", "MIA", "spread")]
    assert sp.money_a == 45


def test_espn_odds_moneyline_both_formats():
    from edgesim.providers.espn import parse_odds
    old = parse_odds({"spread": -3.5, "overUnder": 49.5,
                      "homeTeamOdds": {"moneyLine": -160, "spreadOdds": -115},
                      "awayTeamOdds": {"moneyLine": 135, "spreadOdds": -105}})
    assert (old["espn_home_ml"], old["espn_away_ml"]) == (-160, 135)
    assert old["espn_spread_home_price"] == -115 and old["espn_open_spread"] is None
    new = parse_odds({"moneyline": {"home": {"open": {"odds": "-170"}, "close": {"odds": "-140"}},
                                    "away": {"close": {"odds": "+118"}}},
                      "pointSpread": {"home": {"open": {"line": "-4.5"}, "close": {"line": "-2.5", "odds": "-115"}}},
                      "total": {"over": {"open": {"line": "o48.5"}, "close": {"line": "o49.5", "odds": "EVEN"}}}})
    assert (new["espn_home_ml"], new["espn_away_ml"], new["espn_open_home_ml"]) == (-140, 118, -170)
    assert (new["espn_spread"], new["espn_open_spread"], new["espn_spread_home_price"]) == (-2.5, -4.5, -115)
    assert (new["espn_total"], new["espn_open_total"], new["espn_over_price"]) == (49.5, 48.5, 100)


def test_espn_odds_off_and_placeholder_prices_are_missing():
    from edgesim.providers.espn import parse_odds
    o = parse_odds({"spread": -34.5, "moneyline": {"home": {"close": {"odds": "OFF"}, "open": {"odds": "-100000"}},
                                                   "away": {"close": {"odds": "OFF"}, "open": {"odds": "+5000"}}}})
    assert o["espn_home_ml"] is None and o["espn_away_ml"] is None and o["espn_open_home_ml"] is None


def test_espn_boxscore_skips_athletes_without_ids():
    data = {"boxscore": {"players": [
        {"team": {"abbreviation": "GS"}, "statistics": [{"labels": ["MIN", "PTS"], "athletes": [
            {"athlete": {"displayName": "No Id"}, "stats": ["20", "10"]},
            {"stats": ["12", "4"]},
            {"athlete": {"id": "7", "displayName": "Has Id"}, "stats": ["30", "21"]}]}]},
        {"statistics": []}]}}
    rows = parse_boxscore(data, "401", pd.Timestamp("2026-11-04"))
    assert [r["player"] for r in rows] == ["No Id", "Has Id"]
