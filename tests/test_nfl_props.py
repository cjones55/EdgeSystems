import numpy as np
import pandas as pd
import pytest

from edgesim import nfl_props as N
from edgesim.plays import UNIT_SIZES
from edgesim.providers.odds_api import parse_props


def _games(n=12):
    """Two teams, one QB and one WR each, n weekly games."""
    rows = []
    for w in range(n):
        gid = f"2025_{w:02d}_AAA_BBB"
        for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
            base = 260 if team == "AAA" else 200
            rows.append(dict(player_id=f"{team}qb", player=f"{team} Quarterback", position="QB",
                             position_group="QB", season=2025, week=w, game_id=gid, team=team,
                             opponent_team=opp, date=pd.Timestamp("2025-09-07") + pd.Timedelta(weeks=w),
                             implied=24.0, **{k: 0.0 for k in N.STATS}))
            rows[-1].update(pass_yds=base + 10 * (w % 3), pass_att=34, pass_cmp=22, pass_tds=2,
                            pass_ints=1, rush_yds=10, rush_att=3)
            rows.append(dict(player_id=f"{team}wr", player=f"{team} Receiver", position="WR",
                             position_group="WR", season=2025, week=w, game_id=gid, team=team,
                             opponent_team=opp, date=pd.Timestamp("2025-09-07") + pd.Timedelta(weeks=w),
                             implied=24.0, **{k: 0.0 for k in N.STATS}))
            rows[-1].update(rec=6, rec_yds=80, rush_rec_yds=80, anytime_td=0.5 * (w % 2))
    return pd.DataFrame(rows)


def test_features_use_only_earlier_games():
    f = N.build_features(_games())
    qb = f[f.player_id == "AAAqb"].sort_values("date")
    assert np.isnan(qb.proj_pass_yds.iloc[0])          # nothing before week 0
    # ~265 of recent form, nudged up because BBB allows more passing than the league
    assert 260 <= qb.proj_pass_yds.iloc[-1] <= 300


def test_project_upcoming_and_price():
    pg = _games()
    day = pg.date.max() + pd.Timedelta(days=7)
    proj = N.project_upcoming(pg, [("AAA", "BBB", -3.0, 47.0)], day)
    assert set(proj.player) == {"AAA Quarterback", "AAA Receiver", "BBB Quarterback", "BBB Receiver"}
    assert (proj.game_id.str.startswith("next_")).all()
    qb = proj[proj.player == "AAA Quarterback"].iloc[0]
    assert qb.opponent_team == "BBB" and qb.proj_pass_yds > 200
    lines = pd.DataFrame([dict(player="AAA Quarterback", stat="pass_yds", line=199.5,
                               over_price=-110, under_price=-110, book="fanduel"),
                          dict(player="AAA Receiver", stat="anytime_td", line=0.5,
                               over_price=200, under_price=None, book="draftkings")])
    priced = N.price_props(lines, proj)
    over = priced[(priced.stat == "pass_yds") & (priced.side == "over")].iloc[0]
    assert over.model_prob > 0.6 and over.play
    assert set(priced.units.dropna()) <= set(UNIT_SIZES) - {10}
    assert "anytime TD" in priced[priced.stat == "anytime_td"].pick.iloc[0]


def test_prob_over_is_monotone_and_handles_pushes():
    p = N.NFLPropParams()
    ps = [N.prob_over("rec_yds", 60, x, p)[0] for x in (20.5, 40.5, 60.5, 80.5, 120.5)]
    assert all(a > b for a, b in zip(ps, ps[1:]))
    po, push = N.prob_over("rec", 5.0, 5, p)
    assert push > 0 and 0 < po < 1
    assert N.prob_over("anytime_td", 0.5, 0.5, p)[0] == pytest.approx(1 - np.exp(-0.5))


def test_odds_api_nfl_props_and_lines():
    data = {"id": "e1", "bookmakers": [
        {"key": "fanduel", "markets": [
            {"key": "player_reception_yds", "outcomes": [
                {"name": "Over", "description": "A Receiver", "point": 64.5, "price": -114},
                {"name": "Under", "description": "A Receiver", "point": 64.5, "price": -106}]},
            {"key": "player_anytime_td", "outcomes": [
                {"name": "Yes", "description": "A Receiver", "price": 180}]}]},
        {"key": "pinnacle", "markets": [
            {"key": "player_reception_yds", "outcomes": [
                {"name": "Over", "description": "A Receiver", "point": 62.5, "price": -105},
                {"name": "Under", "description": "A Receiver", "point": 62.5, "price": -115}]}]}]}
    raw = parse_props(data, "nfl")
    assert set(raw.stat) == {"rec_yds", "anytime_td"}
    lines = N.lines_from_odds(raw, ["fanduel", "draftkings"]).set_index("stat")
    assert lines.loc["rec_yds", "line"] == 64.5 and lines.loc["rec_yds", "book"] == "fanduel/fanduel"
    assert lines.loc["anytime_td", "line"] == 0.5 and pd.isna(lines.loc["anytime_td", "under_price"])


def test_box_scores():
    from edgesim import boxscore as BX
    from edgesim.gamesim import SimResult
    pg = _games()
    proj = N.project_upcoming(pg, [("AAA", "BBB", -3.0, 47.0)], pg.date.max() + pd.Timedelta(days=7))
    box = BX.nfl_box(proj, "AAA")
    assert list(box) == ["Passing", "Receiving"] or "Passing" in box
    assert box["Passing"].player.iloc[0] == "AAA Quarterback"
    sim = SimResult(home=np.array([24, 24, 27, 20]), away=np.array([17, 17, 20, 23]),
                    went_ot=np.zeros(4, bool))
    ls = BX.likely_scores(sim)
    assert (ls.iloc[0].away, ls.iloc[0].home, ls.iloc[0].chance) == (17, 24, 0.5)


def test_qb_out_lowers_team_projection():
    from edgesim import nfl_qb as Q
    from edgesim.plays import GameMarket, PlayConfig, evaluate_game
    from edgesim.ratings import TeamRatings
    from edgesim.sports import get
    pg = _games(10)
    pg["attempts"] = np.where(pg.position == "QB", 34, 0)
    pg["passing_epa"] = np.where(pg.player_id == "AAAqb", 12.0, np.where(pg.position == "QB", 0.0, np.nan))
    pg["rushing_epa"] = 0.0
    # a backup who threw a few passes last season
    extra = pg[pg.player_id == "AAAqb"].head(1).assign(
        player_id="AAAbk", player="AAA Backup", attempts=5, passing_epa=-6.0,
        date=pg.date.min() - pd.Timedelta(days=200), game_id="2024_18_AAA_BBB")
    pg = pd.concat([extra, pg], ignore_index=True)
    day = pg.date.max() + pd.Timedelta(days=7)
    healthy = Q.team_qb_status(pg, "AAA", day, set())
    hurt = Q.team_qb_status(pg, "AAA", day, {"aaa quarterback"})
    assert abs(healthy.adjustment) < 0.5
    assert hurt.starter == "AAA Backup" and hurt.adjustment < -3 and hurt.out == ["AAA Quarterback"]
    r = TeamRatings(get("nfl").rating)
    g0 = GameMarket(home="AAA", away="BBB", spread=-3, total=45)
    g1 = GameMarket(home="AAA", away="BBB", spread=-3, total=45, home_adj=hurt.adjustment)
    s0, _ = evaluate_game(g0, r, cfg=PlayConfig.for_sport("nfl", n_sims=4000))
    s1, _ = evaluate_game(g1, r, cfg=PlayConfig.for_sport("nfl", n_sims=4000))
    assert s1["home_win"] < s0["home_win"] - 0.05
