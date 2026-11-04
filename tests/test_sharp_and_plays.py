import pytest

from edgesim.plays import GameMarket, PlayConfig, daily_card, evaluate_game, prop_plays
from edgesim.ratings import TeamRatings
from edgesim.sharp import MarketSplit, detect, movement_toward_a


def test_movement_conventions():
    assert movement_toward_a(MarketSplit("spread", -3, -4.5, None, None)) == 1.5
    assert movement_toward_a(MarketSplit("total", 220, 218, None, None)) == -2
    assert movement_toward_a(MarketSplit("ml", -150, -170, None, None)) > 0


def test_reverse_line_movement():
    # 75% of bets on home, but the home spread got cheaper (-5 -> -4): sharp on away
    s = detect(MarketSplit("spread", -5, -4, 75, 70))
    assert s.side == "B"
    assert any("Reverse line move" in r for r in s.reasons)


def test_money_ticket_split():
    s = detect(MarketSplit("total", 220, 220, 40, 65))
    assert s.side == "A"                       # big money on the over
    assert any("split" in r for r in s.reasons)


def test_steam_only():
    s = detect(MarketSplit("spread", -2, -4.5, None, None))
    assert s.side == "A" and s.score > 0


def test_no_signal():
    s = detect(MarketSplit("spread", -3, -3, 52, 50))
    assert s.side is None and s.score == 0


def _ratings_home_strong():
    r = TeamRatings()
    for _ in range(40):
        r.update("BOS", "DET", 125, 100)
        r.update("DET", "BOS", 98, 120)
    return r


def test_no_plays_when_market_agrees_with_model():
    r = TeamRatings()           # everyone average: fair line is just home court
    eh, ea = r.expected("BOS", "MIA")
    g = GameMarket("BOS", "MIA", spread=-round(eh - ea) - 0.5, total=round(eh + ea) + 0.5)
    plays, cands, sims = daily_card([g], r)
    assert plays.empty
    assert len(cands) == 4


def test_big_model_edge_with_sharp_agreement_is_tier_a():
    r = _ratings_home_strong()
    g = GameMarket("BOS", "DET", spread=-1.5, total=220.5, spread_home_price=-105,
                   spread_away_price=-115, open_spread=-0.5)
    splits = {("BOS", "DET", "spread"): MarketSplit("spread", -0.5, -1.5, 35, 60)}
    plays, cands, _ = daily_card([g], r, splits)
    top = plays.iloc[0]
    assert top.pick.startswith("BOS")
    assert top.tier.startswith("A")
    assert top.ev > 0


def test_sharp_against_blocks_play():
    r = _ratings_home_strong()
    g = GameMarket("BOS", "DET", spread=-1.5, open_spread=-3.5)
    splits = {("BOS", "DET", "spread"): MarketSplit("spread", -3.5, -1.5, 80, 40)}
    plays, _, _ = daily_card([g], r, splits)
    assert not (plays.pick.str.startswith("BOS")).any()


def test_prop_plays_prices_both_sides():
    import pandas as pd
    feat = pd.DataFrame([dict(player="A B", team="BOS", proj_min=34.0, min_sd=4.0,
                              rate_pts=0.8, opp_pts=1.0)])
    lines = pd.DataFrame([dict(player="A B", stat="pts", line=20.5, over_price=-110,
                               under_price=-110, book="x")])
    out = prop_plays(lines, feat)
    assert set(out.pick) == {"Over 20.5", "Under 20.5"}
    over = out[out.pick == "Over 20.5"].iloc[0]
    assert over.projection == pytest.approx(27.2)
    assert over.ev > 0 and bool(over.play)
