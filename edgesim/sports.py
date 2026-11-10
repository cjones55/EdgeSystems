"""Per-sport configuration: data feeds, rating parameters, simulator.

Rating parameters were tuned on each sport's training seasons only (see
``train``); the backtest reports results on the ``test`` seasons.
``rating_sd`` in each simulator is the model's measured extra error vs the
closing line on the training seasons, so the sims are not over-confident.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from . import gamesim, gamesim_nfl, gamesim_nhl
from .ratings import RatingParams
from .sharp import SharpConfig


@dataclass
class Sport:
    key: str
    label: str
    espn_path: str
    odds_key: str
    season_month: int                 # month the season starts (for ESPN history)
    rating: RatingParams
    simulate: Callable
    sim_config: Callable              # (**overrides) -> config object
    train: range
    test: range
    blend: dict = field(default_factory=lambda: {"spread": 0.12, "total": 0.10, "ml": 0.04,
                                                 "prop": 0.5})
    spread_label: str = "Spread"
    college: bool = False
    has_props: bool = False
    espn_params: dict = field(default_factory=dict)
    sharp: SharpConfig = field(default_factory=SharpConfig)
    archive_note: str = ""
    # ratings learn from a blend of the final score and the closing line's implied
    # score: 0 = scores only, 1 = closing lines only (games without lines use scores)
    market_weight: float = 0.0
    # extra rating uncertainty early in a season: variance A / (n + c)^2 added to the
    # margin (first pair) and total (second pair), n = fewer games played this season
    early_var: tuple = (0.0, 1.0, 0.0, 1.0)
    # the model's weight in the market blend ramps up from 0 over this many games of the
    # season (0 = full weight from the first game)
    blend_ramp_games: int = 0

    def sim(self, exp_home: float, exp_away: float, **overrides):
        return self.simulate(exp_home, exp_away, self.sim_config(**overrides))

    def early_sd(self, games_played: float) -> dict:
        """rating_sd / rating_total_sd overrides for a game where the less experienced
        team has played ``games_played`` games this season (empty when not needed)."""
        am, cm, at, ct = self.early_var
        if not am and not at:
            return {}
        base = self.sim_config()
        n = max(float(games_played), 0.0)
        return dict(rating_sd=float(np.hypot(base.rating_sd, np.sqrt(am / (n + cm) ** 2))),
                    rating_total_sd=float(np.hypot(base.rating_total_sd, np.sqrt(at / (n + ct) ** 2))))

    def sim_matchup(self, ratings, home: str, away: str, exp_home: float, exp_away: float,
                    **overrides):
        n = min(ratings.games.get(home, 0), ratings.games.get(away, 0))
        return self.sim(exp_home, exp_away, **{**self.early_sd(n), **overrides})


def _cfg(cls, **defaults):
    return lambda **kw: cls(**{**defaults, **kw})


SPORTS: dict[str, Sport] = {
    "nba": Sport(
        "nba", "NBA", "basketball/nba", "basketball_nba", 10, RatingParams(),
        gamesim.simulate_game, _cfg(gamesim.SimConfig, total_sd=16.0),
        range(2011, 2016), range(2016, 2022),
        has_props=True, archive_note="2011-12 to 2021-22, open + close lines"),
    "nfl": Sport(
        "nfl", "NFL", "football/nfl", "americanfootball_nfl", 9,
        RatingParams(k=0.06, k_early=0.1, early_games=4, season_regress=0.6, league_avg=44,
                     hca=2.5, b2b=0, rest_bonus=0),
        gamesim_nfl.simulate_game,
        _cfg(gamesim_nfl.NFLSimConfig, rating_sd=2.4, rating_total_sd=2.2),
        range(2011, 2016), range(2016, 2022),
        sharp=SharpConfig(rlm_min_move={"spread": 0.5, "total": 0.5, "ml": 0.02},
                          steam_move={"spread": 1.5, "total": 1.5, "ml": 0.05}),
        archive_note="2011 to 2021 seasons, open + close lines"),
    "nhl": Sport(
        "nhl", "NHL", "hockey/nhl", "icehockey_nhl", 10,
        RatingParams(k=0.01, k_early=0.02, early_games=10, season_regress=0.7, league_avg=5.6,
                     hca=0.25, b2b=0.25, rest_bonus=0.0, k_league=0.002, k_hca=0.0005),
        gamesim_nhl.simulate_game, _cfg(gamesim_nhl.NHLSimConfig),
        range(2011, 2016), range(2016, 2022), spread_label="Puck line",
        blend={"spread": 0.10, "total": 0.10, "ml": 0.06, "prop": 0.5},
        sharp=SharpConfig(rlm_min_move={"spread": 9.0, "total": 0.5, "ml": 0.015},
                          steam_move={"spread": 9.0, "total": 0.5, "ml": 0.04}),
        archive_note="2011-12 to 2021-22, open + close moneylines, closing puck line and total with prices"),
    "ncaaf": Sport(
        "ncaaf", "College football", "football/college-football", "americanfootball_ncaaf", 8,
        RatingParams(k=0.3, k_early=0.3, early_games=4, season_regress=0.95, league_avg=55,
                     hca=3.0, b2b=0, rest_bonus=0, recentre=False, new_team_off=-8,
                     new_team_def=-8, err_cap=18),
        gamesim_nfl.simulate_game,
        _cfg(gamesim_nfl.NFLSimConfig, drives_mean=12.5, rating_sd=7.0, rating_total_sd=8.0,
             ot_tie_prob=0.0, ot_fg_share=0.35),
        range(2015, 2020), range(2020, 2027), college=True,
        espn_params={"groups": "80", "limit": "300"},
        sharp=SharpConfig(rlm_min_move={"spread": 1.0, "total": 1.0, "ml": 0.02},
                          steam_move={"spread": 2.5, "total": 2.5, "ml": 0.05}),
        archive_note="2015 to 2026 FBS games, closing lines (opening lines and moneylines for about half)",
        market_weight=0.7),
    "ncaab": Sport(
        "ncaab", "College basketball", "basketball/mens-college-basketball", "basketball_ncaab", 11,
        RatingParams(k=0.08, k_early=0.12, early_games=8, season_regress=0.85, league_avg=140,
                     hca=3.2, b2b=0, rest_bonus=0, recentre=False, err_cap=22),
        gamesim.simulate_game,
        _cfg(gamesim.SimConfig, margin_sd=10.6, total_sd=14.5, rating_sd=0.5,
             rating_total_sd=4.0, minutes=40.0, ppp=1.04, pace_sd=3.5),
        range(2021, 2022), range(2022, 2027), college=True,
        espn_params={"groups": "50", "limit": "400"},
        # fitted on 2021-2023 closing lines: model error by games played vs late season
        early_var=(1400.0, 5.3, 1200.0, 4.2),
        blend_ramp_games=10, market_weight=0.5,
        sharp=SharpConfig(rlm_min_move={"spread": 1.0, "total": 1.5, "ml": 0.02},
                          steam_move={"spread": 2.0, "total": 3.0, "ml": 0.05}),
        archive_note="results 2015-2026; near-closing lines from many books for 2020-21 to 2022-23"),
}


def get(key: str) -> Sport:
    try:
        return SPORTS[key.lower()]
    except KeyError:
        raise ValueError(f"Unknown sport {key!r}; choose from {', '.join(SPORTS)}") from None


def rating_points(g, w: float) -> tuple[float, float]:
    """Scores to update ratings with: the final score blended with the closing
    line's implied score (weight ``w``) when the game has a closing spread and total."""
    hp, ap = g.home_pts, g.away_pts
    cs, ct = getattr(g, "close_spread", None), getattr(g, "close_total", None)
    if w > 0 and cs is not None and ct is not None and cs == cs and ct == ct:
        mh, ma = ct / 2 - cs / 2, ct / 2 + cs / 2
        hp, ap = (1 - w) * hp + w * mh, (1 - w) * ap + w * ma
    return hp, ap


def load_history(key: str):
    """Historical games with lines for a sport (standard columns)."""
    from .data.college import load_ncaab, load_ncaaf
    from .data.historical import load_games
    if key == "ncaaf":
        return load_ncaaf()
    if key == "ncaab":
        return load_ncaab()
    return load_games(sport=key)


__all__ = ["Sport", "SPORTS", "get", "load_history", "replace"]
