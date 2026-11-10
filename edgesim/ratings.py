"""Online team strength model (offense / defense in points per game).

Expected score for the home team:
    league_avg/2 + hca/2 + off[home] - def[away] + rest adjustments
Ratings update after every game with a learning rate (a Kalman-style
exponentially weighted fit), and regress toward the mean between seasons.
The model is strictly walk-forward: a prediction for a game only uses
games played before it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RatingParams:
    k: float = 0.03               # learning rate per game
    k_early: float = 0.05         # learning rate for a team's first games of a season
    early_games: int = 12
    hca: float = 2.6              # home court advantage in points (margin)
    b2b: float = 1.6              # margin penalty for playing on 0 days rest
    rest_bonus: float = 0.4       # per extra rest day beyond 1, capped at 2 extra
    season_regress: float = 0.6   # share of rating kept into a new season
    league_avg: float = 210.0     # starting league total, learned online
    k_league: float = 0.004
    k_hca: float = 0.002
    new_team_off: float = 0.0     # starting rating for a team never seen (e.g. FCS in college)
    new_team_def: float = 0.0
    err_cap: float = 1e9          # cap per-game surprise (limits blowouts of weak teams)
    recentre: bool = True         # keep ratings mean-zero (off for 300+ team college leagues)


@dataclass
class TeamRatings:
    params: RatingParams = field(default_factory=RatingParams)
    off: dict[str, float] = field(default_factory=dict)
    deff: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)
    season: int | None = None

    def __post_init__(self):
        self.league_avg = self.params.league_avg
        self.hca = self.params.hca

    # ------------------------------------------------------------------
    def _rest_adj(self, rest: float) -> float:
        p = self.params
        if rest <= 1:
            return -p.b2b / 2
        return min(rest - 2, 2) * p.rest_bonus / 2

    def _o(self, t: str) -> float:
        return self.off.get(t, self.params.new_team_off)

    def _d(self, t: str) -> float:
        return self.deff.get(t, self.params.new_team_def)

    def expected(self, home: str, away: str, home_rest: float = 2, away_rest: float = 2,
                 neutral: bool = False) -> tuple[float, float]:
        """Expected (home points, away points). ``neutral`` removes home advantage."""
        base = self.league_avg / 2
        rest_h, rest_a = self._rest_adj(home_rest), self._rest_adj(away_rest)
        h = 0.0 if neutral else self.hca / 2
        eh = base + h + self._o(home) - self._d(away) + rest_h - rest_a
        ea = base - h + self._o(away) - self._d(home) + rest_a - rest_h
        return eh, ea

    def new_season(self, season: int) -> None:
        r = self.params.season_regress
        self.off = {t: v * r for t, v in self.off.items()}
        self.deff = {t: v * r for t, v in self.deff.items()}
        self.games = {}
        self.season = season

    def update(self, home: str, away: str, home_pts: float, away_pts: float,
               home_rest: float = 2, away_rest: float = 2, season: int | None = None,
               neutral: bool = False) -> None:
        if season is not None and season != self.season:
            self.new_season(season)
        p = self.params
        for t in (home, away):
            self.off.setdefault(t, p.new_team_off)
            self.deff.setdefault(t, p.new_team_def)
        eh, ea = self.expected(home, away, home_rest, away_rest, neutral)
        eh_err = max(-p.err_cap, min(p.err_cap, home_pts - eh))
        ea_err = max(-p.err_cap, min(p.err_cap, away_pts - ea))
        kh = p.k_early if self.games.get(home, 0) < p.early_games else p.k
        ka = p.k_early if self.games.get(away, 0) < p.early_games else p.k
        self.off[home] = self.off.get(home, 0) + kh * eh_err
        self.deff[away] = self.deff.get(away, 0) - ka * eh_err
        self.off[away] = self.off.get(away, 0) + ka * ea_err
        self.deff[home] = self.deff.get(home, 0) - kh * ea_err
        self.league_avg += p.k_league * (eh_err + ea_err)
        if not neutral:
            self.hca += p.k_hca * (eh_err - ea_err)
        self.games[home] = self.games.get(home, 0) + 1
        self.games[away] = self.games.get(away, 0) + 1
        if p.recentre:
            self._recentre()

    def _recentre(self) -> None:
        """Keep ratings mean-zero so league_avg carries the level."""
        for dct in (self.off, self.deff):
            if dct:
                m = sum(dct.values()) / len(dct)
                for t in dct:
                    dct[t] -= m

    def table(self) -> list[dict]:
        rows = [dict(team=t, off=self.off.get(t, 0), deff=self.deff.get(t, 0),
                     net=self.off.get(t, 0) + self.deff.get(t, 0)) for t in self.off]
        return sorted(rows, key=lambda r: -r["net"])
