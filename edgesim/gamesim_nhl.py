"""Monte Carlo NHL game simulator.

Goals arrive as a Poisson process at each team's expected rate, with:
  * model uncertainty on each team's scoring rate (lognormal)
  * goalie pulls: a team trailing by 1-2 late pulls its goalie, which
    raises both teams' scoring rates (empty-net goals). This is what makes
    the puck line (-1.5) and late totals realistic.
  * 3-on-3 overtime (sudden death, 5 minutes) and a shootout. Books grade
    totals with the shootout winner counted as one goal, and so do we.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gamesim import SimResult


@dataclass
class NHLSimConfig:
    n_sims: int = 20000
    rating_sd: float = 0.08        # lognormal sd on each team's scoring rate
    game_sd: float = 0.04          # shared shock (pace / goalie quality / refs)
    lead_mult: float = 0.90        # score effects: leading teams sit back...
    trail_mult: float = 1.12       # ...trailing teams push
    pull_minutes: tuple = (1.6, 2.6)   # goalie pulled when down 1, down 2
    leader_mult: float = 5.0       # scoring-rate multiplier vs an empty net
    trailer_mult: float = 2.2      # extra attacker
    ot_rate: float = 0.18          # combined goals per minute in 3v3 OT
    home_so_edge: float = 0.0
    reg_scale: float = 0.92        # regulation share of final-score goals (ENG + OT/SO add the rest)
    seed: int | None = None


def simulate_game(exp_home: float, exp_away: float, cfg: NHLSimConfig | None = None) -> SimResult:
    cfg = cfg or NHLSimConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_sims
    shared = rng.normal(0, cfg.game_sd, n)
    lh = exp_home * np.exp(rng.normal(0, cfg.rating_sd, n) + shared - (cfg.rating_sd ** 2 + cfg.game_sd ** 2) / 2)
    la = exp_away * np.exp(rng.normal(0, cfg.rating_sd, n) + shared - (cfg.rating_sd ** 2 + cfg.game_sd ** 2) / 2)
    # regulation is 60 minutes; ratings are learned on final scores, which
    # include empty-net and OT/shootout goals, so scale down to regulation rates
    lh, la = lh * cfg.reg_scale, la * cfg.reg_scale
    late = max(cfg.pull_minutes)
    home = np.zeros(n, dtype=np.int64)
    away = np.zeros(n, dtype=np.int64)
    seg = (60 - late) / 6
    for _ in range(6):                      # play in segments so score effects apply
        d = home - away
        mh = np.where(d > 0, cfg.lead_mult, np.where(d < 0, cfg.trail_mult, 1.0))
        ma = np.where(d < 0, cfg.lead_mult, np.where(d > 0, cfg.trail_mult, 1.0))
        home += rng.poisson(lh * mh * seg / 60)
        away += rng.poisson(la * ma * seg / 60)
    # final minutes, with goalie pulls when a team trails by 1 or 2
    m = home - away
    for diff, minutes in ((1, cfg.pull_minutes[0]), (2, cfg.pull_minutes[1])):
        for sign in (1, -1):                 # +1: home leads
            idx = np.nonzero(m == sign * diff)[0]
            if not len(idx):
                continue
            normal = late - minutes
            lead_rate = (lh if sign == 1 else la)[idx]
            trail_rate = (la if sign == 1 else lh)[idx]
            lead_g = rng.poisson(lead_rate * (normal + minutes * cfg.leader_mult) / 60)
            trail_g = rng.poisson(trail_rate * (normal + minutes * cfg.trailer_mult) / 60)
            if sign == 1:
                home[idx] += lead_g
                away[idx] += trail_g
            else:
                away[idx] += lead_g
                home[idx] += trail_g
    rest = np.nonzero((np.abs(m) != 1) & (np.abs(m) != 2))[0]
    home[rest] += rng.poisson(lh[rest] * late / 60)
    away[rest] += rng.poisson(la[rest] * late / 60)
    # overtime and shootout
    tied = np.nonzero(home == away)[0]
    went_ot = np.zeros(n, dtype=bool)
    went_ot[tied] = True
    if len(tied):
        share = lh[tied] / (lh[tied] + la[tied])
        ot_goal = rng.random(len(tied)) < 1 - np.exp(-cfg.ot_rate * 5)
        so_home = rng.random(len(tied)) < 0.5 + cfg.home_so_edge
        home_wins = np.where(ot_goal, rng.random(len(tied)) < share, so_home)
        home[tied] += home_wins
        away[tied] += ~home_wins
    return SimResult(home=home, away=away, went_ot=went_ot)
