"""Monte Carlo NBA game simulator.

Each simulation:
  1. Draws model uncertainty: a team-strength shock and a pace/total shock
     (our ratings are not perfect, so the sim should not be over-confident).
  2. Draws the number of possessions (pace varies game to game).
  3. Plays every possession for both teams from a points-per-possession
     distribution over {0, 1, 2, 3} tuned to each team's expected efficiency.
  4. Rescales the regulation margin/total noise so the spread of outcomes
     matches what NBA games actually show around a sharp closing line
     (margin SD ~12, total SD ~17.5, measured on 13,900 games). Raw possession
     sims are too noisy because they ignore game-script effects (fouling,
     garbage time, coasting with a lead).
  5. Plays 5-minute overtimes until ties are broken.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEAGUE_PPP = 1.12            # points per possession (modern NBA ~1.12-1.15)
BASE_PMF = np.array([0.50, 0.05, 0.33, 0.12])   # P(0,1,2,3 pts) on a possession
BASE_MEAN = float(BASE_PMF @ np.arange(4))


def possession_pmf(ppp: float) -> np.ndarray:
    """Outcome distribution for one possession with mean ``ppp``."""
    f = ppp / BASE_MEAN
    scoring = BASE_PMF[1:] * f
    if scoring.sum() >= 1:
        raise ValueError(f"ppp {ppp} too high")
    return np.concatenate([[1 - scoring.sum()], scoring])


@dataclass
class SimConfig:
    n_sims: int = 20000
    margin_sd: float = 11.8      # outcome noise around the true mean (calibrated)
    total_sd: float = 17.3
    rating_sd: float = 2.3       # our model's own error on the margin vs the market
    rating_total_sd: float = 4.0  # model error on the total
    pace_sd: float = 4.0         # possessions
    seed: int | None = None


@dataclass
class SimResult:
    home: np.ndarray
    away: np.ndarray
    went_ot: np.ndarray

    @property
    def margin(self) -> np.ndarray:
        return self.home - self.away

    @property
    def total(self) -> np.ndarray:
        return self.home + self.away

    def p_home_win(self) -> float:
        return float((self.margin > 0).mean())

    def p_cover(self, home_spread: float) -> tuple[float, float]:
        """(P home covers, P push) for a home spread like -4.5."""
        adj = self.margin + home_spread
        return float((adj > 0).mean()), float((adj == 0).mean())

    def p_over(self, total: float) -> tuple[float, float]:
        return float((self.total > total).mean()), float((self.total == total).mean())

    def fair_spread(self) -> float:
        return -float(np.median(self.margin))

    def fair_total(self) -> float:
        return float(np.median(self.total))

    def summary(self) -> dict:
        return dict(
            home_pts=float(self.home.mean()), away_pts=float(self.away.mean()),
            home_win=self.p_home_win(), fair_spread=self.fair_spread(),
            fair_total=self.fair_total(), margin_sd=float(self.margin.std()),
            total_sd=float(self.total.std()), ot_rate=float(self.went_ot.mean()),
        )


def _play(rng: np.random.Generator, poss: np.ndarray, ppp: np.ndarray) -> np.ndarray:
    """Points scored over ``poss`` possessions with per-sim efficiency ``ppp``."""
    ppp = np.clip(ppp, 0.6, 1.6)
    # group sims by rounded efficiency to vectorise the multinomial draws
    keys = np.round(ppp, 3)
    pts = np.empty(len(poss), dtype=np.int64)
    for k in np.unique(keys):
        idx = np.nonzero(keys == k)[0]
        counts = rng.multinomial(poss[idx], possession_pmf(float(k)))
        pts[idx] = counts @ np.arange(4)
    return pts


def simulate_game(exp_home: float, exp_away: float, cfg: SimConfig | None = None
                  ) -> SimResult:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_sims
    exp_margin, exp_total = exp_home - exp_away, exp_home + exp_away

    # 1. model uncertainty
    true_margin = exp_margin + rng.normal(0, cfg.rating_sd, n)
    true_total = exp_total + rng.normal(0, cfg.rating_total_sd, n)
    true_home = (true_total + true_margin) / 2
    true_away = (true_total - true_margin) / 2

    # 2-3. possessions
    poss_mean = exp_total / 2 / LEAGUE_PPP
    poss = np.maximum(60, np.round(rng.normal(poss_mean, cfg.pace_sd, n))).astype(np.int64)
    h_raw = _play(rng, poss, true_home / poss_mean)
    a_raw = _play(rng, poss, true_away / poss_mean)

    # 4. calibrate noise around each sim's true mean
    m_dev = (h_raw - a_raw) - (h_raw - a_raw).mean()
    t_dev = (h_raw + a_raw) - (h_raw + a_raw).mean()
    m_dev *= cfg.margin_sd / max(m_dev.std(), 1e-9)
    t_dev *= cfg.total_sd / max(t_dev.std(), 1e-9)
    margin = true_margin + m_dev
    total = true_total + t_dev
    home = np.round((total + margin) / 2).astype(np.int64)
    away = np.round((total - margin) / 2).astype(np.int64)

    # 5. overtime
    went_ot = np.zeros(n, dtype=bool)
    ot_poss = max(8, int(round(poss_mean * 5 / 48)))
    tied = np.nonzero(home == away)[0]
    while len(tied):
        went_ot[tied] = True
        p = np.full(len(tied), ot_poss)
        home[tied] += _play(rng, p, true_home[tied] / poss_mean)
        away[tied] += _play(rng, p, true_away[tied] / poss_mean)
        tied = tied[home[tied] == away[tied]]
    return SimResult(home=home, away=away, went_ot=went_ot)
