"""Odds math: conversions, vig removal, expected value, Kelly sizing."""
from __future__ import annotations

import math
from typing import Sequence


def american_to_decimal(american: float) -> float:
    if american == 0:
        raise ValueError("American odds cannot be 0")
    return 1 + (american / 100.0 if american > 0 else 100.0 / -american)


def decimal_to_american(dec: float) -> float:
    if dec <= 1:
        raise ValueError("Decimal odds must be > 1")
    return (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)


def implied_prob(american: float) -> float:
    """Raw implied probability (includes the bookmaker's margin)."""
    return 1.0 / american_to_decimal(american)


def prob_to_american(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError("probability must be in (0, 1)")
    return decimal_to_american(1 / p)


def no_vig_probs(prices: Sequence[float]) -> list[float]:
    """Remove the vig by proportional normalisation of implied probabilities."""
    raw = [implied_prob(p) for p in prices]
    s = sum(raw)
    return [r / s for r in raw]


def expected_value(p_win: float, american: float, p_push: float = 0.0) -> float:
    """EV per 1 unit staked. Pushes return the stake."""
    payout = american_to_decimal(american) - 1
    p_loss = 1 - p_win - p_push
    return p_win * payout - p_loss


def kelly_fraction(p_win: float, american: float, p_push: float = 0.0,
                   fraction: float = 0.25, cap: float = 0.03) -> float:
    """Fractional Kelly stake as a share of bankroll (capped). 0 when -EV."""
    b = american_to_decimal(american) - 1
    # with pushes, condition on decision (push returns stake)
    decided = 1 - p_push
    if decided <= 0:
        return 0.0
    p = p_win / decided
    f = (b * p - (1 - p)) / b
    return max(0.0, min(cap, f * fraction))


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
