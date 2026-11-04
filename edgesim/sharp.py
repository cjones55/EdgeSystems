"""Sharp-money detection from public betting splits and line movement.

Conventions: every market is described from the point of view of "side A"
  spread -> home team (line = home spread, e.g. -4.5)
  total  -> over      (line = total points)
  ml     -> home team (line = home American price)
"Movement toward A" is positive when side A got more expensive since the open,
which is what happens when the market takes money on side A.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .odds import implied_prob


@dataclass
class MarketSplit:
    market: str                 # "spread" | "total" | "ml"
    open_line: float | None
    current_line: float | None
    tickets_a: float | None     # % of bets on side A (0-100)
    money_a: float | None       # % of handle on side A (0-100)


@dataclass
class SharpSignal:
    market: str
    side: str | None            # "A" | "B" | None
    score: float                # 0-100 confidence that the sharp side is ``side``
    reasons: list[str] = field(default_factory=list)


@dataclass
class SharpConfig:
    public_majority: float = 60.0     # tickets % that counts as "public side"
    rlm_min_move: dict = field(default_factory=lambda: {"spread": 0.5, "total": 1.0, "ml": 0.02})
    steam_move: dict = field(default_factory=lambda: {"spread": 1.5, "total": 2.5, "ml": 0.05})
    split_gap: float = 10.0           # money% - tickets% gap that signals big bettors
    w_rlm: float = 45.0
    w_split: float = 30.0
    w_steam: float = 20.0
    w_contrarian: float = 10.0


def movement_toward_a(m: MarketSplit) -> float | None:
    if m.open_line is None or m.current_line is None:
        return None
    if m.market == "spread":
        return m.open_line - m.current_line          # -3 -> -4.5 gives +1.5
    if m.market == "total":
        return m.current_line - m.open_line
    if m.market == "ml":
        return implied_prob(m.current_line) - implied_prob(m.open_line)
    raise ValueError(m.market)


def detect(m: MarketSplit, cfg: SharpConfig | None = None) -> SharpSignal:
    cfg = cfg or SharpConfig()
    votes = {"A": 0.0, "B": 0.0}
    reasons: list[str] = []
    move = movement_toward_a(m)
    unit = "pts" if m.market != "ml" else "prob"

    if m.tickets_a is not None and move is not None:
        public = "A" if m.tickets_a >= cfg.public_majority else (
            "B" if m.tickets_a <= 100 - cfg.public_majority else None)
        if public:
            against = (public == "A" and move <= -cfg.rlm_min_move[m.market]) or \
                      (public == "B" and move >= cfg.rlm_min_move[m.market])
            if against:
                sharp = "B" if public == "A" else "A"
                votes[sharp] += cfg.w_rlm
                pct = m.tickets_a if public == "A" else 100 - m.tickets_a
                reasons.append(f"Reverse line move: {pct:.0f}% of bets on side {public} "
                               f"but line moved {abs(move):.2f} {unit} the other way")

    if m.tickets_a is not None and m.money_a is not None:
        gap = m.money_a - m.tickets_a
        if abs(gap) >= cfg.split_gap:
            side = "A" if gap > 0 else "B"
            votes[side] += cfg.w_split * min(abs(gap) / (2 * cfg.split_gap), 1.0)
            reasons.append(f"Money/ticket split: side {side} has {abs(gap):.0f} pts more "
                           f"of the handle than of the bets")

    if move is not None and abs(move) >= cfg.steam_move[m.market]:
        side = "A" if move > 0 else "B"
        votes[side] += cfg.w_steam
        reasons.append(f"Steam: line moved {abs(move):.2f} {unit} toward side {side}")

    if m.tickets_a is not None and min(m.tickets_a, 100 - m.tickets_a) <= 30:
        side = "A" if m.tickets_a <= 30 else "B"
        votes[side] += cfg.w_contrarian
        reasons.append(f"Contrarian: only {min(m.tickets_a, 100 - m.tickets_a):.0f}% "
                       f"of bets on side {side}")

    net = votes["A"] - votes["B"]
    if net == 0:
        return SharpSignal(m.market, None, 0.0, reasons)
    return SharpSignal(m.market, "A" if net > 0 else "B", min(abs(net), 100.0), reasons)
