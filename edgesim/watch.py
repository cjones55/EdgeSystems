"""Watch mode: re-check the lines on a timer and alert on what matters early.

The backtest's clearest finding is about timing: the side a big line move goes
toward wins 58-60% against the *opening* number but only about 50% against the
close. Being first to a number is worth more than any after-the-fact signal.
So each scan:

  * snapshots the lines (ESPN without an API key, every book with one) so the
    day's first pull is the open and movement can be measured,
  * saves the public betting % (when Action Network is on) so those signals
    build a history that the tracker can grade,
  * raises an alert once per game/market/side for
      NEW PLAY    a side newly clears the card's bar (logged to the tracker),
      EARLY       the sim disagrees with the market by a wide margin and the line
                  has not moved yet: the number to beat before any move
                  (logged as tier "W: early lean" so its CLV gets graded),
      STEAM       the line has moved a full steam threshold since the open,
                  marked with whether the sim agrees (alert only: the backtest
                  says chasing a finished move is break-even),
      SHARP       a reverse line move or money/ticket split with a sharp score
                  of 40+ on one side (alert only).

Alerts print to the terminal, append to reports/alerts_<date>.md, and can pop a
desktop notification (macOS: osascript; Linux: notify-send).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "watch"
REPORTS = ROOT / "reports"

# sim-vs-market gap (probability) that counts as an early lean, per market
EARLY_GAP = {"spread": 0.08, "total": 0.08, "ml": 0.08}
# a line counts as "not moved yet" when it moved less than this toward or away
EARLY_STILL = {"spread": 0.5, "total": 0.5, "ml": 0.015}
# moves bigger than this are almost always a data glitch (side flipped, wrong game)
MAX_MOVE = {"spread": 8.0, "total": 10.0, "ml": 0.25}


@dataclass
class Alert:
    kind: str           # NEW PLAY | EARLY | STEAM | SHARP
    sport: str
    game: str
    pick: str
    price: float | None
    detail: str
    units: int | None = None
    key: str = ""

    def line(self) -> str:
        u = f" [{self.units}u]" if self.units else ""
        p = f" ({self.price:+.0f})" if self.price is not None and not pd.isna(self.price) else ""
        return f"{self.kind:8s} {self.sport.upper():5s} {self.pick}{p}{u}  {self.game}  {self.detail}"


@dataclass
class WatchState:
    seen: set = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "WatchState":
        if path.exists():
            return cls(set(json.loads(path.read_text()).get("seen", [])))
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seen": sorted(self.seen)}))


def _fmt_move(market: str, move: float) -> str:
    return f"{move:+.3f} prob" if market == "ml" else f"{move:+.1f} pts"


def find_alerts(cands: pd.DataFrame, sport: str, steam: dict,
                sharp_min: float = 40.0) -> tuple[list[Alert], pd.DataFrame]:
    """Alerts for one scan. Returns (alerts, early-lean rows to log to the tracker).

    ``cands`` is the card's every-market table (``DayResult.candidates``)."""
    alerts: list[Alert] = []
    early_rows = []
    if cands.empty:
        return alerts, pd.DataFrame()
    for r in cands.itertuples(index=False):
        base = f"{r.game}|{r.market}|{r.side}"
        move = getattr(r, "move", None)
        move = None if move is None or pd.isna(move) else float(move)
        if move is not None and abs(move) > MAX_MOVE[r.market]:
            move = None
        tier = getattr(r, "tier", None)
        units = getattr(r, "units", None)
        units = None if units is None or pd.isna(units) else int(units)
        if isinstance(tier, str) and tier:
            alerts.append(Alert("NEW PLAY", sport, r.game, r.pick, r.price,
                                f"{tier}, EV {r.ev:+.1%}, sim {r.model_prob:.0%} vs market "
                                f"{r.market_prob:.0%}", units, f"play|{base}"))
        gap = r.model_prob - r.market_prob
        still = move is None or abs(move) < EARLY_STILL[r.market]
        if gap >= EARLY_GAP[r.market] and still and not (isinstance(tier, str) and tier):
            alerts.append(Alert("EARLY", sport, r.game, r.pick, r.price,
                                f"sim {r.model_prob:.0%} vs market {r.market_prob:.0%}, line has "
                                "not moved yet (watch for the market to follow)", 1,
                                f"early|{base}"))
            early_rows.append(r._asdict() | {"tier": "W: early lean", "units": 1})
        if move is not None and move >= steam[r.market]:
            agree = "sim AGREES" if gap > 0 else "sim disagrees"
            alerts.append(Alert("STEAM", sport, r.game, r.pick, r.price,
                                f"line moved {_fmt_move(r.market, move)} toward this side since "
                                f"the open; {agree} ({r.model_prob:.0%}). The open was the "
                                "value; only chase if the price still has EV.", None,
                                f"steam|{base}"))
        if getattr(r, "sharp_side", "-") == "this side" and r.sharp_score >= sharp_min:
            alerts.append(Alert("SHARP", sport, r.game, r.pick, r.price,
                                f"score {r.sharp_score:.0f}: {r.sharp_reasons}", None,
                                f"sharp|{base}"))
    return alerts, pd.DataFrame(early_rows)


def notify(title: str, body: str) -> None:
    try:
        if shutil.which("osascript"):
            body_q = body.replace('"', "'")
            subprocess.run(["osascript", "-e",
                            f'display notification "{body_q}" with title "{title}"'],
                           timeout=5, check=False)
        elif shutil.which("notify-send"):
            subprocess.run(["notify-send", title, body], timeout=5, check=False)
    except Exception:
        pass


def games_left(res) -> int:
    """Games that have not started yet (unknown start counts as not started)."""
    now = datetime.now(timezone.utc)
    n = 0
    for g in res.ctx.games:
        try:
            st = pd.Timestamp(g.start)
            st = st.tz_localize("UTC") if st.tzinfo is None else st
            n += st > now
        except Exception:
            n += 1
    return n


def scan(day: date, sport: str, action_network: bool = True, log: bool = True,
         desktop: bool = False) -> tuple[list[Alert], int]:
    """One pass for one sport. Returns (new alerts, games not started yet)."""
    from . import pipeline, tracker
    from .sports import get

    ctx = pipeline.live_context(day, sport=sport, action_network=action_network)
    res = pipeline.run_day(ctx)
    pipeline.write_report(res)
    state_path = STATE_DIR / f"{sport}_{day:%Y-%m-%d}.json"
    state = WatchState.load(state_path)
    alerts, early = find_alerts(res.candidates.assign(tier=_tiers(res)), sport,
                                get(sport).sharp.steam_move)
    new = [a for a in alerts if a.key not in state.seen]
    if log:
        if not res.plays.empty:
            tracker.log_plays(res.plays, day, sport=sport)
        new_early = {a.key.split("|", 1)[1] for a in new if a.kind == "EARLY"}
        if not early.empty and new_early:
            keep = early[(early.game + "|" + early.market + "|" + early.side).isin(new_early)]
            tracker.log_plays(keep, day, sport=sport)
    if new:
        REPORTS.mkdir(parents=True, exist_ok=True)
        with open(REPORTS / f"alerts_{day:%Y-%m-%d}.md", "a") as f:
            stamp = datetime.now().strftime("%H:%M")
            for a in new:
                f.write(f"- {stamp} `{a.kind}` **{a.pick}**"
                        f"{f' ({a.price:+.0f})' if a.price is not None and not pd.isna(a.price) else ''}"
                        f"{f' [{a.units}u]' if a.units else ''} | {a.sport.upper()} {a.game} | {a.detail}\n")
        if desktop:
            for a in new:
                if a.kind in ("NEW PLAY", "EARLY", "STEAM"):
                    notify(f"EdgeSim {a.kind}", f"{a.pick} {a.game}")
    state.seen |= {a.key for a in new}
    state.save(state_path)
    return new, games_left(res)


def _tiers(res) -> pd.Series:
    """Tier per candidate row (candidates carry it already; plays may be filtered)."""
    if "tier" in res.candidates.columns:
        return res.candidates["tier"]
    return pd.Series([None] * len(res.candidates), index=res.candidates.index)


def watch(sports: list[str], every_min: float = 15, until: str | None = None,
          action_network: bool = True, desktop: bool = True, once: bool = False) -> None:
    """Scan every ``every_min`` minutes until ``until`` (HH:MM local) or until no
    game on any watched slate is still to start."""
    stop = None
    if until:
        h, m = map(int, until.split(":"))
        stop = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    while True:
        day = date.today()
        pending = 0
        print(f"\n--- scan {datetime.now():%H:%M} ---")
        for sport in sports:
            try:
                new, left = scan(day, sport, action_network=action_network, desktop=desktop)
            except Exception as e:
                print(f"{sport.upper()}: scan failed ({e})")
                continue
            pending += left
            for a in new:
                print(a.line())
            if not new:
                print(f"{sport.upper()}: nothing new ({left} games still to start)")
        if once or pending == 0 or (stop and datetime.now() >= stop):
            print("Watch finished." if not once else "")
            return
        time.sleep(every_min * 60)
