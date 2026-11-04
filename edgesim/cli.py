"""Command line entry point:  python -m edgesim <command> [--sport nba|nfl|nhl|ncaaf|ncaab|all]

  today      build today's card from live data (ESPN + Odds API + your splits)
  demo       run the full pipeline on an archived date, offline
  sim        simulate one matchup with current ratings
  backtest   re-run the historical backtests and rewrite BACKTEST.md + backtests/
  watch      re-check lines every N minutes and alert on new plays, early leans, steam
  grade      grade logged plays (results + closing line value)
  doctor     check which data sources are reachable from this machine
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPORT_CHOICES = ["nba", "nfl", "nhl", "ncaaf", "ncaab"]
DEMO_DATES = {"nba": "2022-01-17", "nfl": "2021-11-14", "nhl": "2022-01-15",
              "ncaaf": "2025-11-01", "ncaab": "2022-02-12"}


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _sports(arg: str) -> list[str]:
    return SPORT_CHOICES if arg == "all" else [arg]


def _show(res, out) -> None:
    from .sports import get
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"\n######## {get(res.ctx.sport).label} - {res.ctx.day} ########")
    for n in res.ctx.notes:
        print(f"NOTE: {n}")
    print("\n=== GAME PLAYS ===")
    if res.plays.empty:
        print("None cleared the bar today.")
    else:
        print(res.plays[["units", "tier", "game", "pick", "price", "book", "blended_prob", "ev", "kelly",
                         "sharp_score"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not res.props.empty and res.props.play.any():
        print("\n=== PROP PLAYS (top 15) ===")
        print(res.props[res.props.play].head(15)[
            ["units", "player", "stat", "pick", "price", "projection", "blended_prob", "ev"]
        ].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nFull report: {out}")


def main(argv=None) -> int:
    _load_env()
    ap = argparse.ArgumentParser(prog="edgesim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("today")
    t.add_argument("--sport", default="nba", choices=SPORT_CHOICES + ["all"])
    t.add_argument("--date", default=None)
    t.add_argument("--props", action="store_true", help="price NBA player props (uses Odds API credits)")
    t.add_argument("--action-network", action="store_true", help="try the experimental splits feed")
    t.add_argument("--no-log", action="store_true")
    d = sub.add_parser("demo")
    d.add_argument("--sport", default="nba", choices=SPORT_CHOICES)
    d.add_argument("--date", default=None)
    s = sub.add_parser("sim")
    s.add_argument("away")
    s.add_argument("home")
    s.add_argument("--sport", default="nba", choices=SPORT_CHOICES)
    s.add_argument("--spread", type=float)
    s.add_argument("--total", type=float)
    s.add_argument("--neutral", action="store_true")
    s.add_argument("--demo-date", default=None, help="use ratings as of an archived date")
    b = sub.add_parser("backtest")
    b.add_argument("--sport", default="all", choices=SPORT_CHOICES + ["all"])
    b.add_argument("--cache", default=None, help="folder to cache simulated predictions in")
    w = sub.add_parser("watch")
    w.add_argument("--sport", default="all", choices=SPORT_CHOICES + ["all"])
    w.add_argument("--every", type=float, default=15, help="minutes between scans")
    w.add_argument("--until", default=None, help="stop at this local time, HH:MM")
    w.add_argument("--no-action-network", action="store_true")
    w.add_argument("--no-desktop", action="store_true", help="no desktop notifications")
    w.add_argument("--once", action="store_true", help="one scan, then exit")
    sub.add_parser("grade")
    sub.add_parser("doctor")
    a = ap.parse_args(argv)

    from . import pipeline, tracker

    if a.cmd == "today":
        day = date.fromisoformat(a.date) if a.date else date.today()
        for sport in _sports(a.sport):
            try:
                ctx = pipeline.live_context(day, with_props=a.props, sport=sport,
                                            action_network=a.action_network)
            except Exception as e:
                print(f"{sport.upper()}: could not build the card ({e})")
                continue
            res = pipeline.run_day(ctx)
            out = pipeline.write_report(res)
            if not a.no_log:
                tracker.log_plays(res.plays, day, sport=sport)
            _show(res, out)
    elif a.cmd == "demo":
        res = pipeline.run_day(pipeline.demo_context(a.date or DEMO_DATES[a.sport], sport=a.sport))
        _show(res, pipeline.write_report(res))
    elif a.cmd == "sim":
        from .resolve import resolve
        from .sports import get, load_history
        sp = get(a.sport)
        home, away = resolve(a.home, a.sport), resolve(a.away, a.sport)
        if not home or not away:
            print("Unknown team")
            return 2
        if a.demo_date:
            r = pipeline.fit_ratings(load_history(a.sport), pd.Timestamp(a.demo_date), a.sport)
        else:
            today = date.today()
            r = pipeline.fit_ratings(pipeline.live_results(today, a.sport), pd.Timestamp(today), a.sport)
        eh, ea = r.expected(home, away, neutral=a.neutral)
        sim = sp.sim(eh, ea)
        sm = sim.summary()
        print(f"{a.away} {'vs' if a.neutral else '@'} {a.home} ({sp.label}): projected "
              f"{sm['away_pts']:.1f} - {sm['home_pts']:.1f}")
        print(f"  {a.home} win {sm['home_win']:.1%} | fair {sp.spread_label.lower()} "
              f"{sm['fair_spread']:+.1f} | fair total {sm['fair_total']:.1f} | OT {sm['ot_rate']:.1%}")
        if a.spread is not None:
            c, p = sim.p_cover(a.spread)
            print(f"  {a.home} {a.spread:+g}: cover {c:.1%}, push {p:.1%}")
        if a.total is not None:
            o, p = sim.p_over(a.total)
            print(f"  Total {a.total:g}: over {o:.1%}, push {p:.1%}")
    elif a.cmd == "backtest":
        from .backtest.report import run
        cache = Path(a.cache) if a.cache else None
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
        print(f"Wrote {run(_sports(a.sport), cache_dir=cache)}")
    elif a.cmd == "watch":
        from .watch import watch
        watch(_sports(a.sport), every_min=a.every, until=a.until,
              action_network=not a.no_action_network, desktop=not a.no_desktop, once=a.once)
    elif a.cmd == "grade":
        log = tracker.grade()
        print(tracker.summary(log).to_string(index=False) if not log.empty else "No plays logged yet.")
    elif a.cmd == "doctor":
        import requests
        from .sports import SPORTS
        key = os.environ.get("ODDS_API_KEY", "missing")
        for k, sp in SPORTS.items():
            for name, url in ((f"ESPN {sp.label}", f"https://site.api.espn.com/apis/site/v2/sports/{sp.espn_path}/scoreboard"),
                              (f"Odds API {sp.label}", f"https://api.the-odds-api.com/v4/sports/{sp.odds_key}/events?apiKey={key}")):
                try:
                    code = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).status_code
                    print(f"{name:36s} HTTP {code}")
                except Exception as e:
                    print(f"{name:36s} FAILED ({type(e).__name__})")
        print(f"ODDS_API_KEY set: {'yes' if os.environ.get('ODDS_API_KEY') else 'no'}")
        for f in ("nba_archive_10Y.json", "nfl_archive_10Y.json", "nhl_archive_10Y.json",
                  "ncaaf", "ncaab"):
            print(f"Historical data {f:24s} {'present' if (ROOT / 'data/raw' / f).exists() else 'MISSING (run scripts/fetch_data.py)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
