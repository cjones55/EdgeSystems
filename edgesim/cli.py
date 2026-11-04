"""Command line entry point:  python -m edgesim <command>

  today      build today's card from live data (ESPN + Odds API + your splits)
  demo       run the full pipeline on an archived date, offline
  sim        simulate one matchup with current ratings
  backtest   re-run the full historical backtest and rewrite BACKTEST.md
  grade      grade logged plays (results + closing line value)
  doctor     check which data sources are reachable from this machine
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _show(res, out) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for n in res.ctx.notes:
        print(f"NOTE: {n}")
    print("\n=== GAME PLAYS ===")
    if res.plays.empty:
        print("None cleared the bar today.")
    else:
        print(res.plays[["tier", "game", "pick", "price", "book", "blended_prob", "ev", "kelly",
                         "sharp_score"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not res.props.empty and res.props.play.any():
        print("\n=== PROP PLAYS (top 15) ===")
        print(res.props[res.props.play].head(15)[
            ["player", "stat", "pick", "price", "projection", "blended_prob", "ev"]
        ].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nFull report: {out}")


def main(argv=None) -> int:
    _load_env()
    ap = argparse.ArgumentParser(prog="edgesim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("today")
    t.add_argument("--date", default=None)
    t.add_argument("--props", action="store_true", help="price player props (uses Odds API credits)")
    t.add_argument("--action-network", action="store_true", help="try the experimental splits feed")
    t.add_argument("--no-log", action="store_true")
    d = sub.add_parser("demo")
    d.add_argument("--date", default="2022-01-17")
    s = sub.add_parser("sim")
    s.add_argument("away")
    s.add_argument("home")
    s.add_argument("--spread", type=float)
    s.add_argument("--total", type=float)
    s.add_argument("--demo-date", default=None, help="use ratings as of an archived date")
    b = sub.add_parser("backtest")
    b.add_argument("--cache", default=None)
    sub.add_parser("grade")
    sub.add_parser("doctor")
    a = ap.parse_args(argv)

    from . import pipeline, tracker

    if a.cmd == "today":
        day = date.fromisoformat(a.date) if a.date else date.today()
        ctx = pipeline.live_context(day, with_props=a.props, action_network=a.action_network)
        res = pipeline.run_day(ctx)
        out = pipeline.write_report(res)
        if not a.no_log:
            tracker.log_plays(res.plays, day)
        _show(res, out)
    elif a.cmd == "demo":
        res = pipeline.run_day(pipeline.demo_context(a.date))
        _show(res, pipeline.write_report(res))
    elif a.cmd == "sim":
        from .gamesim import simulate_game
        from .teams import to_code
        home, away = to_code(a.home), to_code(a.away)
        if not home or not away:
            print("Unknown team")
            return 2
        if a.demo_date:
            from .data.historical import load_games
            r = pipeline.fit_ratings(load_games(), pd.Timestamp(a.demo_date))
        else:
            from datetime import timedelta
            from .data.historical import add_rest
            from .providers import espn
            today = date.today()
            st = espn.season_start(today)
            res = add_rest(espn.results(date(st.year - 1, 10, 1), today - timedelta(days=1)))
            r = pipeline.fit_ratings(res, pd.Timestamp(today))
        eh, ea = r.expected(home, away)
        sim = simulate_game(eh, ea)
        sm = sim.summary()
        print(f"{away} @ {home}: projected {sm['away_pts']:.1f} - {sm['home_pts']:.1f}")
        print(f"  {home} win {sm['home_win']:.1%} | fair spread {home} {sm['fair_spread']:+.1f} | "
              f"fair total {sm['fair_total']:.1f} | OT {sm['ot_rate']:.1%}")
        if a.spread is not None:
            c, p = sim.p_cover(a.spread)
            print(f"  {home} {a.spread:+g}: cover {c:.1%}, push {p:.1%}")
        if a.total is not None:
            o, p = sim.p_over(a.total)
            print(f"  Total {a.total:g}: over {o:.1%}, push {p:.1%}")
    elif a.cmd == "backtest":
        from .backtest.report import run
        print(f"Wrote {run(pred_cache=Path(a.cache) if a.cache else None)}")
    elif a.cmd == "grade":
        log = tracker.grade()
        print(tracker.summary(log).to_string(index=False) if not log.empty else "No plays logged yet.")
    elif a.cmd == "doctor":
        import requests
        checks = {
            "ESPN scoreboard": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            "The Odds API": "https://api.the-odds-api.com/v4/sports?apiKey=" + os.environ.get("ODDS_API_KEY", "missing"),
            "Action Network (experimental)": "https://api.actionnetwork.com/web/v2/scoreboard/nba",
        }
        for name, url in checks.items():
            try:
                code = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).status_code
                print(f"{name:32s} HTTP {code}")
            except Exception as e:
                print(f"{name:32s} FAILED ({type(e).__name__})")
        print(f"ODDS_API_KEY set: {'yes' if os.environ.get('ODDS_API_KEY') else 'no'}")
        print(f"Historical data present: {(ROOT / 'data/raw/nba_archive_10Y.json').exists()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
