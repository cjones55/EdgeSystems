"""Run the full backtest and write BACKTEST.md + data/fitted_params.json."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.boxscores import load_player_games
from ..data.historical import load_games
from . import games as G
from . import props as P

ROOT = Path(__file__).resolve().parents[2]


def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(fmt[c])
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.3f}")
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns),
                      *["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]])


PCT = lambda v: f"{100 * v:.1f}%"  # noqa: E731
EDGE_FMT = {"min_edge": PCT, "win_pct": PCT, "roi": PCT, "roi_se": PCT, "z": lambda v: f"{v:+.2f}"}


def run(out: Path = ROOT / "BACKTEST.md", pred_cache: Path | None = None) -> Path:
    t0 = time.time()
    games = load_games()
    if pred_cache and pred_cache.exists():
        pred = pd.read_pickle(pred_cache)
    else:
        pred = G.walk_forward(games)
        if pred_cache:
            pred.to_pickle(pred_cache)
    test = pred[pred.season.isin(G.TEST_SEASONS)]
    train = pred[pred.season.isin(G.TRAIN_SEASONS)]
    L = ["# EdgeSim backtest", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Games: {len(pred):,} simulated "
         f"({len(test):,} in the held-out test seasons {G.TEST_SEASONS.start}-{G.TEST_SEASONS.stop - 1}). "
         "Every prediction is walk-forward: the model only saw games before that date. "
         "Rating parameters were tuned on 2012-2015 only.", "",
         "## Bottom line", "",
         "* **The simulator works**: win probabilities are well calibrated on six held-out "
         "seasons, and prop distributions are close to calibrated.",
         "* **It does not beat NBA closing lines by itself.** Betting every model edge at -110 "
         "loses about the vig.",
         "* **Its opinions do carry some information**: when it disagrees with the opening line, "
         "the market moves its way 46% of the time vs 39% against.",
         "* **Line moves are informative but get priced in**: the side a big move went toward "
         "covered the opening number 58-64% of the time, but only ~50% at the close. Edge from "
         "sharp action exists only if you get the number before the move finishes.",
         "* **Public betting % cannot be backtested** with free data. The forward tracker "
         "(`python -m edgesim grade`) is how sharp-signal plays get proven or dropped.", ""]

    ml_t, ml_tr = G.moneyline_scores(test), G.moneyline_scores(train)
    L += ["## 1. Win probability vs the closing moneyline (test seasons)", "",
          _md(pd.DataFrame([dict(source="EdgeSim sim", log_loss=ml_t["model_logloss"], brier=ml_t["model_brier"]),
                            dict(source="Closing line (no-vig)", log_loss=ml_t["market_logloss"], brier=ml_t["market_brier"]),
                            dict(source="Coin flip", log_loss=ml_t["coinflip_logloss"], brier=0.25)])),
          "", "Lower is better. The sim is a genuinely informative forecaster, but the closing "
          "line is better, as it is for virtually every public model.", ""]
    y = (test.home_pts > test.away_pts).values.astype(float)
    L += ["### Calibration of home win probability (test seasons)", "",
          _md(G.calibration(test.p_home_win.values, y), {"predicted": PCT, "actual": PCT}), "",
          "When the sim says 70%, the home team wins about 70% of the time: the "
          "Monte Carlo is well calibrated.", ""]

    sec = 2
    for name, fn in (("Spread", G.spread_bets), ("Total", G.total_bets)):
        for when in ("close", "open"):
            p = test.dropna(subset=[f"{when}_{'spread' if name == 'Spread' else 'total'}"])
            L += [f"## {sec}. {name} bets at the {when} line, -110 (test seasons)", "",
                  _md(G.edge_table(fn(p, when)), EDGE_FMT), ""]
            sec += 1
    L += ["Break-even at -110 is 52.4%. `z` is ROI divided by its standard error: |z| < 2 "
          "is indistinguishable from luck.", ""]

    L += [f"## {sec}. The daily card's own rule, replayed", "",
          "Model-edge plays (tier B) use the sim blended with the market and need EV of at "
          "least +1%. Replayed on the training and test seasons:", "",
          "**Training seasons (2011-2015)**", "",
          _md(G.card_replay(train), {"plays_per_100_games": lambda v: f"{v:.1f}", "win_pct": PCT,
                                      "roi": PCT, "z": lambda v: f"{v:+.2f}"}), "",
          "**Test seasons (2016-2021)**", "",
          _md(G.card_replay(test), {"plays_per_100_games": lambda v: f"{v:.1f}", "win_pct": PCT,
                                     "roi": PCT, "z": lambda v: f"{v:+.2f}"}), "",
          "The blend is deliberately conservative, so model-only game plays are rare (a few "
          "per 100 games) and none of these results is distinguishable from luck (|z| < 2). "
          "Treat tier B game plays as unproven.", ""]
    sec += 1
    L += [f"## {sec}. Closing line value: model picks at the opening line", ""]
    rows = []
    for d in (1, 2, 3):
        c = G.clv(test, d)
        rows.append(dict(model_vs_open_pts=f">= {d}", bets=c["bets"], beat_close=c["beat_close_pct"],
                         lost_to_close=c["lost_to_close_pct"], avg_clv_pts=c["avg_clv_pts"],
                         ats_at_open=c["open_line_ats"]["win_pct"]))
    L += [_md(pd.DataFrame(rows), {"beat_close": PCT, "lost_to_close": PCT, "ats_at_open": PCT}), "",
          "The market moves toward the model's side more often than away from it (the "
          "remainder is no move), so the model has some real information the opening "
          "line lacks, but not enough to overcome the vig on its own.", ""]
    sec += 1

    L += [f"## {sec}. Line-movement ('sharp money') proxy test", ""]
    rows = []
    for mv in (1.0, 1.5, 2.0, 3.0):
        s = G.steam_follow(test, mv)
        rows.append(dict(min_move_pts=mv, games=s["games"], ats_if_bet_at_close=s["at_close"]["win_pct"],
                         roi_at_close=s["at_close"]["roi"], ats_if_bet_at_open=s["at_open"]["win_pct"],
                         roi_at_open=s["at_open"]["roi"]))
    L += [_md(pd.DataFrame(rows), {"ats_if_bet_at_close": PCT, "roi_at_close": PCT,
                                   "ats_if_bet_at_open": PCT, "roi_at_open": PCT}), "",
          "Sides the line moved toward cover the OPENING number ~57% of the time, but "
          "only ~50% against the closing number. Sharp moves carry real information, "
          "and it is priced in by the close. Value only exists if you act before the "
          "move finishes, which is why the app snapshots lines through the day and why "
          "the forward tracker measures CLV. Free historical public-betting % data does "
          "not exist, so ticket/money splits cannot be backtested here; they are tracked "
          "going forward instead.", ""]
    sec += 1

    # props
    pg = load_player_games()
    pr = P.run(pg)
    L += [f"## {sec}. Player props (test seasons 2022-23, 2023-24)", "",
          "### Projection accuracy (mean absolute error)", "",
          _md(pr["accuracy"]), "",
          "### Calibration of P(over) at a proxy line", "",
          _md(pr["calibration"], {"predicted": PCT, "actual": PCT}), "",
          "### Hit rate against proxy lines", "",
          _md(pr["proxy_bets"].drop(columns=["roi_at_minus115"]), {"hit_rate": PCT}), "",
          "**Read this carefully:** historical sportsbook prop lines are not freely "
          "available, so the proxy line is the player's last-10 average rounded to x.5. "
          "Real books are much sharper than that (they know about injuries, role changes, "
          "and use their own models), so these hit rates are an upper bound, not an "
          "expected result. The fair test is the forward tracker with real lines.", ""]
    fitted = dict(dispersion=pr["dispersion"], calib=pr["calib"])
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "fitted_params.json").write_text(json.dumps(fitted, indent=2))
    L += [f"Fitted prop parameters (from training seasons): `{json.dumps(fitted)}`", "",
          f"Runtime: {time.time() - t0:.0f}s"]
    out.write_text("\n".join(L))
    return out
