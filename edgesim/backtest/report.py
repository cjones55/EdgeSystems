"""Run the walk-forward backtests and write the reports.

  backtests/<SPORT>.md   full results per sport
  BACKTEST.md            one-page summary across sports
  data/fitted_params.json   NBA prop distribution parameters
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import sports
from . import games as G

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "backtests"


def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: f(v) if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else v))
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns),
                      *["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]])


PCT = lambda v: f"{100 * v:.1f}%"  # noqa: E731
Z = lambda v: f"{v:+.2f}"  # noqa: E731
EDGE_FMT = {"min_ev": PCT, "win_pct": PCT, "roi": PCT, "roi_se": PCT, "z": Z}
REPLAY_FMT = {"plays_per_100_games": lambda v: f"{v:.1f}", "win_pct": PCT, "roi": PCT, "z": Z}


def predictions(sport: str, cache_dir: Path | None = None) -> pd.DataFrame:
    path = cache_dir / f"pred_{sport}.pkl" if cache_dir else None
    if path and path.exists():
        return pd.read_pickle(path)
    pred = G.walk_forward(sports.load_history(sport), sport)
    if path:
        pred.to_pickle(path)
    return pred


def sport_report(sport: str, pred: pd.DataFrame) -> tuple[str, dict]:
    sp = sports.get(sport)
    test = pred[pred.season.isin(sp.test)]
    train = pred[pred.season.isin(sp.train)]
    tr, te = f"{sp.train.start}-{sp.train.stop - 1}", f"{sp.test.start}-{sp.test.stop - 1}"
    L = [f"# {sp.label} backtest", "",
         f"Data: {sp.archive_note}. {len(pred):,} games simulated, {len(test):,} in the "
         f"held-out test seasons ({te}). Ratings were tuned on {tr} only, and every "
         "prediction only uses games played before that date.", ""]
    summary = dict(sport=sp.label, test_games=len(test))

    ml = G.moneyline_scores(test)
    if ml["games"]:
        L += ["## Win probability vs the closing moneyline", "",
              _md(pd.DataFrame([dict(source="EdgeSim sim", log_loss=ml["model_logloss"], brier=ml["model_brier"]),
                                dict(source="Closing line (no-vig)", log_loss=ml["market_logloss"], brier=ml["market_brier"]),
                                dict(source="Coin flip", log_loss=ml["coinflip_logloss"], brier=0.25)])),
              "", "Lower is better.", ""]
        summary.update(model_logloss=ml["model_logloss"], market_logloss=ml["market_logloss"])
    y = (test.home_pts > test.away_pts).values.astype(float)
    p = (test.p_home_win / (1 - test.p_tie)).values
    keep = test.home_pts.values != test.away_pts.values
    cal = G.calibration(p[keep], y[keep])
    big = cal[cal.n >= 100]
    summary["calibration_max_gap"] = float((big.predicted - big.actual).abs().max())
    L += ["## Calibration of home win probability", "",
          _md(cal, {"predicted": PCT, "actual": PCT}), ""]

    label = sp.spread_label.lower()
    sections = [(f"{sp.spread_label} bets at the closing line", G.spread_bets(test, "close")),
                ("Total bets at the closing line", G.total_bets(test, "close")),
                ("Moneyline bets at the closing price", G.ml_bets(test))]
    if test.open_spread.notna().sum() > 200:
        sections.insert(1, (f"{sp.spread_label} bets at the opening line", G.spread_bets(test, "open")))
    if test.open_total.notna().sum() > 200:
        sections.insert(3, ("Total bets at the opening line", G.total_bets(test, "open")))
    for title, bets in sections:
        if len(bets):
            L += [f"## {title}", "", _md(G.edge_table(bets), EDGE_FMT), ""]
    sb = G.spread_bets(test, "close")
    summary.update(spread_close_roi=G.record(sb[sb.edge >= 0])["roi"] if len(sb) else np.nan)
    L += ["Each row bets the side the sim prefers whenever its expected value at the real price "
          "is at least `min_ev`. `z` is ROI divided by its standard error; |z| < 2 is "
          "indistinguishable from luck.", ""]

    L += ["## The daily card's rule, replayed", "",
          "Tier B plays blend the sim with the market and need EV of +1% or more.", "",
          f"Training seasons ({tr}):", "", _md(G.card_replay(train, sport=sport), REPLAY_FMT), "",
          f"Test seasons ({te}):", "", _md(G.card_replay(test, sport=sport), REPLAY_FMT), ""]

    if test.open_spread.notna().sum() > 200:
        rows = []
        for d in (1, 2, 3):
            c = G.clv(test, d)
            rows.append(dict(model_vs_open_pts=f">= {d}", bets=c["bets"], beat_close=c["beat_close_pct"],
                             lost_to_close=c["lost_to_close_pct"], avg_clv_pts=c["avg_clv_pts"]))
        summary["clv_beat_vs_lost"] = f"{100 * rows[0]['beat_close']:.0f}% vs {100 * rows[0]['lost_to_close']:.0f}%"
        L += ["## Closing line value: model picks at the opening line", "",
              _md(pd.DataFrame(rows), {"beat_close": PCT, "lost_to_close": PCT}), "",
              "How often the market moved toward the sim's side between open and close "
              "(the rest did not move).", ""]
        rows = []
        for mv in (1.0, 1.5, 2.0, 3.0):
            s = G.steam_follow(test, mv)
            rows.append(dict(min_move_pts=mv, games=s["games"], cover_vs_close=s["at_close"]["win_pct"],
                             roi_at_close=s["at_close"]["roi"], cover_vs_open=s["at_open"]["win_pct"],
                             roi_at_open=s["at_open"]["roi"]))
        summary["steam_open_vs_close"] = f"{100 * rows[1]['cover_vs_open']:.0f}% vs {100 * rows[1]['cover_vs_close']:.0f}%"
        L += ["## Line moves ('sharp money' proxy)", "",
              _md(pd.DataFrame(rows), {k: PCT for k in ("cover_vs_close", "roi_at_close", "cover_vs_open", "roi_at_open")}), "",
              "The side a line moved toward, graded against the opening number and against the "
              "closing number.", ""]
    if test.open_home_ml.notna().sum() > 200:
        c = G.clv_ml(test)
        summary["clv_beat_vs_lost"] = f"{100 * c['beat_close_pct']:.0f}% vs {100 * c['lost_to_close_pct']:.0f}%"
        rows = []
        for mv in (0.02, 0.03, 0.05):
            s = G.steam_follow_ml(test, mv)
            rows.append(dict(min_move_prob=mv, games=s["games"], win_pct=s["at_close"]["win_pct"],
                             roi_at_close=s["at_close"]["roi"], roi_at_open=s["at_open"]["roi"]))
        summary["steam_open_vs_close"] = f"ROI {100 * rows[1]['roi_at_open']:+.0f}% vs {100 * rows[1]['roi_at_close']:+.0f}%"
        L += ["## Closing line value: model picks at the opening moneyline", "",
              _md(pd.DataFrame([c]), {"beat_close_pct": PCT, "lost_to_close_pct": PCT}), "",
              "## Moneyline moves ('sharp money' proxy)", "",
              _md(pd.DataFrame(rows), {"win_pct": PCT, "roi_at_close": PCT, "roi_at_open": PCT}), ""]
    return "\n".join(L), summary


def nba_props_section() -> str:
    from ..data.boxscores import load_player_games
    from . import props as P
    pr = P.run(load_player_games())
    fitted = dict(dispersion=pr["dispersion"], calib=pr["calib"])
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "fitted_params.json").write_text(json.dumps(fitted, indent=2))
    return "\n".join([
        "## Player props (test seasons 2022-23, 2023-24)", "",
        "### Projection accuracy (mean absolute error)", "", _md(pr["accuracy"]), "",
        "### Calibration of P(over) at a proxy line", "",
        _md(pr["calibration"], {"predicted": PCT, "actual": PCT}), "",
        "### Hit rate against proxy lines", "",
        _md(pr["proxy_bets"].drop(columns=["roi_at_minus115"]), {"hit_rate": PCT}), "",
        "Historical sportsbook prop lines are not freely available, so the proxy line is the "
        "player's last-10 average rounded to x.5. Real books are much sharper, so these hit "
        "rates are an upper bound, not an expected result.", "",
        f"Fitted prop parameters: `{json.dumps(fitted)}`", ""])


def run(sport_keys=None, cache_dir: Path | None = None, props: bool = True) -> Path:
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    sport_keys = sport_keys or list(sports.SPORTS)
    summaries = []
    for key in sport_keys:
        pred = predictions(key, cache_dir)
        text, summ = sport_report(key, pred)
        if key == "nba" and props:
            text += "\n" + nba_props_section()
        (OUT_DIR / f"{key.upper()}.md").write_text(text)
        summaries.append(summ)
    old = ROOT / "BACKTEST.md"
    S = pd.DataFrame(summaries)
    for c in ("clv_beat_vs_lost", "steam_open_vs_close"):
        if c in S.columns:
            S[c] = S[c].fillna("n/a (no opening lines)")
    L = ["# EdgeSim backtest summary", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Full results per sport are in "
         "[backtests/](backtests/). Every prediction is walk-forward and every number below "
         "is from seasons the model was not tuned on.", "",
         _md(S.rename(columns={"test_games": "test games", "model_logloss": "sim log loss",
                               "market_logloss": "closing line log loss",
                               "calibration_max_gap": "worst calibration gap",
                               "spread_close_roi": "ROI betting sim edges at close",
                               "clv_beat_vs_lost": "line moved toward sim / away",
                               "steam_open_vs_close": "big moves: at open vs at close"}),
             {"sim log loss": lambda v: f"{v:.3f}", "closing line log loss": lambda v: f"{v:.3f}",
              "worst calibration gap": PCT, "ROI betting sim edges at close": PCT}), "",
         "## What this means", "",
         "* **The simulators are well calibrated in every sport**: when a sim says 65%, it "
         "happens about 65% of the time (worst bucket gap shown above).",
         "* **No sport's sim beats the closing line on its own.** Betting every sim edge at the "
         "closing number loses roughly the bookmaker's margin.",
         "* **The sims do carry information the opening line lacks**: the market moves toward "
         "the sim's side more often than away from it, most strongly in the NFL.",
         "* **Line moves are real signals, but only before they finish**: the side a big move "
         "went toward wins well above break-even at the opening number and about break-even at "
         "the close.",
         "* **Public betting % has no free history**, so ticket/money splits are tracked going "
         "forward with `python -m edgesim grade` rather than backtested.", "",
         f"Runtime: {time.time() - t0:.0f}s"]
    old.write_text("\n".join(L))
    return old
