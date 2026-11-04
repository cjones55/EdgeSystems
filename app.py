"""EdgeSim local dashboard.  Run:  streamlit run app.py"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from edgesim import pipeline, tracker
from edgesim.cli import DEMO_DATES, _load_env
from edgesim.providers import public_betting
from edgesim.props import COMBOS, simulate_player
from edgesim.resolve import display
from edgesim.sports import SPORTS

ROOT = Path(__file__).resolve().parent
_load_env()
st.set_page_config(page_title="EdgeSim", layout="wide")
st.title("EdgeSim")

PCT_COLS = ["model_prob", "market_prob", "blended_prob", "home_win"]


def fmt(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    f = {c: "{:.1%}" for c in PCT_COLS if c in df.columns}
    f.update({c: "{:+.1%}" for c in ("ev",) if c in df.columns})
    f.update({c: "{:.2%}" for c in ("kelly",) if c in df.columns})
    f.update({c: "{:.1f}" for c in ("proj_home", "proj_away", "projection", "proj_minutes",
                                    "sharp_score", "fair_total", "market_total")
              if c in df.columns})
    f.update({c: "{:+.1f}" for c in ("fair_spread", "market_spread") if c in df.columns})
    f.update({c: "{:+.0f}" for c in ("price",) if c in df.columns})
    f.update({c: "{:.0f}%" for c in ("tickets_pct", "money_pct") if c in df.columns})
    f.update({c: "{:.0f}u" for c in ("units",) if c in df.columns})
    return df.style.format(f, na_rep="")


with st.sidebar:
    sport = st.selectbox("Sport", list(SPORTS), format_func=lambda k: SPORTS[k].label)
    mode = st.radio("Data", ["Live (today)", "Demo (archived date)"])
    if mode.startswith("Live"):
        day = st.date_input("Date", date.today())
        with_props = SPORTS[sport].has_props and st.checkbox(
            "Price player props (uses Odds API credits)", False)
        an = st.checkbox("Try Action Network splits (experimental)", False)
    else:
        day = st.date_input("Archived date", date.fromisoformat(DEMO_DATES[sport]))
        st.caption(f"Archive: {SPORTS[sport].archive_note}")
    run = st.button("Run sims", type="primary")
    st.caption("Bets are tiered A (model + sharp agree), B (model edge), C (sharp lean). "
               "Units: 1, 2, 3, 5 (max), 10 (super max). "
               "See the Backtest tab for what has and has not been proven.")

if run:
    with st.spinner("Loading data and simulating..."):
        try:
            if mode.startswith("Live"):
                ctx = pipeline.live_context(day, with_props=with_props, action_network=an,
                                            sport=sport)
            else:
                ctx = pipeline.demo_context(day, sport=sport)
            st.session_state.res = pipeline.run_day(ctx)
            pipeline.write_report(st.session_state.res)
        except Exception as e:  # show the problem instead of a stack trace
            st.error(f"Could not build the card: {e}")

res = st.session_state.get("res")
if res is not None and res.ctx.sport != sport:
    res = None
tabs = st.tabs(["Today's card", "Game simulator", "Props", "Public betting %", "Backtest",
                "Tracker", "Alerts"])

# ---------------------------------------------------------------- card
with tabs[0]:
    if res is None:
        st.info("Pick a mode in the sidebar and press **Run sims**.")
    else:
        for n in res.ctx.notes:
            st.caption(n)
        st.subheader(f"{SPORTS[res.ctx.sport].label} plays for {res.ctx.day}")
        if res.plays.empty:
            st.write("Nothing cleared the bar. That is the expected outcome on most days: NBA "
                     "lines are efficient and the system is built not to force bets.")
        else:
            st.dataframe(fmt(res.plays[["units", "tier", "game", "pick", "price", "book", "blended_prob",
                                        "market_prob", "model_prob", "ev", "kelly", "sharp_score",
                                        "tickets_pct", "money_pct", "sharp_reasons"]]),
                         use_container_width=True, hide_index=True)
            if res.ctx.source == "live" and st.button("Log these plays to the tracker"):
                n = tracker.log_plays(res.plays, res.ctx.day, sport=res.ctx.sport)
                st.success(f"Tracker now holds {n} plays.")
        st.subheader("Game simulations")
        st.dataframe(fmt(res.sims.drop(columns=["home", "away"], errors="ignore")), use_container_width=True,
                     hide_index=True)
        with st.expander("Every market and side (including non-plays)"):
            st.dataframe(fmt(res.candidates.drop(columns=["home", "away", "side", "line"])),
                         use_container_width=True, hide_index=True)
        if not res.props.empty:
            st.subheader("Prop plays")
            st.dataframe(fmt(res.props[res.props.play].head(40)), use_container_width=True,
                         hide_index=True)

# ---------------------------------------------------------------- simulator
with tabs[1]:
    if res is None:
        st.info("Run the card first so the simulator uses the same ratings.")
    else:
        spc = SPORTS[res.ctx.sport]
        keys = sorted(set(res.ctx.ratings.off) | {g.home for g in res.ctx.games} | {g.away for g in res.ctx.games},
                      key=lambda k: display(k, res.ctx.sport))
        name = {k: display(k, res.ctx.sport) for k in keys}
        g0 = res.ctx.games[0] if res.ctx.games else None
        c1, c2, c3, c4 = st.columns(4)
        away = c1.selectbox("Away", keys, index=keys.index(g0.away) if g0 and g0.away in keys else 0,
                            format_func=name.get)
        home = c2.selectbox("Home", keys, index=keys.index(g0.home) if g0 and g0.home in keys else 1,
                            format_func=name.get)
        spread = c3.number_input(f"Home {spc.spread_label.lower()}",
                                 value=-1.5 if res.ctx.sport == "nhl" else -2.5, step=0.5)
        total = c4.number_input("Total", value=float(round(res.ctx.ratings.league_avg * 2) / 2 + 0.5), step=0.5)
        neutral = st.checkbox("Neutral site", False)
        eh, ea = res.ctx.ratings.expected(home, away, neutral=neutral)
        sim = spc.sim(eh, ea)
        s = sim.summary()
        m = st.columns(5)
        m[0].metric(f"{name[home]} win", f"{s['home_win']:.1%}")
        m[1].metric(f"Fair {spc.spread_label.lower()}", f"{s['fair_spread']:+.1f}")
        m[2].metric("Fair total", f"{s['fair_total']:.1f}")
        cov, push = sim.p_cover(spread)
        m[3].metric(f"{name[home]} {spread:+g} covers", f"{cov:.1%}", f"push {push:.1%}", delta_color="off")
        ov, push = sim.p_over(total)
        m[4].metric(f"Over {total:g}", f"{ov:.1%}", f"push {push:.1%}", delta_color="off")
        h1, h2 = st.columns(2)
        h1.caption(f"{name[home]} margin distribution")
        h1.bar_chart(pd.Series(sim.margin).value_counts().sort_index())
        h2.caption("Total distribution")
        h2.bar_chart(pd.Series(sim.total).value_counts().sort_index())

# ---------------------------------------------------------------- props
with tabs[2]:
    feat = res.ctx.player_features if res is not None else pd.DataFrame()
    if feat.empty:
        st.info("Run a demo date, or a live date with props enabled, to load player projections.")
    else:
        player = st.selectbox("Player", sorted(feat.player.unique()))
        x = feat[feat.player == player].iloc[-1]
        stat = st.selectbox("Stat", ["pts", "reb", "ast", "fg3m", *COMBOS])
        rates = {s: x[f"rate_{s}"] for s in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]}
        opp = {s: x[f"opp_{s}"] for s in rates}
        sim = simulate_player(rates, x.proj_min, x.min_sd, n=20000, opp=opp, seed=1)
        line = st.number_input("Line", value=float(np.floor(sim[stat].mean()) + 0.5), step=0.5)
        a, b, c = st.columns(3)
        a.metric("Projection", f"{sim[stat].mean():.1f}")
        b.metric(f"Over {line:g}", f"{(sim[stat] > line).mean():.1%}")
        c.metric("Projected minutes", f"{x.proj_min:.1f} ± {x.min_sd:.1f}")
        st.bar_chart(sim[stat].value_counts().sort_index())
        st.caption("Uncalibrated simulation view. The card's prop prices also apply the fitted "
                   "calibration and blend with the market.")

# ---------------------------------------------------------------- splits
with tabs[3]:
    sday = st.date_input("Splits for date", day, key="splits_day")
    st.caption(f"Sport: {SPORTS[sport].label}")
    st.write("Side A is the **home team** for spread/ml and the **over** for totals. Lines are "
             "from side A's view (home spread like -4.5; ML like -150). Copy numbers from any "
             "public-betting source you trust.")
    cur = public_betting.load_csv(sday, sport)
    if cur.empty and res is not None:
        cur = pd.DataFrame([dict(home=g.home_name or g.home, away=g.away_name or g.away, market=m,
                                 open_line=g.open_spread if m == "spread" else g.open_total,
                                 current_line=g.spread if m == "spread" else g.total,
                                 tickets_pct_a=None, money_pct_a=None)
                            for g in res.ctx.games for m in ("spread", "total")])
    edited = st.data_editor(cur if not cur.empty else pd.DataFrame(columns=public_betting.COLUMNS),
                            num_rows="dynamic", use_container_width=True)
    if st.button("Save splits"):
        p = public_betting.save_csv(sday, edited, sport)
        st.success(f"Saved {p}. Re-run the sims to use them.")

# ---------------------------------------------------------------- backtest
with tabs[4]:
    bt = ROOT / "BACKTEST.md"
    st.markdown(bt.read_text() if bt.exists() else "Run `python -m edgesim backtest` first.")
    detail = ROOT / "backtests" / f"{sport.upper()}.md"
    if detail.exists():
        with st.expander(f"Full {SPORTS[sport].label} backtest", expanded=False):
            st.markdown(detail.read_text())

# ---------------------------------------------------------------- tracker
with tabs[5]:
    st.write("Every logged play is graded against the final score and the closing line. "
             "Run the card again shortly before tip-off so the last odds snapshot is a real close.")
    if st.button("Grade logged plays"):
        try:
            tracker.grade()
        except Exception as e:
            st.error(f"Grading failed: {e}")
    if tracker.LOG.exists():
        log = pd.read_csv(tracker.LOG)
        if "result" in log.columns:
            st.dataframe(tracker.summary(log), hide_index=True)
        st.dataframe(log, use_container_width=True, hide_index=True)
    else:
        st.info("No plays logged yet.")

# ---------------------------------------------------------------- alerts
with tabs[6]:
    st.write("Line-movement alerts. The opening number is where the value is: big moves win "
             "58-60% against the open in the backtest but only about 50% against the close. "
             "For hands-off alerts all day, run `python -m edgesim watch` in a terminal.")
    st.caption("NEW PLAY: clears the card's bar. EARLY: sim far from the market and the line "
               "hasn't moved yet. STEAM: line already moved (the open was the value). "
               "SHARP: reverse line move or money/ticket split.")
    if st.button("Scan all sports now"):
        from edgesim.watch import scan
        with st.spinner("Scanning..."):
            for k in SPORTS:
                try:
                    new, left = scan(date.today(), k, action_network=True)
                    st.write(f"**{SPORTS[k].label}**: {len(new)} new alerts, {left} games still to start")
                except Exception as e:
                    st.write(f"**{SPORTS[k].label}**: scan failed ({e})")
    af = ROOT / "reports" / f"alerts_{date.today():%Y-%m-%d}.md"
    if af.exists():
        st.markdown("\n".join(reversed(af.read_text().splitlines())))
    else:
        st.info("No alerts yet today.")
