"""EdgeSim local dashboard.  Run:  streamlit run app.py"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from edgesim import pipeline, tracker
from edgesim.cli import _load_env
from edgesim.gamesim import simulate_game
from edgesim.providers import public_betting
from edgesim.props import COMBOS, simulate_player
from edgesim.teams import TEAMS

ROOT = Path(__file__).resolve().parent
_load_env()
st.set_page_config(page_title="EdgeSim NBA", layout="wide")
st.title("EdgeSim NBA")

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
    return df.style.format(f, na_rep="")


with st.sidebar:
    mode = st.radio("Data", ["Live (today)", "Demo (archived date)"])
    if mode.startswith("Live"):
        day = st.date_input("Date", date.today())
        with_props = st.checkbox("Price player props (uses Odds API credits)", False)
        an = st.checkbox("Try Action Network splits (experimental)", False)
    else:
        day = st.date_input("Archived date", date(2022, 1, 17), min_value=date(2011, 12, 25),
                            max_value=date(2022, 6, 16))
    run = st.button("Run sims", type="primary")
    st.caption("Bets are tiered A (model + sharp agree), B (model edge), C (sharp lean). "
               "See the Backtest tab for what has and has not been proven.")

if run:
    with st.spinner("Loading data and simulating..."):
        try:
            if mode.startswith("Live"):
                ctx = pipeline.live_context(day, with_props=with_props, action_network=an)
            else:
                ctx = pipeline.demo_context(day)
            st.session_state.res = pipeline.run_day(ctx)
            pipeline.write_report(st.session_state.res)
        except Exception as e:  # show the problem instead of a stack trace
            st.error(f"Could not build the card: {e}")

res = st.session_state.get("res")
tabs = st.tabs(["Today's card", "Game simulator", "Props", "Public betting %", "Backtest",
                "Tracker"])

# ---------------------------------------------------------------- card
with tabs[0]:
    if res is None:
        st.info("Pick a mode in the sidebar and press **Run sims**.")
    else:
        for n in res.ctx.notes:
            st.caption(n)
        st.subheader(f"Plays for {res.ctx.day}")
        if res.plays.empty:
            st.write("Nothing cleared the bar. That is the expected outcome on most days: NBA "
                     "lines are efficient and the system is built not to force bets.")
        else:
            st.dataframe(fmt(res.plays[["tier", "game", "pick", "price", "book", "blended_prob",
                                        "market_prob", "model_prob", "ev", "kelly", "sharp_score",
                                        "tickets_pct", "money_pct", "sharp_reasons"]]),
                         use_container_width=True, hide_index=True)
            if res.ctx.source == "live" and st.button("Log these plays to the tracker"):
                n = tracker.log_plays(res.plays, res.ctx.day)
                st.success(f"Tracker now holds {n} plays.")
        st.subheader("Game simulations")
        st.dataframe(fmt(res.sims.drop(columns=["home", "away"])), use_container_width=True,
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
        codes = sorted(TEAMS)
        c1, c2, c3, c4 = st.columns(4)
        away = c1.selectbox("Away", codes, index=codes.index("BOS"))
        home = c2.selectbox("Home", codes, index=codes.index("MIA"))
        spread = c3.number_input("Home spread", value=-2.5, step=0.5)
        total = c4.number_input("Total", value=220.5, step=0.5)
        hr = st.slider("Home rest days", 1, 4, 2)
        ar = st.slider("Away rest days", 1, 4, 2)
        eh, ea = res.ctx.ratings.expected(home, away, hr, ar)
        sim = simulate_game(eh, ea)
        s = sim.summary()
        m = st.columns(5)
        m[0].metric(f"{home} win", f"{s['home_win']:.1%}")
        m[1].metric("Fair spread", f"{home} {s['fair_spread']:+.1f}")
        m[2].metric("Fair total", f"{s['fair_total']:.1f}")
        cov, push = sim.p_cover(spread)
        m[3].metric(f"{home} {spread:+g} covers", f"{cov:.1%}", f"push {push:.1%}", delta_color="off")
        ov, push = sim.p_over(total)
        m[4].metric(f"Over {total:g}", f"{ov:.1%}", f"push {push:.1%}", delta_color="off")
        h1, h2 = st.columns(2)
        h1.caption(f"{home} margin distribution")
        h1.bar_chart(pd.Series(sim.margin).value_counts().sort_index())
        h2.caption("Total points distribution")
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
    st.write("Side A is the **home team** for spread/ml and the **over** for totals. Lines are "
             "from side A's view (home spread like -4.5; ML like -150). Copy numbers from any "
             "public-betting source you trust.")
    cur = public_betting.load_csv(sday)
    if cur.empty and res is not None:
        cur = pd.DataFrame([dict(home=g.home, away=g.away, market=m,
                                 open_line=g.open_spread if m == "spread" else g.open_total,
                                 current_line=g.spread if m == "spread" else g.total,
                                 tickets_pct_a=None, money_pct_a=None)
                            for g in res.ctx.games for m in ("spread", "total")])
    edited = st.data_editor(cur if not cur.empty else pd.DataFrame(columns=public_betting.COLUMNS),
                            num_rows="dynamic", use_container_width=True)
    if st.button("Save splits"):
        p = public_betting.save_csv(sday, edited)
        st.success(f"Saved {p}. Re-run the sims to use them.")

# ---------------------------------------------------------------- backtest
with tabs[4]:
    bt = ROOT / "BACKTEST.md"
    st.markdown(bt.read_text() if bt.exists() else "Run `python -m edgesim backtest` first.")

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
