"""EdgeSim local dashboard.  Run:  streamlit run app.py"""
from __future__ import annotations

import html
from datetime import date, datetime
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
st.set_page_config(page_title="EdgeSim", page_icon="🎯", layout="wide")

ICON = {"nba": "🏀", "ncaab": "🏀", "nfl": "🏈", "ncaaf": "🏈", "nhl": "🏒"}
UNIT_COLOR = {1: "#64748b", 2: "#2563eb", 3: "#7c3aed", 5: "#ea580c", 10: "#dc2626"}

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.play {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px;
       margin-bottom: 10px; background: rgba(128,128,128,.05);}
.play .top {display: flex; align-items: center; gap: 10px; flex-wrap: wrap;}
.play .units {color: #fff; font-weight: 700; border-radius: 8px; padding: 4px 10px;
              font-size: 1.05rem; min-width: 46px; text-align: center;}
.play .pick {font-size: 1.25rem; font-weight: 700;}
.play .price {font-size: 1.05rem; opacity: .85;}
.play .chip {font-size: .75rem; border-radius: 999px; padding: 2px 9px;
             border: 1px solid rgba(128,128,128,.4); opacity: .9;}
.play .chip.A {background: rgba(22,163,74,.15); border-color: rgba(22,163,74,.6);}
.play .chip.B {background: rgba(37,99,235,.12); border-color: rgba(37,99,235,.5);}
.play .chip.C {background: rgba(234,179,8,.15); border-color: rgba(234,179,8,.6);}
.play .meta {opacity: .7; font-size: .85rem; margin-top: 4px;}
.play .stats {display: flex; gap: 22px; margin-top: 10px; flex-wrap: wrap;}
.play .stat b {display: block; font-size: 1.05rem;}
.play .stat span {font-size: .72rem; opacity: .65; text-transform: uppercase; letter-spacing: .03em;}
.play .why {font-size: .82rem; opacity: .8; margin-top: 8px;}
.pos {color: #16a34a;} .neg {color: #dc2626;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def local_time(s) -> str:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if t.tzinfo is None:
            return ""
        return t.astimezone().strftime("%-I:%M %p")
    except Exception:
        return ""


def fmt_price(p) -> str:
    return "" if p is None or pd.isna(p) else f"{p:+.0f}"


def _num(v) -> float | None:
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def play_card(r: dict) -> str:
    sport = r["sport"]
    u = _num(r.get("units"))
    u = int(u) if u else None
    badge = (f'<div class="units" style="background:{UNIT_COLOR.get(u, "#64748b")}">{u}u</div>'
             if u else "")
    book = r.get("book")
    book = f" @ {html.escape(str(book))}" if isinstance(book, str) and book else ""
    edge = r["model_prob"] - r["market_prob"]
    tk, mn = _num(r.get("tickets_pct")), _num(r.get("money_pct"))
    pub = ""
    if tk is not None:
        pub = (f'<div class="stat"><b>{tk:.0f}% / {"-" if mn is None else f"{mn:.0f}%"}</b>'
               f'<span>bets / money</span></div>')
    why = r.get("sharp_reasons")
    if isinstance(why, str) and why:
        sep = " vs " if " vs " in str(r["game"]) else " @ "
        away_n, _, home_n = str(r["game"]).partition(sep)
        a, b = ("the over", "the under") if r.get("market") == "total" else (home_n, away_n)
        why = why.replace("side A", a).replace("side B", b)
    why = html.escape(why) if isinstance(why, str) and why else ""
    tier = r.get("tier") if isinstance(r.get("tier"), str) else ""
    start = local_time(r.get("start"))
    return f"""
<div class="play">
  <div class="top">{badge}<div class="pick">{html.escape(str(r["pick"]))}</div>
    <div class="price">{fmt_price(r["price"])}{book}</div>
    <div class="chip {tier[:1]}">{html.escape(tier)}</div></div>
  <div class="meta">{ICON.get(sport, "")} {SPORTS[sport].label} · {html.escape(str(r["game"]))}{f" · {start}" if start else ""}</div>
  <div class="stats">
    <div class="stat"><b>{r["model_prob"]:.1%}</b><span>sim</span></div>
    <div class="stat"><b>{r["market_prob"]:.1%}</b><span>market (no vig)</span></div>
    <div class="stat"><b class="{'pos' if edge > 0 else 'neg'}">{edge:+.1%}</b><span>sim edge</span></div>
    <div class="stat"><b class="{'pos' if r["ev"] > 0 else 'neg'}">{r["ev"]:+.1%}</b><span>EV after blend</span></div>
    <div class="stat"><b>{r["sharp_score"]:.0f}</b><span>sharp score</span></div>
    {pub}
  </div>
  {f'<div class="why">{why}</div>' if why else ""}
</div>"""


def pct_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c] * 100
    return df


SIM_COLS = {"model_prob": st.column_config.NumberColumn("sim", format="%.1f%%"),
            "market_prob": st.column_config.NumberColumn("market", format="%.1f%%"),
            "ev": st.column_config.NumberColumn("EV", format="%+.1f%%"),
            "price": st.column_config.NumberColumn(format="%+d"),
            "units": st.column_config.NumberColumn(format="%du")}

def render_game(k: str, res, g) -> None:
    """Full drill-in for one game: team projections, distributions, box score."""
    from edgesim import boxscore as BX
    spc = SPORTS[k]
    sim = BX.game_sim(g, res.ctx.ratings, k)
    t = BX.team_summary(sim, g)
    hn, an = g.home_name or g.home, g.away_name or g.away
    st.markdown(f"#### {an} {t['away_pts']:.1f}  -  {t['home_pts']:.1f} {hn}")
    m = st.columns(5)
    m[0].metric(f"{hn} win", f"{t['home_win']:.1%}")
    fair_ml = BX_fair(t["home_win"])
    m[1].metric(f"Fair {hn} ML", f"{fair_ml:+.0f}",
                f"market {g.home_ml:+.0f}" if g.home_ml else None, delta_color="off")
    m[2].metric(f"Fair {spc.spread_label.lower()}", f"{t['fair_spread']:+.1f}",
                f"market {g.spread:+g}" if g.spread is not None else None, delta_color="off")
    m[3].metric("Fair total", f"{t['fair_total']:.1f}",
                f"market {g.total:g}" if g.total is not None else None, delta_color="off")
    m[4].metric("Overtime", f"{sim.went_ot.mean():.1%}")
    if "home_cover" in t or "over" in t:
        bits = []
        if "home_cover" in t:
            push_txt = f" (push {t['spread_push']:.1%})" if t.get("spread_push", 0) > 0.001 else ""
            bits.append(f"**{hn} {g.spread:+g}** covers {t['home_cover']:.1%}"
                        f"{push_txt}")
        if "over" in t:
            bits.append(f"**Over {g.total:g}** {t['over']:.1%}")
        st.markdown(" · ".join(bits))
    c1, c2, c3 = st.columns([2, 2, 1.4])
    c1.caption(f"{hn} margin (positive = {hn} wins)")
    c1.bar_chart(pd.Series(sim.margin).value_counts(normalize=True).sort_index() * 100, color="#2563eb",
                 y_label="% of sims")
    c2.caption("Total points")
    c2.bar_chart(pd.Series(sim.total).value_counts(normalize=True).sort_index() * 100, color="#7c3aed",
                 y_label="% of sims")
    ls = BX.likely_scores(sim)
    ls["chance"] = ls.chance * 100
    c3.caption("Most likely final scores")
    c3.dataframe(ls.rename(columns={"away": an, "home": hn}), hide_index=True,
                 column_config={"chance": st.column_config.NumberColumn(format="%.1f%%")})
    # candidates for this game
    cands = res.candidates[(res.candidates.home == g.home) & (res.candidates.away == g.away)]
    if not cands.empty:
        with st.expander("Every market for this game"):
            st.dataframe(pct_cols(cands[["pick", "price", "model_prob", "market_prob", "ev",
                                         "units", "sharp_score", "sharp_reasons"]],
                                  ["model_prob", "market_prob", "ev"]),
                         hide_index=True, use_container_width=True, column_config=SIM_COLS)
    # player box
    feat = res.ctx.player_features
    st.markdown("##### Projected box score")
    if feat.empty:
        st.caption("Player projections are NBA and NFL only. NBA needs **Price props** on in Live "
                   "mode (it pulls recent box scores); NFL projections load automatically.")
        return
    cols = st.columns(2)
    for col, team, name, pts in ((cols[0], g.away, an, t["away_pts"]), (cols[1], g.home, hn, t["home_pts"])):
        col.markdown(f"**{name}** · {pts:.1f} projected")
        if k == "nba":
            box = BX.nba_box(feat, team, pts)
            if box.empty:
                col.caption("No projected players.")
                continue
            col.dataframe(box.round(1), hide_index=True, use_container_width=True,
                          column_config={c: st.column_config.NumberColumn(format="%.1f")
                                         for c in ["min", *BX.NBA_STATS]})
        else:
            tabs_ = BX.nfl_box(feat, team)
            if not tabs_:
                col.caption("No projected players.")
            for title, df in tabs_.items():
                col.caption(title)
                cfg = {c: st.column_config.NumberColumn(format="%.1f") for c in df.columns
                       if c not in ("player", "position", "td_chance")}
                if "td_chance" in df:
                    df = df.assign(td_chance=df.td_chance * 100)
                    cfg["td_chance"] = st.column_config.NumberColumn("anytime TD", format="%.0f%%")
                col.dataframe(df, hide_index=True, use_container_width=True, column_config=cfg)
    st.caption("Player lines are each player's expected stats (averages, not a single simulated "
               "game). NFL numbers are the props model: recent form adjusted for the game's "
               "implied points.")


def BX_fair(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🎯 EdgeSim")
    mode = st.radio("Data", ["Live", "Demo (archive)"], horizontal=True,
                    label_visibility="collapsed")
    live = mode == "Live"
    if live:
        sports = st.multiselect("Sports", list(SPORTS), default=list(SPORTS),
                                format_func=lambda k: f"{ICON[k]} {SPORTS[k].label}")
        day = st.date_input("Date", date.today())
        an = st.toggle("Public betting % (Action Network)", True)
        with_props = bool({"nba", "nfl"} & set(sports)) and st.toggle(
            "Price NBA + NFL props (uses Odds API credits)", False)
    else:
        one = st.selectbox("Sport", list(SPORTS),
                           format_func=lambda k: f"{ICON[k]} {SPORTS[k].label}")
        sports = [one]
        day = st.date_input("Archived date", date.fromisoformat(DEMO_DATES[one]))
        st.caption(f"Archive: {SPORTS[one].archive_note}")
        an = with_props = False
    run = st.button("Run sims", type="primary", use_container_width=True, disabled=not sports)
    st.divider()
    st.markdown("**Units:** 1 · 2 · 3 · 5 max · 10 super max")
    st.caption("**A** model + sharp agree · **B** model edge · **C** sharp lean. "
               "The Backtest tab shows what has and has not been proven.")

if run:
    results, errors = {}, {}
    prog = st.progress(0.0, "Simulating...")
    for i, sp in enumerate(sports):
        prog.progress(i / len(sports), f"Simulating {SPORTS[sp].label}...")
        try:
            ctx = (pipeline.live_context(day, with_props=with_props, action_network=an, sport=sp)
                   if live else pipeline.demo_context(day, sport=sp))
            res = pipeline.run_day(ctx)
            pipeline.write_report(res)
            results[sp] = res
        except Exception as e:  # show the problem instead of a stack trace
            errors[sp] = str(e)
    prog.empty()
    st.session_state.results, st.session_state.errors = results, errors

results: dict = st.session_state.get("results", {})
errors: dict = st.session_state.get("errors", {})


def all_plays() -> pd.DataFrame:
    frames = []
    for k, r in results.items():
        if r.plays.empty:
            continue
        starts = dict(zip(r.sims.game, r.sims.start)) if not r.sims.empty else {}
        frames.append(r.plays.assign(sport=k, start=r.plays.game.map(starts)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


plays = all_plays()

# ------------------------------------------------------------------ header
if results:
    any_res = next(iter(results.values()))
    st.markdown(f"### {any_res.ctx.day:%A, %B %-d, %Y}"
                f"{'' if any_res.ctx.source == 'live' else '  ·  demo'}")
    m = st.columns(4)
    m[0].metric("Games simulated", sum(len(r.sims) for r in results.values()))
    m[1].metric("Plays", len(plays))
    m[2].metric("Units on the card", f"{plays.units.sum():.0f}u" if not plays.empty else "0u")
    m[3].metric("Best EV", f"{plays.ev.max():+.1%}" if not plays.empty else "-")
else:
    st.markdown("### Pick your sports and press **Run sims**")
for sp, e in errors.items():
    st.error(f"{SPORTS[sp].label}: could not build the card ({e})")

tabs = st.tabs(["🎯 Plays", "📋 Games", "🎲 Simulator", "👤 Props", "🔔 Alerts",
                "📊 Public betting %", "📈 Tracker", "🧪 Backtest"])

# ------------------------------------------------------------------ plays
with tabs[0]:
    if not results:
        st.info("Live mode runs today's slate for every sport you pick. Demo mode replays an "
                "archived date offline, with no API key needed.")
    else:
        with st.expander("Notes from the data sources", expanded=False):
            for k, r in results.items():
                for n in r.ctx.notes:
                    st.caption(f"{ICON[k]} {n}")
        if plays.empty:
            st.success("Nothing cleared the bar. That is the expected outcome on most days: "
                       "the system is built not to force bets.")
        else:
            f1, f2, f3 = st.columns([2, 2, 3])
            min_u = f1.select_slider("Minimum units", [1, 2, 3, 5, 10], value=1)
            tiers = sorted(plays.tier.dropna().unique())
            pick_t = f2.multiselect("Tiers", tiers, default=tiers)
            sort_by = f3.radio("Sort by", ["Units", "EV", "Start time"], horizontal=True)
            show = plays[(plays.units.fillna(0) >= min_u) & plays.tier.isin(pick_t)]
            key = {"Units": ["units", "ev"], "EV": ["ev"], "Start time": ["start"]}[sort_by]
            show = show.sort_values(key, ascending=sort_by == "Start time", na_position="last")
            left, right = st.columns(2)
            for i, r in enumerate(show.to_dict("records")):
                (left if i % 2 == 0 else right).markdown(play_card(r), unsafe_allow_html=True)
            live_res = {k: r for k, r in results.items() if r.ctx.source == "live"}
            if live_res and st.button("Log these plays to the tracker"):
                n = 0
                for k, r in live_res.items():
                    n = tracker.log_plays(r.plays, r.ctx.day, sport=k)
                st.success(f"Logged. The tracker now holds {n} plays.")
        prop_frames = [r.props[r.props.play].assign(sport=k) for k, r in results.items()
                       if not r.props.empty and "play" in r.props and r.props.play.any()]
        if prop_frames:
            pp = pd.concat(prop_frames, ignore_index=True).sort_values(["units", "ev"], ascending=False)
            st.markdown(f"#### Prop plays ({len(pp)})")
            st.dataframe(pct_cols(pp[["units", "sport", "pick", "price", "book", "projection",
                                      "model_prob", "market_prob", "ev"]].head(40),
                                  ["model_prob", "market_prob", "ev"]),
                         hide_index=True, use_container_width=True,
                         column_config=SIM_COLS | {"projection": st.column_config.NumberColumn(format="%.1f")})
        frames = [r.candidates.assign(sport=k) for k, r in results.items() if not r.candidates.empty]
        if frames:
            near = pd.concat(frames, ignore_index=True)
            near = near[near.tier.isna()].sort_values("ev", ascending=False).head(25)
            with st.expander("Closest to a play (every other side, best EV first)"):
                st.dataframe(pct_cols(near[["sport", "game", "pick", "price", "model_prob",
                                            "market_prob", "ev", "sharp_score", "move"]],
                                      ["model_prob", "market_prob", "ev"]),
                             hide_index=True, use_container_width=True,
                             column_config=SIM_COLS | {
                                 "sharp_score": st.column_config.NumberColumn("sharp", format="%.0f"),
                                 "move": st.column_config.NumberColumn("line move", format="%+.2f",
                                                                       help="Movement toward this side since the open")})

# ------------------------------------------------------------------ games
with tabs[1]:
    if not results:
        st.info("Run the sims to see every game's projection.")
    for k, r in results.items():
        if r.sims.empty:
            continue
        st.markdown(f"#### {ICON[k]} {SPORTS[k].label}")
        g = r.sims.copy()
        g["time"] = g.start.map(local_time)
        g["projected"] = [f"{a:.0f} - {h:.0f}" for a, h in zip(g.proj_away, g.proj_home)]
        g["spread_edge"] = g.market_spread - g.fair_spread
        g["total_edge"] = g.fair_total - g.market_total
        mls = {(x.home, x.away): x.home_ml for x in r.ctx.games}
        g["market_ml"] = [mls.get((h, a)) for h, a in zip(g.home, g.away)]
        g["fair_ml"] = [round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)
                        if 0 < p < 1 else None for p in g.home_win]
        g["home_win"] = g.home_win * 100
        cols = ["time", "game", "projected", "home_win", "fair_ml", "market_ml",
                "fair_spread", "market_spread",
                "spread_edge", "fair_total", "market_total", "total_edge"]
        st.dataframe(g[cols], hide_index=True, use_container_width=True, column_config={
            "home_win": st.column_config.ProgressColumn("home win", format="%.0f%%",
                                                        min_value=0, max_value=100),
            "projected": st.column_config.TextColumn("projected (away - home)"),
            "fair_ml": st.column_config.NumberColumn("fair home ML", format="%+d"),
            "market_ml": st.column_config.NumberColumn("market ML", format="%+d"),
            "fair_spread": st.column_config.NumberColumn(f"fair {SPORTS[k].spread_label.lower()}", format="%+.1f"),
            "market_spread": st.column_config.NumberColumn("market", format="%+.1f"),
            "spread_edge": st.column_config.NumberColumn(
                "pts to home", format="%+.1f",
                help="Market minus fair home line: positive means the sim likes the home side"),
            "fair_total": st.column_config.NumberColumn("fair total", format="%.1f"),
            "market_total": st.column_config.NumberColumn("market", format="%.1f"),
            "total_edge": st.column_config.NumberColumn(
                "pts to over", format="%+.1f",
                help="Fair total minus market: positive means the sim likes the over"),
        })

    # ---------------- drill-in
    if results:
        st.divider()
        st.markdown("### 🔍 Game breakdown")
        opts = [(k, i) for k, r in results.items() for i in range(len(r.ctx.games))]
        def _lab(o):
            k, i = o
            g = results[k].ctx.games[i]
            return (f"{ICON[k]} {g.away_name or g.away} {'vs' if g.neutral else '@'} "
                    f"{g.home_name or g.home}")
        if opts:
            k, i = st.selectbox("Open a game", opts, format_func=_lab, key="drill")
            render_game(k, results[k], results[k].ctx.games[i])

# ------------------------------------------------------------------ simulator
with tabs[2]:
    if not results:
        st.info("Run the card first so the simulator uses the same ratings.")
    else:
        sk = st.selectbox("Sport", list(results), key="sim_sport",
                          format_func=lambda k: f"{ICON[k]} {SPORTS[k].label}")
        res = results[sk]
        spc = SPORTS[sk]
        keys = sorted(set(res.ctx.ratings.off) | {g.home for g in res.ctx.games}
                      | {g.away for g in res.ctx.games}, key=lambda k: display(k, sk))
        name = {k: display(k, sk) for k in keys}
        g0 = res.ctx.games[0] if res.ctx.games else None
        c1, c2, c3, c4, c5 = st.columns([3, 3, 2, 2, 1])
        away = c1.selectbox("Away", keys, index=keys.index(g0.away) if g0 and g0.away in keys else 0,
                            format_func=name.get)
        home = c2.selectbox("Home", keys, index=keys.index(g0.home) if g0 and g0.home in keys else 1,
                            format_func=name.get)
        spread = c3.number_input(f"Home {spc.spread_label.lower()}",
                                 value=-1.5 if sk == "nhl" else -2.5, step=0.5)
        total = c4.number_input("Total", value=float(round(res.ctx.ratings.league_avg * 2) / 2 + 0.5),
                                step=0.5)
        neutral = c5.checkbox("Neutral", False)
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
        h1.bar_chart(pd.Series(sim.margin).value_counts().sort_index(), color="#2563eb")
        h2.caption("Total distribution")
        h2.bar_chart(pd.Series(sim.total).value_counts().sort_index(), color="#7c3aed")

# ------------------------------------------------------------------ props
with tabs[3]:
    prop_sports = [k for k in ("nba", "nfl") if k in results
                   and not results[k].ctx.player_features.empty]
    if not prop_sports:
        st.info("Props cover the NBA and NFL. Turn on **Price props** in Live mode, or run an NBA "
                "or NFL demo date.")
    else:
        pk = st.radio("League", prop_sports, horizontal=True,
                      format_func=lambda k: f"{ICON[k]} {SPORTS[k].label}", key="props_league")
        res_p = results[pk]
        feat = res_p.ctx.player_features
        if not res_p.props.empty and res_p.props.play.any():
            st.markdown("#### Prop plays")
            pp = res_p.props[res_p.props.play].head(40)
            st.dataframe(pct_cols(pp[["units", "pick", "price", "book", "projection",
                                      "model_prob", "market_prob", "ev"]],
                                  ["model_prob", "market_prob", "ev"]),
                         hide_index=True, use_container_width=True,
                         column_config=SIM_COLS | {"projection": st.column_config.NumberColumn(format="%.1f")})
        elif res_p.ctx.prop_lines.empty:
            st.caption("No prop lines loaded (they need ODDS_API_KEY). Projections below still work: "
                       "type in your book's line.")
        else:
            st.caption("No prop cleared the bar.")
        st.markdown("#### Player explorer")
        if pk == "nba":
            c1, c2, c3 = st.columns([3, 2, 2])
            player = c1.selectbox("Player", sorted(feat.player.unique()))
            x = feat[feat.player == player].iloc[-1]
            stat = c2.selectbox("Stat", ["pts", "reb", "ast", "fg3m", *COMBOS])
            rates = {s: x[f"rate_{s}"] for s in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]}
            opp = {s: x[f"opp_{s}"] for s in rates}
            sim = simulate_player(rates, x.proj_min, x.min_sd, n=20000, opp=opp, seed=1)
            line = c3.number_input("Line", value=float(np.floor(sim[stat].mean()) + 0.5), step=0.5)
            a, b, c = st.columns(3)
            a.metric("Projection", f"{sim[stat].mean():.1f}")
            b.metric(f"Over {line:g}", f"{(sim[stat] > line).mean():.1%}")
            c.metric("Projected minutes", f"{x.proj_min:.1f} ± {x.min_sd:.1f}")
            st.bar_chart(sim[stat].value_counts().sort_index(), color="#2563eb")
            st.caption("Uncalibrated simulation view. The card's prop prices also apply the fitted "
                       "calibration and blend with the market.")
        else:
            from edgesim import nfl_props as NP
            params = NP.NFLPropParams.load()
            elig = {k: NP.eligible(feat, k) for k in NP.STATS}
            players = sorted(feat[np.logical_or.reduce(list(elig.values()))].player.unique())
            c1, c2, c3 = st.columns([3, 2, 2])
            player = c1.selectbox("Player", players)
            x = feat[feat.player == player].iloc[-1]
            mine = [k for k in NP.STATS if elig[k][feat.player == player].any()]
            stat = c2.selectbox("Stat", mine, format_func=lambda k: NP.STATS[k].label)
            mu = float(x[f"proj_{stat}"])
            default = 0.5 if NP.STATS[stat].kind == "td" else float(np.floor(mu) + 0.5)
            line = c3.number_input("Line", value=default, step=0.5 if mu < 30 else 1.0)
            po, push = NP.prob_over(stat, mu, line, params)
            a, b, c = st.columns(3)
            a.metric("Projection", f"{mu:.2f}" if mu < 5 else f"{mu:.1f}")
            b.metric("Anytime TD" if stat == "anytime_td" else f"Over {line:g}", f"{po:.1%}")
            fair = (-100 * po / (1 - po)) if po >= 0.5 else 100 * (1 - po) / po
            c.metric("Fair price", f"{fair:+.0f}")
            st.caption(f"{x.position} · {x.team} vs {x.opponent_team} · "
                       f"recent form {x[f'form_{stat}']:.1f}, adjusted for the opponent and the "
                       "game's implied points.")
            if NP.STATS[stat].kind != "td":
                grid = np.arange(max(0.5, np.floor(mu * 0.4) + 0.5), np.floor(mu * 1.8) + 1, 
                                 1.0 if mu < 30 else 5.0)
                curve = pd.DataFrame({"line": grid, "P(over)": [
                    NP.prob_over(stat, mu, g, params)[0] * 100 for g in grid]}).set_index("line")
                st.caption("Chance of going over each line")
                st.line_chart(curve, color="#2563eb")

# ------------------------------------------------------------------ alerts
with tabs[4]:
    st.write("The opening number is where the value is: big moves win 58-60% against the open "
             "in the backtest but only about 50% against the close. For hands-off alerts all day, "
             "run `python -m edgesim watch` in a terminal.")
    st.caption("NEW PLAY clears the card's bar · EARLY: sim far from the market and the line "
               "hasn't moved yet · STEAM: line already moved (the open was the value) · "
               "SHARP: reverse line move or money/ticket split.")
    if st.button("Scan all sports now"):
        from edgesim.watch import scan
        with st.spinner("Scanning..."):
            for k in SPORTS:
                try:
                    new, left = scan(date.today(), k, action_network=True)
                    st.write(f"{ICON[k]} **{SPORTS[k].label}**: {len(new)} new alerts, "
                             f"{left} games still to start")
                except Exception as e:
                    st.write(f"{ICON[k]} **{SPORTS[k].label}**: scan failed ({e})")
    af = ROOT / "reports" / f"alerts_{date.today():%Y-%m-%d}.md"
    if af.exists():
        st.markdown("\n".join(reversed(af.read_text().splitlines())))
    else:
        st.info("No alerts yet today.")

# ------------------------------------------------------------------ splits
with tabs[5]:
    c1, c2 = st.columns(2)
    ssport = c1.selectbox("Sport", list(SPORTS), key="splits_sport",
                          format_func=lambda k: f"{ICON[k]} {SPORTS[k].label}")
    sday = c2.date_input("Date", day, key="splits_day")
    st.caption("Side A is the **home team** for spread/ml and the **over** for totals. Lines are "
               "from side A's view (home spread like -4.5; ML like -150). Type in numbers from any "
               "public-betting source you trust; a saved file is used instead of Action Network "
               "for that day.")
    cur = public_betting.load_csv(sday, ssport)
    r0 = results.get(ssport)
    if cur.empty and r0 is not None:
        cur = pd.DataFrame([dict(home=g.home_name or g.home, away=g.away_name or g.away, market=m,
                                 open_line=g.open_spread if m == "spread" else g.open_total,
                                 current_line=g.spread if m == "spread" else g.total,
                                 tickets_pct_a=None, money_pct_a=None)
                            for g in r0.ctx.games for m in ("spread", "total")])
    edited = st.data_editor(cur if not cur.empty else pd.DataFrame(columns=public_betting.COLUMNS),
                            num_rows="dynamic", use_container_width=True)
    if st.button("Save splits"):
        p = public_betting.save_csv(sday, edited, ssport)
        st.success(f"Saved {p}. Re-run the sims to use them.")

# ------------------------------------------------------------------ tracker
with tabs[6]:
    st.write("Every logged play is graded against the final score and the closing line. Run the "
             "card again shortly before the games start so the last odds snapshot is a real close.")
    if st.button("Grade logged plays"):
        try:
            tracker.grade()
        except Exception as e:
            st.error(f"Grading failed: {e}")
    if tracker.LOG.exists():
        log = pd.read_csv(tracker.LOG)
        if "result" in log.columns:
            summ = tracker.summary(log)
            if not summ.empty:
                t1, t2, t3 = st.columns(3)
                t1.metric("Graded bets", int(summ.bets.sum()))
                t2.metric("Profit at your stakes", f"{summ.profit_at_stake.sum():+.1f}u")
                t3.metric("Beat the closing line", f"{np.nanmean(summ.beat_close_pct):.0%}"
                          if summ.beat_close_pct.notna().any() else "-")
                st.dataframe(summ, hide_index=True, use_container_width=True)
        st.dataframe(log.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No plays logged yet.")

# ------------------------------------------------------------------ backtest
with tabs[7]:
    bt = ROOT / "BACKTEST.md"
    st.markdown(bt.read_text() if bt.exists() else "Run `python -m edgesim backtest` first.")
    for k in SPORTS:
        detail = ROOT / "backtests" / f"{k.upper()}.md"
        if detail.exists():
            with st.expander(f"{ICON[k]} Full {SPORTS[k].label} backtest", expanded=False):
                st.markdown(detail.read_text())
