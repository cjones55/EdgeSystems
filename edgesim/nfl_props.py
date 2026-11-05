"""NFL player props: projections, distributions, pricing.

Data: nflverse weekly player stats (2016 on) and the nflverse schedule, which
carries each game's closing spread and total (used for game environment).

Projection for a player and stat in an upcoming game
    recent form   exponentially weighted average of the player's past games
                  (half-life fitted on 2016-2020), only counting games he played
  x opponent      how much the opponent has allowed to that position group,
                  relative to the league, shrunk toward 1
  x environment   the team's implied points for this game (from the spread and
                  total) relative to its recent implied points, to a fitted power

Distributions (fitted on 2016-2020, checked on 2021 on):
  yards           empirical distribution of actual / projection, by stat and
                  projection size (this captures the long right tail and the
                  zero games without assuming a shape)
  counts          negative binomial with a fitted dispersion per stat
  anytime TD      Poisson on projected rushing + receiving TDs
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nfl_players"
FITTED = ROOT / "data" / "fitted_nfl_props.json"


@dataclass(frozen=True)
class StatDef:
    label: str
    cols: tuple
    kind: str                 # yards | count | td
    groups: tuple             # eligible position groups
    min_proj: float           # skip players whose projection is below this (no role)


STATS: dict[str, StatDef] = {
    "pass_yds": StatDef("Passing yards", ("passing_yards",), "yards", ("QB",), 120),
    "pass_att": StatDef("Pass attempts", ("attempts",), "count", ("QB",), 18),
    "pass_cmp": StatDef("Completions", ("completions",), "count", ("QB",), 12),
    "pass_tds": StatDef("Passing TDs", ("passing_tds",), "count", ("QB",), 0.6),
    "pass_ints": StatDef("Interceptions", ("passing_interceptions",), "count", ("QB",), 0.3),
    "rush_yds": StatDef("Rushing yards", ("rushing_yards",), "yards", ("QB", "RB", "WR"), 12),
    "rush_att": StatDef("Rush attempts", ("carries",), "count", ("QB", "RB"), 4),
    "rec": StatDef("Receptions", ("receptions",), "count", ("RB", "WR", "TE"), 1.5),
    "rec_yds": StatDef("Receiving yards", ("receiving_yards",), "yards", ("RB", "WR", "TE"), 12),
    "rush_rec_yds": StatDef("Rush + rec yards", ("rushing_yards", "receiving_yards"), "yards",
                            ("RB", "WR", "TE"), 25),
    "anytime_td": StatDef("Anytime TD", ("rushing_tds", "receiving_tds"), "td",
                          ("QB", "RB", "WR", "TE"), 0.12),
}

# backtest: props the model put above 70% hit about 65-77% of the time
PROB_CAP = 0.72

# The Odds API market keys -> our stat keys
ODDS_MARKETS = {"player_pass_yds": "pass_yds", "player_pass_attempts": "pass_att",
                "player_pass_completions": "pass_cmp", "player_pass_tds": "pass_tds",
                "player_pass_interceptions": "pass_ints", "player_rush_yds": "rush_yds",
                "player_rush_attempts": "rush_att", "player_receptions": "rec",
                "player_reception_yds": "rec_yds", "player_rush_reception_yds": "rush_rec_yds",
                "player_anytime_td": "anytime_td"}

# nflverse team codes that differ from ours
TEAM_FIX = {"LA": "LAR", "OAK": "LV", "SD": "LAC", "STL": "LAR", "WAS": "WSH"}


@dataclass
class NFLPropParams:
    halflife: float = 6.0           # games
    opp_halflife: float = 8.0       # defensive games
    opp_weight: float = 0.5         # 0 = ignore opponent, 1 = full ratio
    env_power: float = 0.6          # implied-points elasticity
    opp_min_games: int = 4
    dispersion: dict = field(default_factory=dict)     # count stat -> NB r
    ratio_q: dict = field(default_factory=dict)        # "stat|bucket" -> quantiles of actual/proj
    buckets: dict = field(default_factory=dict)        # stat -> projection cut points
    td_scale: float = 1.0
    calib_a: float = 1.0            # P(over) recalibration: sigmoid(a * logit(p) + b)
    calib_b: float = 0.0

    @classmethod
    def load(cls, path: Path = FITTED) -> "NFLPropParams":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(**d)

    def save(self, path: Path = FITTED) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__))


# ---------------------------------------------------------------------------
def _fix(t):
    return TEAM_FIX.get(t, t)


def load_player_games(raw: Path = RAW) -> pd.DataFrame:
    """One row per player-game, with game date, opponent and the team's implied points."""
    files = sorted(raw.glob("stats_player_week_*.csv"))
    if not files:
        raise FileNotFoundError("NFL player stats missing: run scripts/fetch_data.py nfl")
    cols = ["player_id", "player_display_name", "position", "position_group", "season", "week",
            "season_type", "game_id", "team", "opponent_team", "completions", "attempts",
            "passing_yards", "passing_tds", "passing_interceptions", "carries", "rushing_yards",
            "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds"]
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
                   ignore_index=True)
    df = df[df.season_type.isin(["REG", "POST"])]
    df = df.rename(columns={"player_display_name": "player"})
    df["team"] = df.team.map(_fix)
    df["opponent_team"] = df.opponent_team.map(_fix)
    g = pd.read_csv(raw / "games.csv", usecols=["game_id", "gameday", "home_team", "away_team",
                                                "spread_line", "total_line"])
    g["home_team"] = g.home_team.map(_fix)
    g["away_team"] = g.away_team.map(_fix)
    df = df.merge(g, on="game_id", how="left")
    df["date"] = pd.to_datetime(df.gameday)
    home = df.team == df.home_team
    # nflverse spread_line is positive when the home team is favoured
    df["implied"] = np.where(home, df.total_line / 2 + df.spread_line / 2,
                             df.total_line / 2 - df.spread_line / 2)
    for k, sd in STATS.items():
        df[k] = df[list(sd.cols)].fillna(0).sum(axis=1)
    return df.drop(columns=["gameday", "spread_line", "total_line"]).sort_values(
        ["date", "game_id"]).reset_index(drop=True)


def _ewm_prior(s: pd.Series, keys: pd.Series, halflife: float) -> pd.Series:
    """EWM of earlier rows within each key (the current row excluded)."""
    return s.groupby(keys).transform(lambda x: x.shift(1).ewm(halflife=halflife, min_periods=1).mean())


def build_features(df: pd.DataFrame, params: NFLPropParams | None = None) -> pd.DataFrame:
    """Add proj_<stat> for every row from strictly earlier games.

    Rows for upcoming games can be appended with stat columns set to NaN."""
    p = params or NFLPropParams()
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    out = df.copy()
    out["n_prev"] = df.groupby("player_id").cumcount()
    # team environment: this game's implied points vs the team's recent implied points
    team_games = df.drop_duplicates(["team", "game_id"])[["team", "game_id", "date", "implied"]]
    team_games = team_games.sort_values("date")
    team_games["implied_base"] = _ewm_prior(team_games.implied, team_games.team, 8)
    out = out.merge(team_games[["team", "game_id", "implied_base"]], on=["team", "game_id"], how="left")
    env = (out.implied / out.implied_base).clip(0.6, 1.6)
    env = env.fillna(1.0) ** p.env_power
    for k in STATS:
        form = _ewm_prior(df[k], df.player_id, p.halflife)
        # opponent: stat allowed to this position group per game, vs the league
        allowed = (df.dropna(subset=[k]).groupby(["opponent_team", "position_group", "game_id", "date"])[k]
                   .sum().reset_index().sort_values("date"))
        key = allowed.opponent_team + "|" + allowed.position_group
        allowed["opp_ewm"] = _ewm_prior(allowed[k], key, p.opp_halflife)
        allowed["opp_n"] = allowed.groupby(key).cumcount()
        lg = allowed.groupby(["position_group", "date"])[k].mean().reset_index()
        lg["lg"] = lg.groupby("position_group")[k].transform(
            lambda x: x.shift(1).ewm(halflife=16, min_periods=1).mean())
        allowed = allowed.merge(lg[["position_group", "date", "lg"]], on=["position_group", "date"])
        allowed["ratio"] = np.where(allowed.opp_n >= p.opp_min_games,
                                    allowed.opp_ewm / allowed.lg.replace(0, np.nan), 1.0)
        m = out[["opponent_team", "position_group", "game_id"]].merge(
            allowed[["opponent_team", "position_group", "game_id", "ratio"]],
            on=["opponent_team", "position_group", "game_id"], how="left")
        # games not played yet: use each defence's latest form, including its last game
        key_all = allowed.opponent_team + "|" + allowed.position_group
        last = allowed.assign(full=allowed.groupby(key_all)[k].transform(
            lambda x: x.ewm(halflife=p.opp_halflife).mean())).groupby(key_all).tail(1)
        lg_last = allowed.groupby("position_group")[k].apply(
            lambda x: x.ewm(halflife=16).mean().iloc[-1])
        last = last.assign(r_last=np.where(last.opp_n + 1 >= p.opp_min_games,
                                           last.full / last.position_group.map(lg_last), 1.0))
        latest = dict(zip(last.opponent_team + "|" + last.position_group, last.r_last))
        missing = m.ratio.isna() & out[k].isna().values
        m.loc[missing, "ratio"] = (m.opponent_team + "|" + m.position_group)[missing].map(latest)
        ratio = m.ratio.fillna(1.0).clip(0.5, 2.0).values
        opp = 1 + p.opp_weight * (ratio - 1)
        out[f"form_{k}"] = form.values
        out[f"proj_{k}"] = form.values * opp * env.values
    return out


# ---------------------------------------------------------------------------
def _bucket(p: NFLPropParams, stat: str, proj: float) -> int:
    cuts = p.buckets.get(stat, [])
    return int(np.searchsorted(cuts, proj))


def prob_over(stat: str, proj: float, line: float, p: NFLPropParams) -> tuple[float, float]:
    """(P(over), P(push)) for one prop, after the fitted recalibration."""
    po, push = raw_prob_over(stat, proj, line, p)
    if not np.isfinite(po) or (p.calib_a == 1.0 and p.calib_b == 0.0):
        return po, push
    decided = 1 - push
    q = np.clip(po / decided if decided > 0 else po, 1e-6, 1 - 1e-6)
    q = 1 / (1 + np.exp(-(p.calib_a * np.log(q / (1 - q)) + p.calib_b)))
    return float(q * decided), push


def raw_prob_over(stat: str, proj: float, line: float, p: NFLPropParams) -> tuple[float, float]:
    """(P(over), P(push)) straight from the fitted distribution."""
    kind = STATS[stat].kind
    if proj is None or not np.isfinite(proj) or proj <= 0:
        return np.nan, 0.0
    if kind == "td":
        lam = proj * p.td_scale
        # anytime TD "line" is 0.5: P(at least one)
        return float(1 - np.exp(-lam)) if line < 1 else float(sps.poisson.sf(np.floor(line), lam)), 0.0
    if kind == "count":
        r = p.dispersion.get(stat, 30.0)
        prob = r / (r + proj)
        dist = sps.nbinom(r, prob)
        if float(line).is_integer():
            return float(dist.sf(line)), float(dist.pmf(line))
        return float(dist.sf(np.floor(line))), 0.0
    q = p.ratio_q.get(f"{stat}|{_bucket(p, stat, proj)}")
    if not q:  # unfitted fallback: gamma with CV 0.6
        shape = 1 / 0.36
        return float(sps.gamma.sf(line, shape, scale=proj / shape)), 0.0
    qs = np.asarray(q)
    levels = np.linspace(0, 1, len(qs))
    # P(actual/proj > line/proj) from the empirical quantiles
    cdf = float(np.interp(line / proj, qs, levels, left=0.0, right=1.0))
    return 1 - cdf, 0.0


def fit_distributions(feat: pd.DataFrame, p: NFLPropParams) -> NFLPropParams:
    for k, sd in STATS.items():
        d = feat[feat.position_group.isin(sd.groups) & (feat[f"proj_{k}"] >= sd.min_proj)
                 & (feat.n_prev >= 3)].dropna(subset=[k, f"proj_{k}"])
        if d.empty:
            continue
        mu, y = d[f"proj_{k}"].values, d[k].values
        if sd.kind == "count":
            # Var = mu + mu^2 / r  (method of moments on the residuals)
            excess = ((y - mu) ** 2 - mu).mean()
            p.dispersion[k] = float(np.clip((mu ** 2).mean() / max(excess, 1e-6), 2, 500))
        elif sd.kind == "td":
            p.td_scale = float(np.clip(d[k].clip(upper=1).gt(0).mean() /
                                       (1 - np.exp(-mu)).mean(), 0.7, 1.3))
        else:
            cuts = list(np.quantile(mu, [1 / 3, 2 / 3]))
            p.buckets[k] = [float(c) for c in cuts]
            b = np.searchsorted(cuts, mu)
            for i in range(3):
                r = y[b == i] / mu[b == i]
                p.ratio_q[f"{k}|{i}"] = [float(v) for v in np.quantile(r, np.linspace(0, 1, 201))]
    return p


def eligible(feat: pd.DataFrame, stat: str) -> pd.Series:
    sd = STATS[stat]
    return feat.position_group.isin(sd.groups) & (feat[f"proj_{stat}"] >= sd.min_proj) & (feat.n_prev >= 3)


# ---------------------------------------------------------------------------
def project_upcoming(pg: pd.DataFrame, games: list[tuple[str, str, float | None, float | None]],
                     day: pd.Timestamp, params: NFLPropParams | None = None,
                     out_players: set[str] | None = None, lookback_games: int = 3) -> pd.DataFrame:
    """Projections for players on teams in ``games`` = [(home, away, home_spread, total)].

    Rosters are players who appeared in one of their team's last ``lookback_games``
    games; ``out_players`` (lower-case names) are dropped."""
    p = params or NFLPropParams.load()
    hist = pg[pg.date < day]
    rows = []
    for home, away, spread, total in games:
        for team, opp, is_home in ((home, away, True), (away, home, False)):
            last = hist[hist.team == team].drop_duplicates("game_id").sort_values("date").tail(lookback_games)
            roster = hist[hist.game_id.isin(last.game_id) & (hist.team == team)]
            roster = roster.sort_values("date").groupby("player_id").tail(1)
            imp = np.nan
            if spread is not None and total is not None:
                # our spreads are the home line (negative = home favoured)
                imp = total / 2 - spread / 2 if is_home else total / 2 + spread / 2
            for r in roster.itertuples():
                rows.append(dict(player_id=r.player_id, player=r.player, position=r.position,
                                 position_group=r.position_group, team=team, opponent_team=opp,
                                 game_id=f"next_{team}", date=day, implied=imp,
                                 **{k: np.nan for k in STATS}))
    if not rows:
        return pd.DataFrame()
    stub = pd.DataFrame(rows)
    feat = build_features(pd.concat([hist, stub], ignore_index=True), p)
    out = feat[feat.game_id.astype(str).str.startswith("next_")].copy()
    if out_players:
        out = out[~out.player.str.lower().isin(out_players)]
    return out


def price_props(lines: pd.DataFrame, proj: pd.DataFrame, params: NFLPropParams | None = None,
                cfg=None) -> pd.DataFrame:
    """Price prop lines: player, stat, line, over_price, under_price, book -> card rows."""
    from .odds import expected_value, kelly_fraction, no_vig_probs
    from .plays import PlayConfig, units
    p = params or NFLPropParams.load()
    cfg = cfg or PlayConfig.for_sport("nfl")
    if lines.empty or proj.empty:
        return pd.DataFrame()
    by_name = proj.drop_duplicates("player").set_index(proj.drop_duplicates("player").player.str.lower())
    rows = []
    for r in lines.itertuples(index=False):
        key = str(r.player).lower()
        if key not in by_name.index or r.stat not in STATS:
            continue
        x = by_name.loc[key]
        mu = x[f"proj_{r.stat}"]
        sd = STATS[r.stat]
        # no projection for players without an established role in this stat
        if not (x.position_group in sd.groups and x.n_prev >= 3 and mu >= sd.min_proj):
            continue
        p_over, p_push = prob_over(r.stat, mu, r.line, p)
        if not np.isfinite(p_over):
            continue
        sides = [("Over", p_over, r.over_price, r.under_price)]
        if r.under_price is not None and not pd.isna(r.under_price):
            sides.append(("Under", 1 - p_over - p_push, r.under_price, r.over_price))
        for side, pm, price, other in sides:
            if price is None or pd.isna(price):
                continue
            if other is None or pd.isna(other):
                mkt = no_vig_probs([price, -price if abs(price) > 100 else 100])[0]
                mkt = min(mkt, 1 / (1 + (price / 100 if price > 0 else 100 / -price)) / 1.045)
            else:
                mkt = no_vig_probs([price, other])[0]
            decided = 1 - p_push
            pm_d = pm / decided if decided > 0 else pm
            # the backtest shows the model runs hot past ~70%: cap what we trust
            pm_d = min(pm_d, PROB_CAP)
            pb = mkt + cfg.blend["prop"] * (pm_d - mkt)
            pw = pb * decided
            ev = expected_value(pw, price, p_push)
            pick = (f"{r.player} anytime TD" if r.stat == "anytime_td" else
                    f"{r.player} {side} {r.line:g} {STATS[r.stat].label.lower()}")
            rows.append(dict(player=r.player, team=x.team, stat=r.stat, side=side.lower(),
                             line=r.line, pick=pick, price=price, book=r.book,
                             projection=float(mu), model_prob=pm_d, market_prob=mkt,
                             blended_prob=pb, ev=ev,
                             kelly=kelly_fraction(pw, price, p_push, cfg.kelly_fraction, cfg.kelly_cap)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["play"] = out.ev >= cfg.min_ev + 0.01
    out["units"] = [min(units("B", e, k), 5) if pl else None
                    for pl, e, k in zip(out.play, out.ev, out.kelly)]
    return out.sort_values("ev", ascending=False).reset_index(drop=True)


def proxy_lines(proj: pd.DataFrame, hist: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Demo-mode stand-in for sportsbook lines: each player's median over his last
    ``n`` games, rounded to x.5, at -115 both ways (anytime TD priced from his
    recent scoring rate)."""
    rows = []
    h = hist[hist.player_id.isin(proj.player_id)].sort_values("date").groupby("player_id").tail(n)
    med = h.groupby("player_id")[list(STATS)].median()
    td_rate = h.assign(scored=h.anytime_td > 0).groupby("player_id").scored.mean()
    for r in proj.itertuples(index=False):
        if r.player_id not in med.index:
            continue
        for k, sd in STATS.items():
            if not (r.position_group in sd.groups and getattr(r, f"proj_{k}") >= sd.min_proj
                    and r.n_prev >= 3):
                continue
            if sd.kind == "td":
                # a book-like price from the player's recent scoring rate (shrunk), with vig
                q = min(0.9, (td_rate.get(r.player_id, 0.3) * 8 + 0.3 * 4) / 12 * 1.07)
                price = round(-100 * q / (1 - q)) if q >= 0.5 else round(100 * (1 - q) / q)
                rows.append(dict(player=r.player, stat=k, line=0.5, over_price=price,
                                 under_price=None, book="proxy"))
            elif med.loc[r.player_id, k] < sd.min_proj:
                continue
            else:
                rows.append(dict(player=r.player, stat=k, line=float(np.floor(med.loc[r.player_id, k]) + 0.5),
                                 over_price=-115, under_price=-115, book="proxy"))
    return pd.DataFrame(rows)


def _proxy_eval(d: pd.DataFrame, k: str, p: NFLPropParams, raw: bool) -> tuple[np.ndarray, np.ndarray]:
    """P(over) and outcomes at the proxy line (last-8 median rounded to x.5)."""
    sd = STATS[k]
    d = d[eligible(d, k) & d[f"med_{k}"].notna() & d[k].notna()]
    if d.empty:
        return np.array([]), np.array([])
    line = np.full(len(d), 0.5) if sd.kind == "td" else (np.floor(d[f"med_{k}"]) + 0.5).values
    fn = raw_prob_over if raw else prob_over
    po = np.array([fn(k, mu, ln, p)[0] for mu, ln in zip(d[f"proj_{k}"], line)])
    hit = (d[k].values > line).astype(float)
    ok = np.isfinite(po)
    return np.clip(po[ok], 1e-4, 1 - 1e-4), hit[ok]


# ---------------------------------------------------------------------------
def fit_and_backtest(train_end: int = 2020, test_start: int = 2021) -> tuple[NFLPropParams, str]:
    """Fit the projection and distribution parameters on 2016..train_end, then score
    every eligible player-game from test_start on. Returns (params, markdown)."""
    pg = load_player_games()
    def score(h, w, e):
        f = build_features(pg, NFLPropParams(halflife=h, opp_weight=w, env_power=e))
        f = f[(f.season > 2016) & (f.season <= train_end)]
        err = 0.0
        for k in ("pass_yds", "rush_yds", "rec_yds", "rec"):
            m = eligible(f, k)
            err += np.abs(f.loc[m, k] - f.loc[m, f"proj_{k}"]).mean() / f.loc[m, k].mean()
        return err

    # coordinate search: half-life, then opponent weight, then implied-points power
    h, w, e = 6.0, 0.5, 0.6
    h = min((4.0, 6.0, 8.0, 12.0), key=lambda v: score(v, w, e))
    w = min((0.0, 0.25, 0.5, 0.75, 1.0), key=lambda v: score(h, v, e))
    e = min((0.0, 0.3, 0.6, 1.0), key=lambda v: score(h, w, v))
    p = NFLPropParams(halflife=h, opp_weight=w, env_power=e)
    feat = build_features(pg, p)
    train = feat[(feat.season > 2016) & (feat.season <= train_end)]
    p = fit_distributions(train, p)
    med8 = pg.groupby("player_id")[list(STATS)].transform(
        lambda x: x.shift(1).rolling(8, min_periods=3).median())
    # recalibrate P(over) on the training seasons (the raw distributions are a bit
    # overconfident when the projection is far from the line)
    tr = train.join(med8.add_prefix("med_"), how="left")
    raws = [_proxy_eval(tr, k, p, raw=True) for k in STATS]
    pr = np.concatenate([r[0] for r in raws])
    hr = np.concatenate([r[1] for r in raws])
    from scipy.optimize import minimize
    lg = np.log(pr / (1 - pr))

    def nll(ab):
        q = np.clip(1 / (1 + np.exp(-(ab[0] * lg + ab[1]))), 1e-6, 1 - 1e-6)
        return -np.mean(hr * np.log(q) + (1 - hr) * np.log(1 - q))
    ab = minimize(nll, [1.0, 0.0], method="Nelder-Mead").x
    p.calib_a, p.calib_b = float(ab[0]), float(ab[1])
    test = feat[feat.season >= test_start].copy()
    # naive baseline: player's last-5 average
    base = pg.groupby("player_id")[list(STATS)].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    test = test.join(base.add_prefix("l5_"), how="left")
    L = ["# NFL player props backtest", "",
         f"Fitted on {2017}-{train_end}, scored on {test_start}-{int(feat.season.max())} "
         "(walk-forward: every projection uses only earlier games).", "",
         f"Chosen settings: form half-life {p.halflife:g} games, opponent weight {p.opp_weight:g}, "
         f"implied-points power {p.env_power:g}. Probability recalibration "
         f"sigmoid({p.calib_a:.2f} x logit(p) {p.calib_b:+.2f}), fitted on the training seasons.", "",
         "## Projection accuracy vs a last-5-games average", "",
         "| stat | player-games | mean actual | model error (MAE) | last-5 error | improvement |",
         "|---|---|---|---|---|---|"]
    for k, sd in STATS.items():
        if sd.kind == "td":
            continue
        m = eligible(test, k) & test[f"l5_{k}"].notna()
        d = test[m]
        mae = np.abs(d[k] - d[f"proj_{k}"]).mean()
        b = np.abs(d[k] - d[f"l5_{k}"]).mean()
        L.append(f"| {sd.label} | {len(d)} | {d[k].mean():.1f} | {mae:.2f} | {b:.2f} | {1 - mae / b:+.1%} |")
    L += ["", "## Calibration of P(over) at proxy lines", "",
          "No free history of sportsbook prop lines exists, so each prop is tested at a stand-in "
          "line: the player's median over his previous 8 games, rounded to x.5 (anytime TD at "
          "0.5). This checks whether the probabilities mean what they say. It cannot show an "
          "edge over real lines, which are sharper than a median.", "",
          "| stat | props | avg P(over) | actual over rate | log loss | log loss at 50/50 |",
          "|---|---|---|---|---|---|"]
    test = test.join(med8.add_prefix("med_"), how="left")
    calib_rows = []
    for k, sd in STATS.items():
        po, hit = _proxy_eval(test, k, p, raw=False)
        if not len(po):
            continue
        ll = -np.mean(hit * np.log(po) + (1 - hit) * np.log(1 - po))
        base_p = np.clip(hit.mean(), 1e-4, 1 - 1e-4) if sd.kind == "td" else 0.5
        ll0 = -np.mean(hit * np.log(base_p) + (1 - hit) * np.log(1 - base_p))
        L.append(f"| {sd.label} | {len(po)} | {po.mean():.1%} | {hit.mean():.1%} | {ll:.4f} | {ll0:.4f} |")
        calib_rows.append(pd.DataFrame(dict(p=po, hit=hit)))
    allc = pd.concat(calib_rows)
    allc["bucket"] = pd.cut(allc.p, [0, .2, .35, .45, .55, .65, .8, 1])
    cb = allc.groupby("bucket", observed=True).agg(n=("hit", "size"), predicted=("p", "mean"),
                                                   actual=("hit", "mean"))
    L += ["", "All stats pooled:", "", "| model P(over) | props | predicted | actual |", "|---|---|---|---|"]
    for b, r in cb.iterrows():
        L.append(f"| {b} | {int(r.n)} | {r.predicted:.1%} | {r.actual:.1%} |")
    L += ["", f"Because the top buckets run hot, the card caps any prop probability at "
          f"{PROB_CAP:.0%} before blending with the market.", ""]
    L += ["", "**Reading this:** the projections should beat a simple recent average, and the "
          "predicted over rates should track the actual ones. Real prop lines are set by books "
          "with the same information plus injury and role news, so real edges will be far "
          "smaller than the gaps a proxy line shows. The tracker grades live props against "
          "real lines."]
    return p, "\n".join(L)


def lines_from_odds(raw: pd.DataFrame, books: list[str] | None = None) -> pd.DataFrame:
    """Odds API prop rows -> one line per player/stat: the most common line among your
    books (or all books when yours don't list it) and the best price on each side."""
    rows = []
    if raw.empty:
        return pd.DataFrame(columns=["player", "stat", "line", "over_price", "under_price", "book"])
    raw = raw.dropna(subset=["player", "line"])
    for (player, stat), g in raw.groupby(["player", "stat"]):
        if books and g.book.isin(books).any():
            g = g[g.book.isin(books)]
        line = g.line.mode().iloc[0]
        at = g[g.line == line]
        ov, un = at[at.side == "over"], at[at.side == "under"]
        if ov.empty:
            continue
        bo = ov.loc[ov.price.idxmax()]
        bu = un.loc[un.price.idxmax()] if not un.empty else None
        rows.append(dict(player=player, stat=stat, line=float(line), over_price=bo.price,
                         under_price=None if bu is None else bu.price,
                         book=bo.book if bu is None else f"{bo.book}/{bu.book}"))
    return pd.DataFrame(rows)
