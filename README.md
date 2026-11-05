# EdgeSim

A local sports betting simulator for the **NBA, NFL, NHL, college football and
college basketball**: Monte Carlo game sims, NBA player prop projections,
sharp-money detection from public betting splits and line movement, and a
daily card of plays. Everything is walk-forward backtested on real
historical lines; see [BACKTEST.md](BACKTEST.md) for the full results.

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # historical data for every sport (needed for backtest/demo)
python -m pytest -q                   # 72 tests
python -m edgesim demo --sport nba    # full pipeline on an archived date, offline
python -m edgesim demo --sport ncaaf  # same for nfl, nhl, ncaaf, ncaab
streamlit run app.py                  # then open http://localhost:8501
```

For live daily plays:

```bash
cp .env.example .env                  # then paste your Odds API key into .env
python -m edgesim doctor              # checks every data source from your machine
python -m edgesim today --sport all   # every sport's card -> reports/<date>_<sport>_live.md
python -m edgesim today --sport nba --props   # also price NBA player props (more API credits)
python -m edgesim sim Duke UNC --sport ncaab --spread -3.5 --total 150.5
python -m edgesim watch --sport all   # re-check lines every 15 min, alert on early edges and steam
python -m edgesim grade               # next day: grade plays vs results + closing line
```

## What each part does

| Part | File | How it works |
|---|---|---|
| Team ratings | `edgesim/ratings.py` | Offense/defense ratings updated after every game, home court and rest/back-to-back adjustments, regression to the mean between seasons. |
| Game sims | `edgesim/gamesim.py` (NBA, college basketball), `gamesim_nfl.py` (NFL, college football), `gamesim_nhl.py` | 20,000 sims per game. Basketball plays out every possession with varying pace. Football plays out drives, so margins land on key numbers like 3 and 7, with college overtime rules. Hockey simulates goals with score effects, empty-net goals, 3-on-3 overtime and shootouts. Every sim adds the model's own uncertainty and is calibrated to real results. Outputs win %, cover %, over %, push %, fair lines. |
| Sport settings | `edgesim/sports.py` | One entry per sport: rating parameters, sim settings, data sources, train/test seasons, how much to trust the market, sharp thresholds. College sports also handle neutral sites and hundreds of teams. |
| Props (NBA) | `edgesim/props.py` | Per-player minutes projection and per-minute production (shrunk toward league average), opponent adjustment, pace and blowout adjustments from the game sim. Negative binomial distributions fitted on past seasons. Points, rebounds, assists, threes, PRA and combos. |
| Sharp detection | `edgesim/sharp.py` | Reverse line movement (most bets on one side, line moves the other way), money vs ticket splits, steam moves, contrarian spots. Scored 0-100. |
| Daily card | `edgesim/plays.py` | Blends the sim with the market's no-vig price, computes EV and a capped quarter-Kelly stake at the best available price, and tiers plays: **A** model + sharp agree, **B** model edge, **C** sharp lean. Plays the sharp side is against are dropped. |
| Tracker | `edgesim/tracker.py` | Logs every live play and grades it on the result and on closing line value (CLV). |

## NFL player props

`edgesim/nfl_props.py` projects passing, rushing and receiving yards, attempts, completions,
TDs, interceptions, receptions and anytime TD from nflverse weekly stats (2016 on): recent
form (4-game half-life), adjusted for the game's implied points. Yards use the empirical spread
of outcomes; counts use a negative binomial; anytime TD is Poisson. On 2021-2026 the projections
beat a last-5-games average for every stat and the probabilities are well calibrated up to about
70%, above which they run hot, so the card caps prop probabilities at 72%. Full results in
[backtests/NFL_PROPS.md](backtests/NFL_PROPS.md). Real lines come from the Odds API (FanDuel and
DraftKings by default); without a key the Props tab still shows every projection and the chance
of going over any line you type in. Refresh the current season weekly with
`python scripts/fetch_data.py nfl --refresh-current`.

## Game breakdown

The Games tab opens any game: projected score, fair moneyline/spread/total next to the market,
cover and over chances, margin and total distributions, the most likely final scores, every
market's numbers, and a projected box score (NBA players with minutes, points, rebounds,
assists and more, adding up to the simulated team score; NFL passing, rushing and receiving with
anytime TD chances).

## Unit sizes

Every play lists a stake on a 1u / 2u / 3u / 5u (max) / 10u (super max) scale, with no 4u or
6-9u. Size follows the capped quarter-Kelly stake (1 unit = 0.5% of bankroll): under 0.75%
is 1u, under 1.5% is 2u, under 2.5% is 3u, otherwise 5u. 10u needs all three of: model and
sharp signal agree, Kelly at its 3% cap, and EV of at least 5%. Sharp leans without a model
edge are 1u. The tracker logs the stake (`stake_u`) and reports profit at that stake.

## Watch mode and alerts

The backtest's clearest edge is timing: big line moves win 58-60% against the opening number
but only about 50% against the close. `python -m edgesim watch` scans every 15 minutes
(snapshotting ESPN lines even without an API key, and saving public betting % history) and
alerts once per side for **NEW PLAY**, **EARLY** (sim far from the market and the line hasn't
moved yet; logged as tier W so its closing line value is graded), **STEAM** (line already
moved, marked with whether the sim agrees) and **SHARP** (reverse line move or money/ticket
split). Alerts go to the terminal, `reports/alerts_<date>.md`, the app's **Alerts** tab, and
desktop notifications on macOS.

## Your books

With an Odds API key, prices and lines come from the books in `EDGESIM_BOOKS` (default
`fanduel,draftkings`), taking the better of the two, so every price on the card is one you
can bet. Set `EDGESIM_BOOKS=all` to shop every book.

## Data sources

| Source | Needs | Used for |
|---|---|---|
| ESPN public API | nothing | schedule, scores, box scores, injuries, fallback lines |
| [The Odds API](https://the-odds-api.com) | `ODDS_API_KEY` (free tier: 500 credits/month) | live spreads/totals/moneylines across books, line shopping, player props |
| Public betting % | your CSV, or the app's **Public betting %** tab | ticket % and money % for sharp signals |
| Action Network (experimental) | nothing, `--action-network` | an unofficial feed of splits; may break at any time |
| Historical archive | `scripts/fetch_data.py` | backtests and demo mode: pro leagues 2011-2022 with lines, college football 2015-2026 with lines, college basketball 2015-2026 results and 2021-2023 closing odds |

Public betting % has no free official API. The CSV format is
`home,away,market,open_line,current_line,tickets_pct_a,money_pct_a`
where side A is the home team (spread, ml) or the over (total), saved as
`data/public_betting/<sport>/YYYY-MM-DD.csv`. Without it, sharp signals use line
movement only (the app snapshots odds every time you run it, and the first
snapshot of the day is the "open").

## What the backtest says, in one paragraph

In every sport the sim is well calibrated (when it says 70%, it happens about
70% of the time) and the line moves toward its side more often than away, but
on its own it does **not** beat closing lines after the vig, and neither does
blindly following line moves after they happen. That is why the card blends
the model toward the market and only flags a play when the price is good
enough, why most days produce few or no game plays, and why every live play is
tracked against the closing line. Sharp signals from public betting % can't be
backtested (no free historical data), so the tracker is how they get proven
or disproven. Full numbers in [BACKTEST.md](BACKTEST.md).

## Running the backtest yourself

```bash
python -m edgesim backtest --sport all   # rewrites BACKTEST.md, backtests/<SPORT>.md, data/fitted_params.json
```

MLB is not included yet: it needs starting-pitcher data to be worth simulating.
```
