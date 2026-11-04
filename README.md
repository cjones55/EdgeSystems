# EdgeSim NBA

A local NBA betting simulator: Monte Carlo game sims, player prop projections,
sharp-money detection from public betting splits and line movement, and a
daily card of plays. Everything is walk-forward backtested on real
historical lines; see [BACKTEST.md](BACKTEST.md) for the full results.

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # ~80 MB of historical data (needed for backtest/demo)
python -m pytest -q                   # 48 tests
python -m edgesim demo                # full pipeline on an archived date, offline
streamlit run app.py                  # dashboard at http://localhost:8501
```

For live daily plays:

```bash
cp .env.example .env                  # then paste your Odds API key into .env
python -m edgesim doctor              # checks every data source from your machine
python -m edgesim today               # today's card -> reports/<date>_live.md
python -m edgesim today --props       # also price player props (uses more API credits)
python -m edgesim grade               # next day: grade plays vs results + closing line
```

## What each part does

| Part | File | How it works |
|---|---|---|
| Team ratings | `edgesim/ratings.py` | Offense/defense ratings updated after every game, home court and rest/back-to-back adjustments, regression to the mean between seasons. |
| Game sim | `edgesim/gamesim.py` | 20,000 sims per game. Plays out every possession, varies pace, adds the model's own uncertainty, calibrates the spread of outcomes to real NBA results, and plays overtime until ties break. Outputs win %, cover %, over %, push %, fair lines. |
| Props | `edgesim/props.py` | Per-player minutes projection and per-minute production (shrunk toward league average), opponent adjustment, pace and blowout adjustments from the game sim. Negative binomial distributions fitted on past seasons. Points, rebounds, assists, threes, PRA and combos. |
| Sharp detection | `edgesim/sharp.py` | Reverse line movement (most bets on one side, line moves the other way), money vs ticket splits, steam moves, contrarian spots. Scored 0-100. |
| Daily card | `edgesim/plays.py` | Blends the sim with the market's no-vig price, computes EV and a capped quarter-Kelly stake at the best available price, and tiers plays: **A** model + sharp agree, **B** model edge, **C** sharp lean. Plays the sharp side is against are dropped. |
| Tracker | `edgesim/tracker.py` | Logs every live play and grades it on the result and on closing line value (CLV). |

## Data sources

| Source | Needs | Used for |
|---|---|---|
| ESPN public API | nothing | schedule, scores, box scores, injuries, fallback lines |
| [The Odds API](https://the-odds-api.com) | `ODDS_API_KEY` (free tier: 500 credits/month) | live spreads/totals/moneylines across books, line shopping, player props |
| Public betting % | your CSV, or the app's **Public betting %** tab | ticket % and money % for sharp signals |
| Action Network (experimental) | nothing, `--action-network` | an unofficial feed of splits; may break at any time |
| Historical archive | `scripts/fetch_data.py` | backtests and demo mode |

Public betting % has no free official API. The CSV format is
`home,away,market,open_line,current_line,tickets_pct_a,money_pct_a`
where side A is the home team (spread, ml) or the over (total), saved as
`data/public_betting/YYYY-MM-DD.csv`. Without it, sharp signals use line
movement only (the app snapshots odds every time you run it, and the first
snapshot of the day is the "open").

## What the backtest says, in one paragraph

The sim is well calibrated (when it says 70%, it happens about 70% of the
time) and its picks get positive closing line value at the open, but on its
own it does **not** beat NBA closing lines after the vig, and neither does
blindly following line moves after they happen. That is why the card blends
the model toward the market and only flags a play when the price is good
enough, why most days produce few or no game plays, and why every live play is
tracked against the closing line. Sharp signals from public betting % can't be
backtested (no free historical data), so the tracker is how they get proven
or disproven. Full numbers in [BACKTEST.md](BACKTEST.md).

## Running the backtest yourself

```bash
python -m edgesim backtest            # ~4 minutes; rewrites BACKTEST.md and data/fitted_params.json
```
