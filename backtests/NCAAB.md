# College basketball backtest

Data: results 2015-2026; near-closing lines from many books for 2020-21 to 2022-23. 11,889 games simulated, 9,370 in the held-out test seasons (2022-2026). Ratings were tuned on 2021-2021 only, and every prediction only uses games played before that date.

## Win probability vs the closing moneyline

| source | log_loss | brier |
|---|---|---|
| EdgeSim sim | 0.553 | 0.188 |
| Closing line (no-vig) | 0.542 | 0.183 |
| Coin flip | 0.693 | 0.250 |

Lower is better.

## Calibration of home win probability

| bucket | n | predicted | actual |
|---|---|---|---|
| 0.0-0.1 | 83 | 7.2% | 4.8% |
| 0.1-0.2 | 322 | 15.4% | 14.6% |
| 0.2-0.3 | 544 | 25.4% | 25.2% |
| 0.3-0.4 | 854 | 35.1% | 33.3% |
| 0.4-0.5 | 1111 | 45.2% | 46.2% |
| 0.5-0.6 | 1383 | 55.2% | 54.1% |
| 0.6-0.7 | 1449 | 65.0% | 65.3% |
| 0.7-0.8 | 1441 | 74.9% | 75.4% |
| 0.8-0.9 | 1329 | 84.8% | 85.1% |
| 0.9-1.0 | 854 | 94.3% | 95.0% |

## Spread bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 9370 | 4623 | 4584 | 163 | 50.2% | -4.1% | 1.0% | -4.16 |
| 0.0% | 7214 | 3559 | 3531 | 124 | 50.2% | -4.1% | 1.1% | -3.68 |
| 2.0% | 6335 | 3121 | 3102 | 112 | 50.2% | -4.2% | 1.2% | -3.51 |
| 4.0% | 5515 | 2721 | 2700 | 94 | 50.2% | -4.1% | 1.3% | -3.22 |
| 6.0% | 4739 | 2343 | 2319 | 77 | 50.3% | -4.0% | 1.4% | -2.90 |
| 8.0% | 4032 | 2011 | 1957 | 64 | 50.7% | -3.2% | 1.5% | -2.14 |
| 10.0% | 3391 | 1682 | 1652 | 57 | 50.4% | -3.6% | 1.6% | -2.23 |

## Total bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 9370 | 4750 | 4537 | 83 | 51.1% | -2.4% | 1.0% | -2.41 |
| 0.0% | 7326 | 3692 | 3567 | 67 | 50.9% | -2.9% | 1.1% | -2.61 |
| 2.0% | 6465 | 3265 | 3139 | 61 | 51.0% | -2.7% | 1.2% | -2.26 |
| 4.0% | 5627 | 2857 | 2716 | 54 | 51.3% | -2.1% | 1.3% | -1.68 |
| 6.0% | 4846 | 2473 | 2325 | 48 | 51.5% | -1.6% | 1.4% | -1.18 |
| 8.0% | 4182 | 2120 | 2020 | 42 | 51.2% | -2.2% | 1.5% | -1.53 |
| 10.0% | 3553 | 1809 | 1708 | 36 | 51.4% | -1.8% | 1.6% | -1.13 |

## Moneyline bets at the closing price

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 9069 | 4738 | 4331 | 0 | 52.2% | -5.5% | 1.3% | -4.28 |
| 0.0% | 6353 | 2886 | 3467 | 0 | 45.4% | -7.7% | 1.7% | -4.56 |
| 2.0% | 5336 | 2251 | 3085 | 0 | 42.2% | -7.8% | 1.9% | -4.05 |
| 4.0% | 4569 | 1815 | 2754 | 0 | 39.7% | -9.0% | 2.1% | -4.25 |
| 6.0% | 3966 | 1497 | 2469 | 0 | 37.7% | -10.0% | 2.3% | -4.27 |
| 8.0% | 3432 | 1243 | 2189 | 0 | 36.2% | -10.3% | 2.6% | -3.99 |
| 10.0% | 2966 | 1014 | 1952 | 0 | 34.2% | -12.3% | 2.8% | -4.35 |

Each row bets the side the sim prefers whenever its expected value at the real price is at least `min_ev`. `z` is ROI divided by its standard error; |z| < 2 is indistinguishable from luck.

## The daily card's rule, replayed

Tier B plays blend the sim with the market and need EV of +1% or more.

Training seasons (2021-2021):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 32 | 1.3 | 43.8% | -16.2% | -0.95 |
| total | close | 6 | 0.2 | 66.7% | 27.2% | +0.68 |
| ml | close | 9 | 0.4 | 0.0% | -100.0% |  |

Test seasons (2022-2026):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 107 | 1.1 | 53.8% | 2.7% | +0.30 |
| total | close | 6 | 0.1 | 83.3% | 59.1% | +1.86 |
| ml | close | 34 | 0.4 | 11.8% | 1.0% | +0.02 |
