# College basketball backtest

Data: results 2015-2026; near-closing lines from many books for 2020-21 to 2022-23. 11,889 games simulated, 9,370 in the held-out test seasons (2022-2026). Ratings were tuned on 2021-2021 only, and every prediction only uses games played before that date.

## Win probability vs the closing moneyline

| source | log_loss | brier |
|---|---|---|
| EdgeSim sim | 0.547 | 0.185 |
| Closing line (no-vig) | 0.542 | 0.183 |
| Coin flip | 0.693 | 0.250 |

Lower is better.

## Calibration of home win probability

| bucket | n | predicted | actual |
|---|---|---|---|
| 0.0-0.1 | 101 | 7.1% | 6.9% |
| 0.1-0.2 | 321 | 15.4% | 13.7% |
| 0.2-0.3 | 590 | 25.3% | 23.1% |
| 0.3-0.4 | 844 | 35.2% | 35.0% |
| 0.4-0.5 | 1166 | 45.4% | 45.6% |
| 0.5-0.6 | 1324 | 55.1% | 54.8% |
| 0.6-0.7 | 1406 | 65.1% | 66.1% |
| 0.7-0.8 | 1457 | 74.9% | 75.5% |
| 0.8-0.9 | 1265 | 84.9% | 85.7% |
| 0.9-1.0 | 896 | 94.5% | 95.4% |

## Spread bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 9370 | 4576 | 4631 | 163 | 49.7% | -5.0% | 1.0% | -5.14 |
| 0.0% | 6367 | 3146 | 3117 | 104 | 50.2% | -4.0% | 1.2% | -3.39 |
| 2.0% | 5215 | 2548 | 2587 | 80 | 49.6% | -5.2% | 1.3% | -3.95 |
| 4.0% | 4165 | 2017 | 2086 | 62 | 49.2% | -6.0% | 1.5% | -4.12 |
| 6.0% | 3202 | 1561 | 1596 | 45 | 49.4% | -5.5% | 1.7% | -3.29 |
| 8.0% | 2466 | 1210 | 1219 | 37 | 49.8% | -4.8% | 1.9% | -2.52 |
| 10.0% | 1872 | 931 | 912 | 29 | 50.5% | -3.5% | 2.2% | -1.59 |

## Total bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 9370 | 4742 | 4545 | 83 | 51.1% | -2.5% | 1.0% | -2.57 |
| 0.0% | 6482 | 3310 | 3110 | 62 | 51.6% | -1.6% | 1.2% | -1.34 |
| 2.0% | 5387 | 2734 | 2600 | 53 | 51.3% | -2.2% | 1.3% | -1.66 |
| 4.0% | 4348 | 2210 | 2094 | 44 | 51.3% | -2.0% | 1.4% | -1.37 |
| 6.0% | 3434 | 1765 | 1635 | 34 | 51.9% | -0.9% | 1.6% | -0.56 |
| 8.0% | 2686 | 1381 | 1280 | 25 | 51.9% | -0.9% | 1.8% | -0.51 |
| 10.0% | 2084 | 1079 | 987 | 18 | 52.2% | -0.3% | 2.1% | -0.15 |

## Moneyline bets at the closing price

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 9069 | 4998 | 4071 | 0 | 55.1% | -6.7% | 1.2% | -5.75 |
| 0.0% | 5233 | 2464 | 2769 | 0 | 47.1% | -6.5% | 1.8% | -3.71 |
| 2.0% | 3979 | 1692 | 2287 | 0 | 42.5% | -7.4% | 2.1% | -3.44 |
| 4.0% | 3127 | 1238 | 1889 | 0 | 39.6% | -8.5% | 2.5% | -3.39 |
| 6.0% | 2472 | 915 | 1557 | 0 | 37.0% | -8.4% | 3.0% | -2.83 |
| 8.0% | 1984 | 698 | 1286 | 0 | 35.2% | -8.5% | 3.4% | -2.51 |
| 10.0% | 1596 | 530 | 1066 | 0 | 33.2% | -8.4% | 4.0% | -2.11 |

Each row bets the side the sim prefers whenever its expected value at the real price is at least `min_ev`. `z` is ROI divided by its standard error; |z| < 2 is indistinguishable from luck.

## The daily card's rule, replayed

Tier B plays blend the sim with the market and need EV of +1% or more.

Training seasons (2021-2021):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 9 | 0.4 | 66.7% | 28.1% | +0.88 |
| total | close | 5 | 0.2 | 60.0% | 14.2% | +0.31 |
| ml | close | 1 | 0.0 | 0.0% | -100.0% |  |

Test seasons (2022-2026):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 5 | 0.1 | 40.0% | -23.6% | -0.51 |
| total | close | 3 | 0.0 | 33.3% | -36.4% | -0.57 |
| ml | close | 6 | 0.1 | 0.0% | -100.0% |  |
