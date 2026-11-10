# College basketball backtest

Data: results 2015-2026; near-closing lines from many books for 2020-21 to 2022-23. 11,889 games simulated, 9,370 in the held-out test seasons (2022-2026). Ratings were tuned on 2021-2021 only, and every prediction only uses games played before that date.

## Win probability vs the closing moneyline

| source | log_loss | brier |
|---|---|---|
| EdgeSim sim | 0.551 | 0.187 |
| Closing line (no-vig) | 0.542 | 0.183 |
| Coin flip | 0.693 | 0.250 |

Lower is better.

## Calibration of home win probability

| bucket | n | predicted | actual |
|---|---|---|---|
| 0.0-0.1 | 110 | 7.3% | 6.4% |
| 0.1-0.2 | 338 | 15.5% | 13.9% |
| 0.2-0.3 | 596 | 25.5% | 26.5% |
| 0.3-0.4 | 899 | 35.0% | 33.9% |
| 0.4-0.5 | 1141 | 45.2% | 48.6% |
| 0.5-0.6 | 1389 | 55.1% | 54.8% |
| 0.6-0.7 | 1439 | 65.1% | 65.9% |
| 0.7-0.8 | 1362 | 74.9% | 77.5% |
| 0.8-0.9 | 1281 | 84.7% | 85.3% |
| 0.9-1.0 | 815 | 94.4% | 95.7% |

## Spread bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 9370 | 4580 | 4627 | 163 | 49.7% | -4.9% | 1.0% | -5.06 |
| 0.0% | 7023 | 3419 | 3474 | 130 | 49.6% | -5.2% | 1.1% | -4.62 |
| 2.0% | 6053 | 2948 | 2990 | 115 | 49.6% | -5.1% | 1.2% | -4.21 |
| 4.0% | 5133 | 2494 | 2542 | 97 | 49.5% | -5.3% | 1.3% | -4.05 |
| 6.0% | 4330 | 2093 | 2152 | 85 | 49.3% | -5.8% | 1.4% | -4.00 |
| 8.0% | 3569 | 1722 | 1783 | 64 | 49.1% | -6.1% | 1.6% | -3.84 |
| 10.0% | 2905 | 1387 | 1466 | 52 | 48.6% | -7.1% | 1.8% | -4.02 |

## Total bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 9370 | 4716 | 4571 | 83 | 50.8% | -3.0% | 1.0% | -3.11 |
| 0.0% | 7323 | 3710 | 3546 | 67 | 51.1% | -2.4% | 1.1% | -2.15 |
| 2.0% | 6499 | 3278 | 3161 | 60 | 50.9% | -2.8% | 1.2% | -2.38 |
| 4.0% | 5712 | 2904 | 2755 | 53 | 51.3% | -2.0% | 1.3% | -1.62 |
| 6.0% | 4957 | 2539 | 2370 | 48 | 51.7% | -1.3% | 1.3% | -0.94 |
| 8.0% | 4233 | 2147 | 2045 | 41 | 51.2% | -2.2% | 1.5% | -1.52 |
| 10.0% | 3599 | 1808 | 1760 | 31 | 50.7% | -3.3% | 1.6% | -2.05 |

## Moneyline bets at the closing price

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 9069 | 4739 | 4330 | 0 | 52.3% | -6.4% | 1.3% | -5.13 |
| 0.0% | 6082 | 2722 | 3360 | 0 | 44.8% | -8.7% | 1.7% | -5.17 |
| 2.0% | 5017 | 2041 | 2976 | 0 | 40.7% | -10.1% | 1.9% | -5.18 |
| 4.0% | 4233 | 1595 | 2638 | 0 | 37.7% | -11.9% | 2.2% | -5.42 |
| 6.0% | 3606 | 1286 | 2320 | 0 | 35.7% | -12.5% | 2.5% | -5.07 |
| 8.0% | 3069 | 1042 | 2027 | 0 | 34.0% | -13.0% | 2.7% | -4.77 |
| 10.0% | 2599 | 842 | 1757 | 0 | 32.4% | -13.0% | 3.1% | -4.20 |

Each row bets the side the sim prefers whenever its expected value at the real price is at least `min_ev`. `z` is ROI divided by its standard error; |z| < 2 is indistinguishable from luck.

## The daily card's rule, replayed

Tier B plays blend the sim with the market and need EV of +1% or more.

Training seasons (2021-2021):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 10 | 0.4 | 50.0% | -4.3% | -0.13 |
| total | close | 6 | 0.2 | 66.7% | 27.2% | +0.68 |
| ml | close | 4 | 0.2 | 0.0% | -100.0% |  |

Test seasons (2022-2026):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 24 | 0.3 | 59.1% | 11.7% | +0.62 |
| total | close | 7 | 0.1 | 71.4% | 36.4% | +1.03 |
| ml | close | 18 | 0.2 | 5.6% | -38.9% | -0.64 |
