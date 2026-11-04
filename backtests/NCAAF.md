# College football backtest

Data: 2015 to 2026 FBS games, closing lines (opening lines and moneylines for about half). 7,633 games simulated, 4,270 in the held-out test seasons (2020-2026). Ratings were tuned on 2015-2019 only, and every prediction only uses games played before that date.

## Win probability vs the closing moneyline

| source | log_loss | brier |
|---|---|---|
| EdgeSim sim | 0.569 | 0.193 |
| Closing line (no-vig) | 0.548 | 0.185 |
| Coin flip | 0.693 | 0.250 |

Lower is better.

## Calibration of home win probability

| bucket | n | predicted | actual |
|---|---|---|---|
| 0.0-0.1 | 103 | 5.9% | 7.8% |
| 0.1-0.2 | 213 | 15.6% | 12.7% |
| 0.2-0.3 | 282 | 25.7% | 28.7% |
| 0.3-0.4 | 400 | 35.0% | 35.0% |
| 0.4-0.5 | 464 | 45.1% | 43.1% |
| 0.5-0.6 | 530 | 55.1% | 50.6% |
| 0.6-0.7 | 569 | 65.2% | 60.6% |
| 0.7-0.8 | 561 | 75.1% | 71.8% |
| 0.8-0.9 | 566 | 85.0% | 80.9% |
| 0.9-1.0 | 582 | 95.0% | 91.9% |

## Spread bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 4270 | 2054 | 2140 | 76 | 49.0% | -6.4% | 1.4% | -4.41 |
| 0.0% | 3663 | 1767 | 1828 | 68 | 49.2% | -6.1% | 1.6% | -3.87 |
| 2.0% | 3376 | 1624 | 1688 | 64 | 49.0% | -6.3% | 1.6% | -3.85 |
| 4.0% | 3117 | 1499 | 1559 | 59 | 49.0% | -6.3% | 1.7% | -3.72 |
| 6.0% | 2845 | 1380 | 1412 | 53 | 49.4% | -5.5% | 1.8% | -3.12 |
| 8.0% | 2589 | 1240 | 1299 | 50 | 48.8% | -6.6% | 1.9% | -3.57 |
| 10.0% | 2350 | 1123 | 1178 | 49 | 48.8% | -6.7% | 1.9% | -3.43 |

## Spread bets at the opening line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 3856 | 1920 | 1882 | 54 | 50.5% | -3.5% | 1.5% | -2.32 |
| 0.0% | 3262 | 1629 | 1585 | 48 | 50.7% | -3.2% | 1.7% | -1.92 |
| 2.0% | 3026 | 1507 | 1477 | 42 | 50.5% | -3.5% | 1.7% | -2.05 |
| 4.0% | 2772 | 1376 | 1357 | 39 | 50.3% | -3.8% | 1.8% | -2.13 |
| 6.0% | 2534 | 1266 | 1230 | 38 | 50.7% | -3.1% | 1.9% | -1.66 |
| 8.0% | 2283 | 1149 | 1099 | 35 | 51.1% | -2.4% | 2.0% | -1.20 |
| 10.0% | 2061 | 1027 | 1002 | 32 | 50.6% | -3.3% | 2.1% | -1.59 |

## Total bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 4270 | 2039 | 2183 | 48 | 48.3% | -7.7% | 1.5% | -5.31 |
| 0.0% | 3583 | 1713 | 1834 | 36 | 48.3% | -7.7% | 1.6% | -4.87 |
| 2.0% | 3263 | 1552 | 1676 | 35 | 48.1% | -8.1% | 1.7% | -4.89 |
| 4.0% | 2990 | 1416 | 1542 | 32 | 47.9% | -8.5% | 1.7% | -4.91 |
| 6.0% | 2676 | 1250 | 1395 | 31 | 47.3% | -9.7% | 1.8% | -5.28 |
| 8.0% | 2427 | 1124 | 1275 | 28 | 46.9% | -10.4% | 1.9% | -5.42 |
| 10.0% | 2173 | 1004 | 1143 | 26 | 46.8% | -10.6% | 2.0% | -5.22 |

## Total bets at the opening line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 3861 | 1924 | 1905 | 32 | 50.2% | -4.0% | 1.5% | -2.64 |
| 0.0% | 3207 | 1584 | 1596 | 27 | 49.8% | -4.9% | 1.7% | -2.90 |
| 2.0% | 2938 | 1450 | 1462 | 26 | 49.8% | -4.9% | 1.8% | -2.79 |
| 4.0% | 2683 | 1317 | 1341 | 25 | 49.5% | -5.4% | 1.8% | -2.92 |
| 6.0% | 2406 | 1190 | 1194 | 22 | 49.9% | -4.7% | 1.9% | -2.41 |
| 8.0% | 2127 | 1040 | 1066 | 21 | 49.4% | -5.7% | 2.1% | -2.75 |
| 10.0% | 1875 | 924 | 935 | 16 | 49.7% | -5.1% | 2.2% | -2.31 |

## Moneyline bets at the closing price

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 3670 | 1943 | 1727 | 0 | 52.9% | -1.9% | 2.1% | -0.91 |
| 0.0% | 2931 | 1428 | 1503 | 0 | 48.7% | -0.9% | 2.6% | -0.36 |
| 2.0% | 2560 | 1174 | 1386 | 0 | 45.9% | -0.9% | 2.8% | -0.30 |
| 4.0% | 2270 | 982 | 1288 | 0 | 43.3% | -0.5% | 3.1% | -0.15 |
| 6.0% | 2027 | 839 | 1188 | 0 | 41.4% | -0.8% | 3.4% | -0.23 |
| 8.0% | 1825 | 729 | 1096 | 0 | 39.9% | 0.3% | 3.7% | +0.07 |
| 10.0% | 1645 | 625 | 1020 | 0 | 38.0% | 0.2% | 4.0% | +0.05 |

Each row bets the side the sim prefers whenever its expected value at the real price is at least `min_ev`. `z` is ROI divided by its standard error; |z| < 2 is indistinguishable from luck.

## The daily card's rule, replayed

Tier B plays blend the sim with the market and need EV of +1% or more.

Training seasons (2015-2019):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 218 | 6.5 | 43.1% | -17.6% | -2.76 |
| total | close | 50 | 1.5 | 36.0% | -31.3% | -2.39 |

Test seasons (2020-2026):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | open | 174 | 4.5 | 48.0% | -8.4% | -1.16 |
| total | open | 25 | 0.6 | 52.0% | -0.7% | -0.04 |
| spread | close | 236 | 5.5 | 47.6% | -8.9% | -1.45 |
| total | close | 26 | 0.6 | 38.5% | -26.6% | -1.43 |
| ml | close | 61 | 1.7 | 27.9% | 18.6% | +0.71 |

## Closing line value: model picks at the opening line

| model_vs_open_pts | bets | beat_close | lost_to_close | avg_clv_pts |
|---|---|---|---|---|
| >= 1 | 3306 | 46.4% | 42.0% | 0.195 |
| >= 2 | 2737 | 46.8% | 41.6% | 0.228 |
| >= 3 | 2171 | 48.3% | 40.9% | 0.284 |

How often the market moved toward the sim's side between open and close (the rest did not move).

## Line moves ('sharp money' proxy)

| min_move_pts | games | cover_vs_close | roi_at_close | cover_vs_open | roi_at_open |
|---|---|---|---|---|---|
| 1.000 | 2237 | 50.2% | -4.1% | 55.8% | 6.4% |
| 1.500 | 1568 | 51.7% | -1.3% | 58.4% | 11.3% |
| 2.000 | 1077 | 51.6% | -1.4% | 59.1% | 12.8% |
| 3.000 | 504 | 52.0% | -0.6% | 61.3% | 16.9% |

The side a line moved toward, graded against the opening number and against the closing number.
