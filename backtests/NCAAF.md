# College football backtest

Data: 2015 to 2026 FBS games, closing lines (opening lines and moneylines for about half). 7,638 games simulated, 4,275 in the held-out test seasons (2020-2026). Ratings were tuned on 2015-2019 only, and every prediction only uses games played before that date.

## Win probability vs the closing moneyline

| source | log_loss | brier |
|---|---|---|
| EdgeSim sim | 0.555 | 0.188 |
| Closing line (no-vig) | 0.548 | 0.185 |
| Coin flip | 0.693 | 0.250 |

Lower is better.

## Calibration of home win probability

| bucket | n | predicted | actual |
|---|---|---|---|
| 0.0-0.1 | 109 | 6.0% | 5.5% |
| 0.1-0.2 | 218 | 15.7% | 12.4% |
| 0.2-0.3 | 357 | 25.3% | 28.3% |
| 0.3-0.4 | 420 | 35.3% | 36.0% |
| 0.4-0.5 | 532 | 45.2% | 44.4% |
| 0.5-0.6 | 548 | 55.1% | 53.6% |
| 0.6-0.7 | 590 | 64.9% | 65.8% |
| 0.7-0.8 | 543 | 75.0% | 74.0% |
| 0.8-0.9 | 517 | 85.0% | 86.5% |
| 0.9-1.0 | 441 | 95.2% | 94.1% |

## Spread bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 4275 | 2074 | 2125 | 76 | 49.4% | -5.6% | 1.4% | -3.87 |
| 0.0% | 3261 | 1585 | 1615 | 61 | 49.5% | -5.3% | 1.7% | -3.22 |
| 2.0% | 2855 | 1384 | 1418 | 53 | 49.4% | -5.6% | 1.8% | -3.16 |
| 4.0% | 2482 | 1207 | 1229 | 46 | 49.5% | -5.3% | 1.9% | -2.80 |
| 6.0% | 2066 | 1003 | 1024 | 39 | 49.5% | -5.4% | 2.1% | -2.61 |
| 8.0% | 1741 | 857 | 851 | 33 | 50.2% | -4.1% | 2.3% | -1.82 |
| 10.0% | 1458 | 717 | 712 | 29 | 50.2% | -4.1% | 2.5% | -1.67 |

## Spread bets at the opening line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 3861 | 1950 | 1857 | 54 | 51.2% | -2.2% | 1.5% | -1.43 |
| 0.0% | 2815 | 1429 | 1346 | 40 | 51.5% | -1.7% | 1.8% | -0.93 |
| 2.0% | 2387 | 1230 | 1127 | 30 | 52.2% | -0.4% | 1.9% | -0.19 |
| 4.0% | 1977 | 1012 | 939 | 26 | 51.9% | -1.0% | 2.1% | -0.45 |
| 6.0% | 1636 | 840 | 776 | 20 | 52.0% | -0.8% | 2.3% | -0.32 |
| 8.0% | 1349 | 702 | 629 | 18 | 52.7% | 0.7% | 2.6% | +0.26 |
| 10.0% | 1107 | 572 | 518 | 17 | 52.5% | 0.2% | 2.8% | +0.06 |

## Total bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 4275 | 2104 | 2123 | 48 | 49.8% | -4.9% | 1.5% | -3.39 |
| 0.0% | 3128 | 1513 | 1578 | 37 | 48.9% | -6.5% | 1.7% | -3.82 |
| 2.0% | 2641 | 1281 | 1328 | 32 | 49.1% | -6.2% | 1.8% | -3.35 |
| 4.0% | 2216 | 1071 | 1116 | 29 | 49.0% | -6.4% | 2.0% | -3.19 |
| 6.0% | 1835 | 868 | 943 | 24 | 47.9% | -8.4% | 2.2% | -3.79 |
| 8.0% | 1494 | 713 | 759 | 22 | 48.4% | -7.4% | 2.5% | -3.03 |
| 10.0% | 1193 | 570 | 604 | 19 | 48.6% | -7.2% | 2.7% | -2.62 |

## Total bets at the opening line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 3866 | 2014 | 1820 | 32 | 52.5% | 0.3% | 1.5% | +0.18 |
| 0.0% | 2803 | 1465 | 1315 | 23 | 52.7% | 0.6% | 1.8% | +0.33 |
| 2.0% | 2322 | 1204 | 1096 | 22 | 52.3% | -0.1% | 2.0% | -0.03 |
| 4.0% | 1893 | 982 | 891 | 20 | 52.4% | 0.1% | 2.2% | +0.04 |
| 6.0% | 1498 | 781 | 700 | 17 | 52.7% | 0.7% | 2.4% | +0.27 |
| 8.0% | 1212 | 626 | 572 | 14 | 52.3% | -0.2% | 2.7% | -0.09 |
| 10.0% | 958 | 495 | 452 | 11 | 52.3% | -0.2% | 3.1% | -0.07 |

## Moneyline bets at the closing price

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 3674 | 1831 | 1843 | 0 | 49.8% | -4.3% | 2.1% | -1.99 |
| 0.0% | 2512 | 1043 | 1469 | 0 | 41.5% | -3.6% | 2.9% | -1.22 |
| 2.0% | 2158 | 828 | 1330 | 0 | 38.4% | -4.0% | 3.3% | -1.23 |
| 4.0% | 1863 | 681 | 1182 | 0 | 36.6% | -2.5% | 3.7% | -0.68 |
| 6.0% | 1636 | 580 | 1056 | 0 | 35.5% | -0.6% | 4.0% | -0.14 |
| 8.0% | 1433 | 497 | 936 | 0 | 34.7% | 1.0% | 4.4% | +0.23 |
| 10.0% | 1278 | 423 | 855 | 0 | 33.1% | 1.4% | 4.8% | +0.29 |

Each row bets the side the sim prefers whenever its expected value at the real price is at least `min_ev`. `z` is ROI divided by its standard error; |z| < 2 is indistinguishable from luck.

## The daily card's rule, replayed

Tier B plays blend the sim with the market and need EV of +1% or more.

Training seasons (2015-2019):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 85 | 2.5 | 48.2% | -7.9% | -0.76 |
| total | close | 7 | 0.2 | 28.6% | -45.5% | -1.29 |

Test seasons (2020-2026):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | open | 21 | 0.5 | 61.9% | 18.2% | +0.88 |
| total | open | 2 | 0.1 | 50.0% | -4.5% | -0.05 |
| spread | close | 38 | 0.9 | 31.6% | -39.7% | -2.72 |
| total | close | 2 | 0.0 | 0.0% | -100.0% |  |
| ml | close | 32 | 0.9 | 43.8% | 72.1% | +1.95 |

## Closing line value: model picks at the opening line

| model_vs_open_pts | bets | beat_close | lost_to_close | avg_clv_pts |
|---|---|---|---|---|
| >= 1 | 2889 | 48.7% | 40.0% | 0.281 |
| >= 2 | 1935 | 50.5% | 38.3% | 0.420 |
| >= 3 | 1252 | 53.5% | 36.4% | 0.606 |

How often the market moved toward the sim's side between open and close (the rest did not move).

## Line moves ('sharp money' proxy)

| min_move_pts | games | cover_vs_close | roi_at_close | cover_vs_open | roi_at_open |
|---|---|---|---|---|---|
| 1.000 | 2240 | 50.2% | -4.1% | 55.8% | 6.5% |
| 1.500 | 1570 | 51.7% | -1.3% | 58.3% | 11.3% |
| 2.000 | 1078 | 51.7% | -1.4% | 59.2% | 12.8% |
| 3.000 | 505 | 52.1% | -0.5% | 61.4% | 17.0% |

The side a line moved toward, graded against the opening number and against the closing number.
