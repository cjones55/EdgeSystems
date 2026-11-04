# NHL backtest

Data: 2011-12 to 2021-22, open + close moneylines, closing puck line and total with prices. 11,984 games simulated, 6,303 in the held-out test seasons (2016-2021). Ratings were tuned on 2011-2015 only, and every prediction only uses games played before that date.

## Win probability vs the closing moneyline

| source | log_loss | brier |
|---|---|---|
| EdgeSim sim | 0.671 | 0.239 |
| Closing line (no-vig) | 0.665 | 0.236 |
| Coin flip | 0.693 | 0.250 |

Lower is better.

## Calibration of home win probability

| bucket | n | predicted | actual |
|---|---|---|---|
| 0.2-0.3 | 4 | 29.6% | 0.0% |
| 0.3-0.4 | 147 | 37.3% | 34.0% |
| 0.4-0.5 | 1580 | 46.1% | 42.0% |
| 0.5-0.6 | 3160 | 55.0% | 55.9% |
| 0.6-0.7 | 1333 | 63.6% | 66.1% |
| 0.7-0.8 | 79 | 71.7% | 79.7% |

## Puck line bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 6301 | 3716 | 2585 | 0 | 59.0% | -5.1% | 1.1% | -4.78 |
| 0.0% | 4526 | 2675 | 1851 | 0 | 59.1% | -4.5% | 1.3% | -3.55 |
| 2.0% | 3658 | 2133 | 1525 | 0 | 58.3% | -4.8% | 1.4% | -3.41 |
| 4.0% | 2810 | 1614 | 1196 | 0 | 57.4% | -4.7% | 1.6% | -2.90 |
| 6.0% | 2159 | 1210 | 949 | 0 | 56.0% | -4.7% | 1.9% | -2.45 |
| 8.0% | 1635 | 907 | 728 | 0 | 55.5% | -3.9% | 2.2% | -1.75 |
| 10.0% | 1246 | 684 | 562 | 0 | 54.9% | -2.0% | 2.6% | -0.74 |

## Total bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 6301 | 2945 | 2967 | 389 | 49.8% | -3.6% | 1.2% | -2.99 |
| 0.0% | 3599 | 1657 | 1692 | 250 | 49.5% | -3.4% | 1.6% | -2.17 |
| 2.0% | 2529 | 1147 | 1205 | 177 | 48.8% | -4.1% | 1.9% | -2.16 |
| 4.0% | 1728 | 784 | 825 | 119 | 48.7% | -3.7% | 2.3% | -1.56 |
| 6.0% | 1107 | 512 | 516 | 79 | 49.8% | -0.7% | 3.0% | -0.22 |
| 8.0% | 670 | 309 | 313 | 48 | 49.7% | 0.5% | 4.0% | +0.12 |
| 10.0% | 403 | 196 | 184 | 23 | 51.6% | 5.9% | 5.4% | +1.09 |

## Moneyline bets at the closing price

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 6303 | 2774 | 3529 | 0 | 44.0% | -3.3% | 1.4% | -2.29 |
| 0.0% | 4954 | 2082 | 2872 | 0 | 42.0% | -3.8% | 1.7% | -2.28 |
| 2.0% | 4114 | 1650 | 2464 | 0 | 40.1% | -5.5% | 1.9% | -2.91 |
| 4.0% | 3400 | 1342 | 2058 | 0 | 39.5% | -4.4% | 2.1% | -2.08 |
| 6.0% | 2817 | 1095 | 1722 | 0 | 38.9% | -3.2% | 2.4% | -1.34 |
| 8.0% | 2294 | 868 | 1426 | 0 | 37.8% | -2.7% | 2.7% | -1.02 |
| 10.0% | 1857 | 688 | 1169 | 0 | 37.0% | -2.0% | 3.1% | -0.66 |

Each row bets the side the sim prefers whenever its expected value at the real price is at least `min_ev`. `z` is ROI divided by its standard error; |z| < 2 is indistinguishable from luck.

## The daily card's rule, replayed

Tier B plays blend the sim with the market and need EV of +1% or more.

Training seasons (2011-2015):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 0 | 0.0 |  |  |  |
| total | close | 0 | 0.0 |  |  |  |
| ml | close | 1 | 0.0 | 100.0% | 200.0% |  |

Test seasons (2016-2021):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | close | 11 | 0.2 | 81.8% | 104.7% | +3.02 |
| total | close | 4 | 0.1 | 50.0% | 290.5% | +1.29 |
| ml | close | 26 | 0.4 | 42.3% | 68.5% | +1.72 |

## Closing line value: model picks at the opening moneyline

| bets | beat_close_pct | lost_to_close_pct | avg_clv_prob |
|---|---|---|---|
| 4121 | 51.7% | 45.4% | 0.004 |

## Moneyline moves ('sharp money' proxy)

| min_move_prob | games | win_pct | roi_at_close | roi_at_open |
|---|---|---|---|---|
| 0.020 | 2859 | 53.9% | -1.7% | 5.9% |
| 0.030 | 1726 | 54.6% | -1.1% | 9.2% |
| 0.050 | 547 | 56.5% | -0.7% | 18.6% |
