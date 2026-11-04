# NFL backtest

Data: 2011 to 2021 seasons, open + close lines. 2,573 games simulated, 1,404 in the held-out test seasons (2016-2021). Ratings were tuned on 2011-2015 only, and every prediction only uses games played before that date.

## Win probability vs the closing moneyline

| source | log_loss | brier |
|---|---|---|
| EdgeSim sim | 0.633 | 0.221 |
| Closing line (no-vig) | 0.608 | 0.210 |
| Coin flip | 0.693 | 0.250 |

Lower is better.

## Calibration of home win probability

| bucket | n | predicted | actual |
|---|---|---|---|
| 0.0-0.1 | 2 | 9.0% | 50.0% |
| 0.1-0.2 | 18 | 17.2% | 27.8% |
| 0.2-0.3 | 77 | 25.7% | 24.7% |
| 0.3-0.4 | 146 | 35.5% | 35.6% |
| 0.4-0.5 | 264 | 45.0% | 42.8% |
| 0.5-0.6 | 331 | 55.1% | 55.9% |
| 0.6-0.7 | 306 | 64.9% | 64.1% |
| 0.7-0.8 | 174 | 74.2% | 75.9% |
| 0.8-0.9 | 72 | 83.6% | 88.9% |
| 0.9-1.0 | 10 | 92.1% | 80.0% |

## Spread bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 1404 | 698 | 668 | 38 | 51.1% | -2.4% | 2.5% | -0.95 |
| 0.0% | 1092 | 529 | 532 | 31 | 49.9% | -4.7% | 2.8% | -1.64 |
| 2.0% | 948 | 461 | 462 | 25 | 49.9% | -4.5% | 3.1% | -1.48 |
| 4.0% | 831 | 396 | 412 | 23 | 49.0% | -6.3% | 3.3% | -1.92 |
| 6.0% | 698 | 341 | 343 | 14 | 49.9% | -4.7% | 3.6% | -1.32 |
| 8.0% | 587 | 284 | 291 | 12 | 49.4% | -5.6% | 3.9% | -1.43 |
| 10.0% | 478 | 227 | 241 | 10 | 48.5% | -7.2% | 4.3% | -1.68 |

## Spread bets at the opening line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 1378 | 656 | 677 | 45 | 49.2% | -5.9% | 2.5% | -2.31 |
| 0.0% | 1023 | 503 | 483 | 37 | 51.0% | -2.5% | 2.9% | -0.86 |
| 2.0% | 892 | 442 | 418 | 32 | 51.4% | -1.8% | 3.1% | -0.58 |
| 4.0% | 755 | 372 | 358 | 25 | 51.0% | -2.6% | 3.4% | -0.77 |
| 6.0% | 648 | 321 | 304 | 23 | 51.4% | -1.9% | 3.7% | -0.51 |
| 8.0% | 535 | 257 | 257 | 21 | 50.0% | -4.4% | 4.0% | -1.08 |
| 10.0% | 450 | 216 | 213 | 21 | 50.3% | -3.7% | 4.4% | -0.84 |

## Total bets at the closing line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 1404 | 669 | 722 | 13 | 48.1% | -8.1% | 2.5% | -3.20 |
| 0.0% | 1103 | 528 | 563 | 12 | 48.4% | -7.5% | 2.9% | -2.63 |
| 2.0% | 985 | 475 | 498 | 12 | 48.8% | -6.7% | 3.0% | -2.22 |
| 4.0% | 870 | 418 | 440 | 12 | 48.7% | -6.9% | 3.2% | -2.15 |
| 6.0% | 760 | 355 | 394 | 11 | 47.4% | -9.4% | 3.4% | -2.73 |
| 8.0% | 645 | 306 | 328 | 11 | 48.3% | -7.7% | 3.7% | -2.07 |
| 10.0% | 568 | 272 | 288 | 8 | 48.6% | -7.2% | 4.0% | -1.80 |

## Total bets at the opening line

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 1401 | 683 | 691 | 27 | 49.7% | -5.0% | 2.5% | -1.98 |
| 0.0% | 1120 | 543 | 550 | 27 | 49.7% | -5.0% | 2.8% | -1.79 |
| 2.0% | 978 | 469 | 484 | 25 | 49.2% | -5.9% | 3.0% | -1.96 |
| 4.0% | 863 | 410 | 429 | 24 | 48.9% | -6.5% | 3.2% | -2.03 |
| 6.0% | 754 | 361 | 371 | 22 | 49.3% | -5.7% | 3.4% | -1.66 |
| 8.0% | 652 | 314 | 317 | 21 | 49.8% | -4.8% | 3.7% | -1.31 |
| 10.0% | 552 | 268 | 268 | 16 | 50.0% | -4.4% | 4.0% | -1.10 |

## Moneyline bets at the closing price

| min_ev | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| all games | 1404 | 538 | 862 | 4 | 38.4% | -6.6% | 3.8% | -1.76 |
| 0.0% | 1186 | 431 | 752 | 3 | 36.4% | -5.2% | 4.3% | -1.21 |
| 2.0% | 1087 | 378 | 706 | 3 | 34.9% | -5.3% | 4.6% | -1.15 |
| 4.0% | 977 | 324 | 650 | 3 | 33.3% | -5.7% | 4.9% | -1.16 |
| 6.0% | 897 | 291 | 604 | 2 | 32.5% | -5.5% | 5.2% | -1.06 |
| 8.0% | 816 | 260 | 555 | 1 | 31.9% | -4.3% | 5.6% | -0.76 |
| 10.0% | 754 | 234 | 519 | 1 | 31.1% | -4.4% | 5.9% | -0.75 |

Each row bets the side the sim prefers whenever its expected value at the real price is at least `min_ev`. `z` is ROI divided by its standard error; |z| < 2 is indistinguishable from luck.

## The daily card's rule, replayed

Tier B plays blend the sim with the market and need EV of +1% or more.

Training seasons (2011-2015):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | open | 6 | 0.5 | 66.7% | 27.3% | +0.68 |
| total | open | 0 | 0.0 |  |  |  |
| spread | close | 7 | 0.6 | 71.4% | 36.4% | +1.03 |
| total | close | 1 | 0.1 | 100.0% | 90.9% |  |
| ml | close | 10 | 0.9 | 40.0% | 86.5% | +1.07 |

Test seasons (2016-2021):

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | open | 14 | 1.0 | 57.1% | 9.1% | +0.35 |
| total | open | 1 | 0.1 | 100.0% | 90.9% |  |
| spread | close | 16 | 1.1 | 37.5% | -28.4% | -1.19 |
| total | close | 0 | 0.0 |  |  |  |
| ml | close | 15 | 1.1 | 6.7% | -30.1% | -0.43 |

## Closing line value: model picks at the opening line

| model_vs_open_pts | bets | beat_close | lost_to_close | avg_clv_pts |
|---|---|---|---|---|
| >= 1 | 1031 | 52.4% | 31.8% | 0.511 |
| >= 2 | 691 | 56.2% | 28.1% | 0.705 |
| >= 3 | 447 | 57.9% | 27.5% | 0.865 |

How often the market moved toward the sim's side between open and close (the rest did not move).

## Line moves ('sharp money' proxy)

| min_move_pts | games | cover_vs_close | roi_at_close | cover_vs_open | roi_at_open |
|---|---|---|---|---|---|
| 1.000 | 824 | 48.1% | -8.1% | 56.7% | 8.0% |
| 1.500 | 594 | 49.7% | -5.1% | 59.4% | 13.1% |
| 2.000 | 423 | 50.6% | -3.4% | 62.6% | 19.0% |
| 3.000 | 177 | 50.6% | -3.4% | 65.7% | 25.2% |

The side a line moved toward, graded against the opening number and against the closing number.
