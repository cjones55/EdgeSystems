# EdgeSim backtest

Generated 2026-11-04 16:27. Games: 13,010 simulated (7,076 in the held-out test seasons 2016-2021). Every prediction is walk-forward: the model only saw games before that date. Rating parameters were tuned on 2012-2015 only.

## Bottom line

* **The simulator works**: win probabilities are well calibrated on six held-out seasons, and prop distributions are close to calibrated.
* **It does not beat NBA closing lines by itself.** Betting every model edge at -110 loses about the vig.
* **Its opinions do carry some information**: when it disagrees with the opening line, the market moves its way 46% of the time vs 39% against.
* **Line moves are informative but get priced in**: the side a big move went toward covered the opening number 58-64% of the time, but only ~50% at the close. Edge from sharp action exists only if you get the number before the move finishes.
* **Public betting % cannot be backtested** with free data. The forward tracker (`python -m edgesim grade`) is how sharp-signal plays get proven or dropped.

## 1. Win probability vs the closing moneyline (test seasons)

| source | log_loss | brier |
|---|---|---|
| EdgeSim sim | 0.624 | 0.217 |
| Closing line (no-vig) | 0.603 | 0.208 |
| Coin flip | 0.693 | 0.250 |

Lower is better. The sim is a genuinely informative forecaster, but the closing line is better, as it is for virtually every public model.

### Calibration of home win probability (test seasons)

| bucket | n | predicted | actual |
|---|---|---|---|
| 0.0-0.1 | 4 | 7.7% | 25.0% |
| 0.1-0.2 | 109 | 16.9% | 16.5% |
| 0.2-0.3 | 349 | 25.9% | 28.4% |
| 0.3-0.4 | 783 | 35.5% | 34.4% |
| 0.4-0.5 | 1169 | 45.3% | 45.6% |
| 0.5-0.6 | 1507 | 55.1% | 55.1% |
| 0.6-0.7 | 1471 | 64.8% | 65.1% |
| 0.7-0.8 | 1085 | 74.6% | 74.2% |
| 0.8-0.9 | 521 | 84.2% | 84.6% |
| 0.9-1.0 | 78 | 91.8% | 89.7% |

When the sim says 70%, the home team wins about 70% of the time: the Monte Carlo is well calibrated.

## 2. Spread bets at the close line, -110 (test seasons)

| min_edge | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| 0.0% | 7076 | 3453 | 3492 | 131 | 49.7% | -5.0% | 1.1% | -4.44 |
| 2.0% | 5863 | 2863 | 2891 | 109 | 49.8% | -4.9% | 1.2% | -3.98 |
| 4.0% | 4735 | 2308 | 2340 | 87 | 49.7% | -5.1% | 1.4% | -3.72 |
| 6.0% | 3683 | 1790 | 1819 | 74 | 49.6% | -5.2% | 1.6% | -3.34 |
| 8.0% | 2825 | 1367 | 1401 | 57 | 49.4% | -5.6% | 1.8% | -3.15 |
| 10.0% | 2126 | 1004 | 1076 | 46 | 48.3% | -7.7% | 2.0% | -3.75 |

## 3. Spread bets at the open line, -110 (test seasons)

| min_edge | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| 0.0% | 7020 | 3450 | 3450 | 120 | 50.0% | -4.5% | 1.1% | -3.96 |
| 2.0% | 5632 | 2792 | 2741 | 99 | 50.5% | -3.6% | 1.3% | -2.86 |
| 4.0% | 4362 | 2159 | 2126 | 77 | 50.4% | -3.7% | 1.4% | -2.61 |
| 6.0% | 3276 | 1600 | 1613 | 63 | 49.8% | -4.8% | 1.7% | -2.93 |
| 8.0% | 2431 | 1187 | 1198 | 46 | 49.8% | -4.9% | 1.9% | -2.55 |
| 10.0% | 1739 | 861 | 846 | 32 | 50.4% | -3.6% | 2.3% | -1.60 |

## 4. Total bets at the close line, -110 (test seasons)

| min_edge | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| 0.0% | 7076 | 3443 | 3536 | 97 | 49.3% | -5.7% | 1.1% | -5.09 |
| 2.0% | 5765 | 2782 | 2904 | 79 | 48.9% | -6.5% | 1.2% | -5.21 |
| 4.0% | 4587 | 2190 | 2329 | 68 | 48.5% | -7.4% | 1.4% | -5.27 |
| 6.0% | 3408 | 1632 | 1725 | 51 | 48.6% | -7.1% | 1.6% | -4.36 |
| 8.0% | 2440 | 1168 | 1240 | 32 | 48.5% | -7.3% | 1.9% | -3.80 |
| 10.0% | 1700 | 819 | 860 | 21 | 48.8% | -6.8% | 2.3% | -2.95 |

## 5. Total bets at the open line, -110 (test seasons)

| min_edge | bets | wins | losses | pushes | win_pct | roi | roi_se | z |
|---|---|---|---|---|---|---|---|---|
| 0.0% | 7066 | 3568 | 3420 | 78 | 51.1% | -2.5% | 1.1% | -2.21 |
| 2.0% | 5467 | 2763 | 2643 | 61 | 51.1% | -2.4% | 1.3% | -1.87 |
| 4.0% | 4087 | 2095 | 1948 | 44 | 51.8% | -1.1% | 1.5% | -0.72 |
| 6.0% | 2866 | 1454 | 1377 | 35 | 51.4% | -1.9% | 1.8% | -1.09 |
| 8.0% | 1929 | 979 | 923 | 27 | 51.5% | -1.7% | 2.2% | -0.79 |
| 10.0% | 1300 | 661 | 623 | 16 | 51.5% | -1.7% | 2.6% | -0.65 |

Break-even at -110 is 52.4%. `z` is ROI divided by its standard error: |z| < 2 is indistinguishable from luck.

## 6. The daily card's own rule, replayed

Model-edge plays (tier B) use the sim blended with the market and need EV of at least +1%. Replayed on the training and test seasons:

**Training seasons (2011-2015)**

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | open | 45 | 0.8 | 51.1% | -2.4% | -0.17 |
| total | open | 48 | 0.8 | 47.9% | -8.5% | -0.61 |
| spread | close | 65 | 1.1 | 51.6% | -1.5% | -0.13 |
| total | close | 54 | 0.9 | 48.1% | -7.9% | -0.62 |
| ml | close | 193 | 3.3 | 9.8% | -12.8% | -0.62 |

**Test seasons (2016-2021)**

| market | line | plays | plays_per_100_games | win_pct | roi | z |
|---|---|---|---|---|---|---|
| spread | open | 104 | 1.5 | 48.5% | -7.3% | -0.78 |
| total | open | 4 | 0.1 | 75.0% | 43.2% | +0.90 |
| spread | close | 154 | 2.2 | 48.7% | -6.9% | -0.91 |
| total | close | 10 | 0.1 | 30.0% | -42.7% | -1.47 |
| ml | close | 158 | 2.2 | 17.7% | 40.2% | +1.54 |

The blend is deliberately conservative, so model-only game plays are rare (a few per 100 games) and none of these results is distinguishable from luck (|z| < 2). Treat tier B game plays as unproven.

## 7. Closing line value: model picks at the opening line

| model_vs_open_pts | bets | beat_close | lost_to_close | avg_clv_pts | ats_at_open |
|---|---|---|---|---|---|
| >= 1 | 5361 | 46.2% | 38.9% | 0.160 | 50.2% |
| >= 2 | 3483 | 46.1% | 39.5% | 0.166 | 50.2% |
| >= 3 | 2148 | 46.0% | 39.1% | 0.190 | 49.6% |

The market moves toward the model's side more often than away from it (the remainder is no move), so the model has some real information the opening line lacks, but not enough to overcome the vig on its own.

## 8. Line-movement ('sharp money') proxy test

| min_move_pts | games | ats_if_bet_at_close | roi_at_close | ats_if_bet_at_open | roi_at_open |
|---|---|---|---|---|---|
| 1.000 | 4042 | 49.7% | -5.0% | 55.7% | 6.3% |
| 1.500 | 2504 | 49.9% | -4.6% | 57.6% | 9.9% |
| 2.000 | 1522 | 50.2% | -4.1% | 59.9% | 14.2% |
| 3.000 | 534 | 50.6% | -3.4% | 63.5% | 20.8% |

Sides the line moved toward cover the OPENING number ~57% of the time, but only ~50% against the closing number. Sharp moves carry real information, and it is priced in by the close. Value only exists if you act before the move finishes, which is why the app snapshots lines through the day and why the forward tracker measures CLV. Free historical public-betting % data does not exist, so ticket/money splits cannot be backtested here; they are tracked going forward instead.

## 9. Player props (test seasons 2022-23, 2023-24)

### Projection accuracy (mean absolute error)

| stat | rows | model_mae | season_avg_mae | model_bias |
|---|---|---|---|---|
| pts | 30806 | 5.251 | 5.307 | -0.201 |
| reb | 30806 | 2.044 | 2.081 | -0.021 |
| ast | 30806 | 1.547 | 1.567 | -0.006 |
| fg3m | 30806 | 1.019 | 1.021 | -0.017 |
| pra | 30806 | 6.710 | 6.823 | -0.217 |

### Calibration of P(over) at a proxy line

| stat | bucket | n | predicted | actual |
|---|---|---|---|---|
| pts | 0.2-0.3 | 396 | 27.8% | 23.2% |
| pts | 0.3-0.4 | 6005 | 36.5% | 34.8% |
| pts | 0.4-0.5 | 15031 | 45.1% | 45.3% |
| pts | 0.5-0.6 | 7750 | 54.0% | 54.2% |
| pts | 0.6-0.7 | 1444 | 63.4% | 60.2% |
| pts | 0.7-0.8 | 163 | 73.4% | 63.8% |
| reb | 0.2-0.3 | 1546 | 26.7% | 25.8% |
| reb | 0.3-0.4 | 7197 | 35.8% | 34.9% |
| reb | 0.4-0.5 | 11326 | 45.0% | 43.2% |
| reb | 0.5-0.6 | 7853 | 54.4% | 53.3% |
| reb | 0.6-0.7 | 2411 | 63.7% | 62.5% |
| reb | 0.7-0.8 | 392 | 73.4% | 72.2% |
| ast | 0.2-0.3 | 2228 | 26.5% | 26.5% |
| ast | 0.3-0.4 | 7913 | 35.6% | 35.3% |
| ast | 0.4-0.5 | 10692 | 44.9% | 44.5% |
| ast | 0.5-0.6 | 6318 | 54.3% | 52.6% |
| ast | 0.6-0.7 | 2861 | 64.3% | 62.4% |
| ast | 0.7-0.8 | 613 | 73.3% | 73.7% |
| fg3m | 0.2-0.3 | 5548 | 25.7% | 27.8% |
| fg3m | 0.3-0.4 | 7608 | 35.1% | 36.2% |
| fg3m | 0.4-0.5 | 6601 | 44.6% | 43.7% |
| fg3m | 0.5-0.6 | 4262 | 54.7% | 54.1% |
| fg3m | 0.6-0.7 | 2368 | 64.2% | 61.2% |
| fg3m | 0.7-0.8 | 775 | 73.9% | 71.1% |
| fg3m | 0.8-0.9 | 54 | 82.7% | 81.5% |
| pra | 0.2-0.3 | 436 | 27.7% | 25.5% |
| pra | 0.3-0.4 | 5852 | 36.5% | 36.0% |
| pra | 0.4-0.5 | 14314 | 45.1% | 46.0% |
| pra | 0.5-0.6 | 8400 | 54.1% | 53.4% |
| pra | 0.6-0.7 | 1647 | 63.3% | 61.2% |
| pra | 0.7-0.8 | 145 | 73.1% | 58.6% |

### Hit rate against proxy lines

| stat | threshold | bets | hit_rate | brier | brier_coinflip |
|---|---|---|---|---|---|
| pts | 0.550 | 18049 | 60.4% | 0.242 | 0.250 |
| pts | 0.580 | 11499 | 62.8% | 0.242 | 0.250 |
| pts | 0.620 | 5253 | 66.8% | 0.242 | 0.250 |
| reb | 0.550 | 20689 | 62.8% | 0.237 | 0.250 |
| reb | 0.580 | 14998 | 64.7% | 0.237 | 0.250 |
| reb | 0.620 | 8819 | 67.9% | 0.237 | 0.250 |
| ast | 0.550 | 21731 | 62.9% | 0.236 | 0.250 |
| ast | 0.580 | 16838 | 64.8% | 0.236 | 0.250 |
| ast | 0.620 | 11061 | 67.3% | 0.236 | 0.250 |
| fg3m | 0.550 | 25497 | 68.3% | 0.212 | 0.250 |
| fg3m | 0.580 | 22227 | 69.9% | 0.212 | 0.250 |
| fg3m | 0.620 | 17795 | 72.6% | 0.212 | 0.250 |
| pra | 0.550 | 17989 | 59.6% | 0.243 | 0.250 |
| pra | 0.580 | 11638 | 62.0% | 0.243 | 0.250 |
| pra | 0.620 | 5384 | 65.5% | 0.243 | 0.250 |

**Read this carefully:** historical sportsbook prop lines are not freely available, so the proxy line is the player's last-10 average rounded to x.5. Real books are much sharper than that (they know about injuries, role changes, and use their own models), so these hit rates are an upper bound, not an expected result. The fair test is the forward tracker with real lines.

Fitted prop parameters (from training seasons): `{"dispersion": {"pts": 8.0, "reb": 50.0, "ast": 30.0, "fg3m": 20.0, "stl": 20.0, "blk": 10.0, "tov": 100.0, "pra": 14.0, "pr": 14.0, "pa": 10.0, "ra": 50.0}, "calib": {"pts": 0.75, "reb": 0.9, "ast": 0.9, "fg3m": 1.15, "pra": 0.8}}`

Runtime: 25s