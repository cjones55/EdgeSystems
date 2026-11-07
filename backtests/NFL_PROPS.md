# NFL player props backtest

Fitted on 2017-2020, scored on 2021-2026 (walk-forward: every projection uses only earlier games).

Chosen settings: form half-life 4 games, opponent weight 0, implied-points power 0.3. Probability recalibration sigmoid(0.91 x logit(p) -0.06), fitted on the training seasons.

## Projection accuracy vs a last-5-games average

| stat | player-games | mean actual | model error (MAE) | last-5 error | improvement |
|---|---|---|---|---|---|
| Passing yards | 3059 | 215.6 | 66.66 | 69.36 | +3.9% |
| Pass attempts | 3071 | 30.2 | 8.09 | 8.32 | +2.8% |
| Completions | 2989 | 19.8 | 5.47 | 5.69 | +4.0% |
| Passing TDs | 2976 | 1.4 | 0.91 | 0.94 | +3.7% |
| Interceptions | 3064 | 0.7 | 0.71 | 0.73 | +2.2% |
| Rushing yards | 8069 | 38.1 | 22.02 | 22.71 | +3.0% |
| Rush attempts | 6676 | 9.8 | 4.02 | 4.12 | +2.3% |
| Receptions | 17138 | 3.2 | 1.62 | 1.67 | +2.9% |
| Receiving yards | 18728 | 33.8 | 21.38 | 22.21 | +3.7% |
| Rush + rec yards | 16043 | 51.4 | 27.34 | 28.44 | +3.9% |

## Calibration of P(over) at proxy lines

No free history of sportsbook prop lines exists, so each prop is tested at a stand-in line: the player's median over his previous 8 games, rounded to x.5 (anytime TD at 0.5). This checks whether the probabilities mean what they say. It cannot show an edge over real lines, which are sharper than a median.

| stat | props | avg P(over) | actual over rate | log loss | log loss at 50/50 |
|---|---|---|---|---|---|
| Passing yards | 3059 | 48.0% | 47.8% | 0.6674 | 0.6931 |
| Pass attempts | 3071 | 44.1% | 47.2% | 0.6777 | 0.6931 |
| Completions | 2989 | 43.6% | 46.5% | 0.6767 | 0.6931 |
| Passing TDs | 2976 | 36.7% | 36.2% | 0.6265 | 0.6931 |
| Interceptions | 3064 | 35.7% | 35.2% | 0.6211 | 0.6931 |
| Rushing yards | 8069 | 44.1% | 45.2% | 0.6675 | 0.6931 |
| Rush attempts | 6676 | 46.1% | 44.2% | 0.6592 | 0.6931 |
| Receptions | 17138 | 42.5% | 40.6% | 0.6533 | 0.6931 |
| Receiving yards | 18728 | 45.7% | 45.5% | 0.6673 | 0.6931 |
| Rush + rec yards | 16043 | 44.7% | 45.4% | 0.6671 | 0.6931 |
| Anytime TD | 18701 | 27.1% | 25.6% | 0.5441 | 0.5685 |

All stats pooled:

| model P(over) | props | predicted | actual |
|---|---|---|---|
| (0.0, 0.2] | 7481 | 15.8% | 16.2% |
| (0.2, 0.35] | 23217 | 29.0% | 27.5% |
| (0.35, 0.45] | 32064 | 40.2% | 40.2% |
| (0.45, 0.55] | 25438 | 49.4% | 49.9% |
| (0.55, 0.65] | 9061 | 58.9% | 58.1% |
| (0.65, 0.8] | 2902 | 70.4% | 64.8% |
| (0.8, 1.0] | 351 | 86.3% | 76.6% |

Because the top buckets run hot, the card caps any prop probability at 72% before blending with the market.

**Reading this:** the projections should beat a simple recent average, and the predicted over rates should track the actual ones. Real prop lines are set by books with the same information plus injury and role news, so real edges will be far smaller than the gaps a proxy line shows. The tracker grades live props against real lines.