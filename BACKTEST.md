# EdgeSim backtest summary

Generated 2026-11-04 17:44. Full results per sport are in [backtests/](backtests/). Every prediction is walk-forward and every number below is from seasons the model was not tuned on.

| sport | test games | sim log loss | closing line log loss | worst calibration gap | ROI betting sim edges at close | line moved toward sim / away | big moves: at open vs at close |
|---|---|---|---|---|---|---|---|
| NBA | 7076 | 0.624 | 0.603 | 2.4% | -4.6% | 46% vs 39% | 58% vs 50% |
| NFL | 1404 | 0.633 | 0.608 | 2.2% | -4.7% | 52% vs 32% | 59% vs 50% |
| NHL | 6303 | 0.671 | 0.665 | 4.2% | -4.5% | 52% vs 45% | ROI +9% vs -1% |
| College football | 4270 | 0.569 | 0.548 | 4.6% | -6.1% | 46% vs 42% | 58% vs 52% |
| College basketball | 9370 | 0.553 | 0.542 | 1.9% | -4.1% | n/a (no opening lines) | n/a (no opening lines) |

## What this means

* **The simulators are well calibrated in every sport**: when a sim says 65%, it happens about 65% of the time (worst bucket gap shown above).
* **No sport's sim beats the closing line on its own.** Betting every sim edge at the closing number loses roughly the bookmaker's margin.
* **The sims do carry information the opening line lacks**: the market moves toward the sim's side more often than away from it, most strongly in the NFL.
* **Line moves are real signals, but only before they finish**: the side a big move went toward wins well above break-even at the opening number and about break-even at the close.
* **Public betting % has no free history**, so ticket/money splits are tracked going forward with `python -m edgesim grade` rather than backtested.

Runtime: 28s