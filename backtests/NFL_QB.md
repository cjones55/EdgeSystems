# NFL quarterback adjustment backtest


Each game's expected points are adjusted by the gap between the expected starter's value (EPA per game, shrunk toward a typical backup) and the QB play already baked into the team's rating. Fitted on 2016-2018, tested on 2019-2021.

Fitted scale: 2.00 (the cap) points per point of EPA gap.

| test games | margin error (RMSE) without | with adjustment | change |
|---|---|---|---|
| all 717 | 13.30 | 13.17 | +1.0% |
| 42 with a QB gap of 4+ EPA | 14.56 | 14.00 | +3.9% |
The adjustment is applied to live NFL cards using ESPN's injury report: if a team's recent starter is listed out, doubtful or on IR, his most recent healthy replacement is assumed to start.
