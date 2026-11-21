# Recovery audit for one recorded run

Workflow demo-research-run (24 steps), recovery policy used: rollback.

## What the run spent

| component | cost |
|---|---|
| base (one clean pass) | 24.00 |
| checkpoint spend | 12.00 |
| rework | 18.00 |
| penalty | 0.00 |
| **total** | **54.00** |

## Checkpoint use

Taken: [6, 12, 18, 24]. Used by a fault: [12, 18, 24]. Never used: [6] (wasted spend 3.00 on this run).

## Counterfactual (same faults, optimal placement)

Optimal checkpoints [6, 7, 15, 16, 24] would have spent 15.00 on checkpoints and 10.00 on rework, total 49.00. Excess of the recorded run: 5.00 (spend difference -3.00, rework difference +8.00).

## Recommendation

1 of 4 checkpoints taken were never used by a fault on this run (spend 3.00 bought no bounded rework this time). Faults forced 18.00 of rework; under the cost-optimal placement [6, 7, 15, 16, 24] the same faults would have cost 10.00. Total excess over the counterfactual: 5.00. Re-place checkpoints at [6, 7, 15, 16, 24] to minimise expected cost on this workflow shape.

