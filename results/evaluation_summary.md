# Evaluation summary

Seeded evaluation: 16 workflow scenarios (lengths 8-64, hazard profiles uniform_low / uniform_high / spiky / back_loaded), checkpoint cost 3.0, failure penalty 600.0, repair rate 0.25, fault persistence 0.30, error threshold 0.05. The tool estimates hazards from 300 logged runs; Monte Carlo uses 300 runs per policy per scenario.

## Aggregates

- Mean saving of the optimal placement vs restart-from-scratch: 40.5%
- Mean saving of the optimal placement vs checkpointing every step: 51.2%
- Best fixed interval costs 1.7% more than the optimal placement on average, and 4.9% more on spiky hazard profiles, where uniform intervals cannot follow the risk.
- Tool regret (placement optimised on estimated hazards, priced on true hazards): 0.13% above the oracle placement on average.
- Tool fixed-interval choice differs from the oracle interval by 0.1 steps on average.
- Analytic expected costs vs seeded Monte Carlo means: 2.4% mean absolute deviation.

## Scenarios

| scenario | restart | every step | oracle fixed (m) | oracle optimal | tool optimal | tool regret % |
|---|---|---|---|---|---|---|
| uniform_low-8 | 9.11 | 31.91 | 12.11 (8) | 12.11 | 12.11 | 0.00 |
| uniform_high-8 | 19.00 | 34.60 | 19.61 (4) | 19.61 | 19.61 | 0.00 |
| spiky-8 | 15.13 | 34.02 | 17.98 (4) | 17.98 | 17.98 | -0.00 |
| back_loaded-8 | 15.24 | 33.44 | 17.40 (5) | 17.40 | 17.52 | 0.72 |
| uniform_low-16 | 25.79 | 67.05 | 27.94 (8) | 27.94 | 27.94 | 0.00 |
| uniform_high-16 | 68.55 | 67.63 | 37.20 (4) | 37.20 | 37.29 | 0.24 |
| spiky-16 | 45.53 | 65.88 | 32.25 (6) | 31.29 | 31.29 | -0.00 |
| back_loaded-16 | 50.44 | 66.26 | 33.44 (6) | 33.13 | 33.24 | 0.31 |
| uniform_low-32 | 61.29 | 128.76 | 48.86 (11) | 48.85 | 48.85 | 0.01 |
| uniform_high-32 | 532.45 | 136.01 | 75.38 (4) | 74.99 | 75.08 | 0.11 |
| spiky-32 | 407.72 | 134.91 | 73.60 (4) | 68.09 | 68.27 | 0.27 |
| back_loaded-32 | 305.50 | 133.37 | 69.88 (5) | 67.23 | 67.23 | 0.00 |
| uniform_low-64 | 258.70 | 264.45 | 106.17 (11) | 106.06 | 106.44 | 0.35 |
| uniform_high-64 | 3684.53 | 270.43 | 147.89 (5) | 147.76 | 147.88 | 0.08 |
| spiky-64 | 3281.27 | 266.88 | 142.61 (5) | 129.33 | 129.33 | 0.00 |
| back_loaded-64 | 2895.76 | 267.06 | 138.07 (5) | 134.52 | 134.62 | 0.07 |

## Fault recovery (severity 0.8, threshold 0.05)

| repair rate | analytic half-life | measured half-life (steps) | analytic recovery | measured recovery (steps) |
|---|---|---|---|---|
| 0.10 | 6.58 | 7 | 26.32 | 27 |
| 0.25 | 2.41 | 3 | 9.64 | 10 |
| 0.50 | 1.00 | 1 | 4.00 | 4 |
