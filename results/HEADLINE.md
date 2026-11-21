# Headline results

Checkpoint placement priced as an economic decision on 16 seeded workflow scenarios (8-64 steps, four hazard profiles), with real measured expected costs and Monte Carlo cross-checks.

- The cost-optimal placement saves **40.5%** vs restart-from-scratch and **51.2%** vs checkpointing every step, on average.
- The best *fixed* interval costs **1.7%** more than the optimal placement on average and **4.9%** more on spiky profiles -- on uniform profiles a fixed interval is nearly optimal, and the placement gain concentrates exactly where hazards are uneven.
- Estimated from a 300-run fault log instead of true hazards, the tool's placement costs only **0.13%** more than the oracle's in true expected cost; its fixed-interval pick is off by **0.1** steps on average.
- Analytic expected costs match seeded Monte Carlo means within **2.4%** on average.
- Recovery half-life at repair rate 0.25: a fault's error halves every **2.4 steps**; a severity-0.8 fault needs **9.6 steps** to fall below the 0.05 threshold, so faults near the end of a run cannot self-repair in time.
- Showcase (spiky-64): restart 3281.27, every-step 266.88, best fixed interval 5 at 142.61, optimal placement [5, 9, 19, 20, 33, 35, 38, 47, 49, 52, 54, 61, 64] at 129.33.

Caveat: all workflows are synthetic and seeded, with hazards known by construction; the experiment validates the cost model and the optimiser under that model, not any live agent system.
