# Changelog

## 0.1.0 — 2026-11-21

- Initial release.
- Analytic cost model for agent workflows: validated segments (rollback
  to checkpoint, re-execute until clean), unvalidated tails priced by
  fault-recovery dynamics (geometric error decay plus a persistent-fault
  fraction that never self-repairs), restart-from-scratch and
  continue-and-repair baselines.
- Dynamic-programming optimiser for checkpoint placement, exhaustive
  grid search for the best fixed interval, and a strategy recommender.
- Hazard estimation from first-pass fault logs (Laplace-smoothed), so
  the optimiser can run on a team's own recorded runs.
- Seeded Monte Carlo simulator for the restart / rollback / repair
  policies, used to cross-check the analytic expected costs.
- Fault recovery measurement: error half-life and time-to-threshold of
  a planted fault under the repair dynamics.
- Trace auditor: deterministic replay of a recorded run that prices
  unused checkpoints, excess rework, and the gap to the optimal
  placement on the same faults.
- Headline evaluation: 16 scenarios (workflow lengths 8-64 x four
  hazard profiles) with oracle vs tool regret, fixed-interval gap, and
  Monte Carlo deviation.
- Streamlit dashboard (3 tabs), CLI, notebooks, and a reproducibility
  script with a headline-drift guard.
