# Checkpoint Recovery Economics

Checkpoint placement for long agent runs, priced as an **economic
decision**: an optimiser that places checkpoints by minimising expected
total cost — checkpoint spend plus expected recompute after failure —
plus a recovery-dynamics model and an auditor that prices the recovery
waste in recorded traces.

## 1. What it is

Long agent runs fail in the middle. The standard responses are folklore:
checkpoint every step (and pay for state you almost never roll back to),
checkpoint every N steps (a number nobody derived), or don't checkpoint
and restart from scratch (and re-pay the whole run when step 57 of 64
goes wrong). None of these is priced.

This tool prices them. Given a workflow — per-step costs, per-step fault
hazards, a checkpoint cost, a failure penalty — it returns:

1. the **cost-optimal checkpoint placement** (dynamic programming over
   the analytic expected-cost model),
2. the expected cost of every standard strategy next to it:
   restart-from-scratch, continue-and-self-repair, never checkpoint,
   every step, and the best fixed interval (grid search),
3. a **recovery half-life** for faults under self-repair — how many
   steps a fault's error needs to halve, and to fall below threshold —
   which is what prices the un-checkpointed tail of a run,
4. an **audit** of a recorded trace: which checkpoints were never used,
   how much rework the faults actually forced, and what the same faults
   would have cost under the optimal placement.

Hazards do not have to be known: the tool estimates them from a log of
past runs (per-step first-pass fault counts) and optimises on the
estimates. Everything is offline, deterministic and seeded. Pure Python
standard library; no API keys, no model downloads, no network.

## 2. The novel mechanism

Three pieces combine into one placement decision:

**Expected-cost model.** A segment closed by a checkpoint is validated:
a fault inside it is detected at the checkpoint and the segment is
re-executed until clean, so its expected cost is
`segment_cost / (1 - q)` for segment fault probability `q`, plus the
checkpoint cost. A segment left unvalidated is executed once, and a
fault inside it costs the failure penalty only if it is still above the
error threshold at the end of the run.

**Recovery dynamics.** A fault's error decays geometrically while the
run continues — each step repairs a fraction `repair_rate` of it — so a
fault at step k contributes `severity * (1 - r)^(N - k)` at the end.
Its **half-life** is `ln(0.5) / ln(1 - r)` steps. But a fraction
`persistence` of faults corrupt a downstream dependency and never decay
at all. It is the persistent fraction that makes long un-checkpointed
tails expensive, and the half-life that makes short ones cheap.

**Placement optimiser.** Dynamic programming over segments finds the
placement minimising expected total cost, including the option of
leaving the tail unvalidated when self-repair genuinely covers it. The
same machinery prices fixed intervals, so "every 5 steps" is compared
against the optimum it approximates — and a hazard estimator
(Laplace-smoothed, from a first-pass fault log) lets the whole pipeline
run on a team's own recorded runs instead of declared hazards.

## 3. Prior art and the differentiator

- **AgenticRAG-FP** injects faults into agentic RAG pipelines to test
  whether post-hoc *diagnosers* can localise the faulty hop.
- **AgentDebug / AgentErrorBench** classify and repair failures in real
  rollouts after the fact.
- The portfolio's own **cascadescope** measures the forward blast
  radius of a fault — how much damage it does downstream.

All of these look backwards: given a failure, explain or repair it.
None measures recovery dynamics (how fast a fault's error decays, what
a checkpoint buys in expectation) or optimises checkpoint placement.
The differentiator is forward-looking reliability engineering: *where
to checkpoint, and which recovery policy pays* — decided before the
run, from costs and hazards, with the answer checked against seeded
Monte Carlo.

## 4. Measured headline results

Seeded evaluation, 16 scenarios (workflow lengths 8–64 × hazard
profiles uniform-low, uniform-high, spiky, back-loaded); checkpoint
cost 3.0, failure penalty 600, repair rate 0.25, persistence 0.3. The
tool sees only a 300-run fault log per scenario; realised costs come
from 300 Monte Carlo runs per policy per scenario. Full detail in
`results/`.

- The cost-optimal placement saves **40.5%** vs restart-from-scratch
  and **51.2%** vs checkpointing every step, on average.
- The best fixed interval costs **1.7%** more than the optimal
  placement on average and **4.9%** more on spiky profiles — on uniform
  profiles a fixed interval is nearly optimal, and the placement gain
  concentrates exactly where hazards are uneven.
- Priced on true hazards, the placement the tool derives from the
  fault log costs only **0.13%** more than the oracle placement that
  knows the hazards; its fixed-interval pick is off by **0.06 steps**
  on average.
- Analytic expected costs match seeded Monte Carlo means within
  **2.4%** on average.
- Recovery half-life at repair rate 0.25: a fault's error halves every
  **2.41 steps**; a severity-0.8 fault needs **9.64 steps** to fall
  below the 0.05 threshold — faults in the last stretch of a run cannot
  self-repair in time.
- **Showcase** (spiky, 64 steps): restart 3281.27, every-step 266.88,
  best fixed interval 5 at 142.61, optimal placement — 13 checkpoints
  clustered around the dangerous steps — at **129.33**.

One honest counterpoint the evaluation surfaces: on a short, safe
workflow (8 steps, hazard 0.02) restart-from-scratch at 9.11 beats
every checkpointing strategy (best placement 12.11). Checkpointing is
insurance; on short cheap runs the premium is not worth paying, and
the recommender says so.

## 5. Quick start

```bash
pip install -r requirements.txt

python -m checkpoint_econ demo                              # built-in demo
python -m checkpoint_econ optimize --workflow examples/demo_workflow.json
python -m checkpoint_econ audit --trace examples/demo_trace.json
python -m checkpoint_econ evaluate                          # regenerate results/
python examples/run_audit.py                                # audit via the example script
streamlit run app.py                                        # dashboard (3 tabs)
```

Workflow format: JSON with `name`, `step_costs`, `step_hazards`, and
optional `checkpoint_cost`, `penalty`, `repair_rate`, `persistence`,
`error_threshold`, `severity_lo`, `severity_hi` (see
`examples/demo_workflow.json`). Trace format: a workflow plus
`checkpoints_taken`, `faults` (each with `step` and `severity`), and
the `recovery` policy the run used (see `examples/demo_trace.json`).

## 6. How it works

1. **Model** — declare or estimate the workflow: step costs, step
   hazards, checkpoint cost, penalty, repair rate, persistence.
2. **Price** — compute the analytic expected cost of every strategy:
   validated segments cost `segment_cost / (1 - q)` plus the
   checkpoint; unvalidated tails cost the penalty times the
   unrecovered-fault probability from the recovery dynamics.
3. **Optimise** — dynamic programming over segment boundaries finds
   the placement (possibly with an unvalidated tail) that minimises
   expected total cost; grid search prices the best fixed interval.
4. **Recommend** — the cheapest strategy overall wins, which is
   sometimes restart (short safe runs) and sometimes repair (when the
   persistent-fault risk is low) — the table shows every price.
5. **Audit** — replay a recorded trace deterministically: unused
   checkpoints, actual rework, and the counterfactual cost of the
   same faults under the optimal placement.

## 7. Repository layout

```
checkpoint_econ/     the package (models, optimize, simulate, audit,
                     evaluate, report, cli)
data/                the 16 evaluation scenario workflows
tests/               pytest suite (83 tests)
notebooks/           01_demo_run.py, 02_evaluation_experiment.py
examples/            demo workflow, demo trace, runner, sample report
scripts/             build_cases.py, reproduce_results.py (drift guard)
results/             evaluation.json/.csv, evaluation_summary.md, HEADLINE.md
app.py               Streamlit dashboard (Optimise / Recovery / Evaluation)
```

## 8. Honest limits

- **Synthetic workflows only.** Every number above is measured on
  seeded synthetic workflows with hazards known by construction. The
  experiment validates the cost model, the optimiser, and the hazard
  estimator under that model; it does not measure any live agent
  system, whose hazards must be estimated from real logs and will be
  noisier.
- **Declared economics.** The penalty (600), checkpoint cost (3.0),
  persistence (0.3), and repair rate (0.25) are declared assumptions,
  stated in the config and the README. Different ratios move the
  optimum — that is the point of the tool — but no number here
  transfers to a real deployment without re-declaring them.
- **Simple fault model.** One fault kind per step, severities uniform
  on [0.4, 1.0], error contributions combining by maximum, retries
  that eventually run clean. Correlated faults (one bad tool failing
  for an hour) and partial checkpoints are not modelled.
- **Audit replay is a convention.** The trace replay assumes each
  faulted segment was re-executed once and the retry was clean, and
  recorded faults are replayed as the decaying kind — whether a
  specific fault was persistent is not something a trace records.
  "Unused checkpoint" means unused *on that run*; insurance is priced
  on expectation, and the audit says so in its recommendation.
- **Hazard estimates are per-step and independent.** The estimator
  prices each step from its own fault count; steps never observed
  faulting get a small smoothed hazard, not zero.

## 9. Reproducing the results

```bash
python scripts/build_cases.py        # regenerate data/ and the demo files
python -m checkpoint_econ evaluate   # regenerate results/
python scripts/reproduce_results.py  # re-run and check for headline drift
pytest                               # full test suite
```

## 10. License

MIT. Author: Amna Raheeba.
