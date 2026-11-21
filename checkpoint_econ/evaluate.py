"""Headline evaluation: checkpoint economics across workflow shapes.

Scenarios cross workflow lengths {8, 16, 32, 64} with hazard profiles
{uniform_low, uniform_high, spiky, back_loaded}. Step costs are drawn
once per scenario from a seeded RNG; hazards are known exactly, which
makes the seeded Monte Carlo an honest check on the analytic model.

For every scenario the evaluation reports:

- analytic expected cost of restart, repair, never-checkpoint,
  every-step, the oracle fixed interval (grid search on true hazards),
  and the DP-optimal placement (true hazards);
- the *tool* numbers: hazards are estimated from a simulated first-pass
  fault log of ``audit_runs`` historical runs, the optimiser runs on
  the estimates, and the resulting placement / interval / recommendation
  are then priced on the true workflow -- the regret against the oracle
  is the price of not knowing the hazards;
- seeded Monte Carlo realised costs for restart, every-step, the
  optimal placement, and repair, with the deviation from the analytic
  expectation;
- a recovery table: analytic vs trajectory-measured half-life and
  recovery time for a planted fault at three repair rates.

Everything is deterministic: same seeds, same numbers.
"""

import csv
import json
import os
import random

from .models import Workflow
from .optimize import (
    best_fixed_interval,
    estimate_hazards,
    expected_cost_every_step,
    expected_cost_fixed_interval,
    expected_cost_never,
    expected_cost_placement,
    expected_cost_repair,
    expected_cost_restart,
    fixed_interval_checkpoints,
    optimal_placement,
    recommend,
)
from .simulate import generate_fault_log, measure_recovery, simulate_policy

PROFILES = ("uniform_low", "uniform_high", "spiky", "back_loaded")

DEFAULT_CONFIG = {
    "lengths": [8, 16, 32, 64],
    "profiles": list(PROFILES),
    "checkpoint_cost": 3.0,
    "penalty": 600.0,
    "repair_rate": 0.25,
    "persistence": 0.3,
    "error_threshold": 0.05,
    "severity_lo": 0.4,
    "severity_hi": 1.0,
    "audit_runs": 300,
    "mc_runs": 300,
    "base_seed": 20261121,
}

SMOKE_CONFIG = dict(
    DEFAULT_CONFIG, lengths=[8, 32], audit_runs=60, mc_runs=40
)


def hazards_for_profile(profile, n_steps, rng):
    """The hazard profile of a scenario (deterministic given the rng)."""
    if profile == "uniform_low":
        return [0.02] * n_steps
    if profile == "uniform_high":
        return [0.08] * n_steps
    if profile == "spiky":
        hazards = [0.01] * n_steps
        spikes = max(1, int(round(0.15 * n_steps)))
        for index in rng.sample(range(n_steps), spikes):
            hazards[index] = 0.35
        return hazards
    if profile == "back_loaded":
        if n_steps == 1:
            return [0.06]
        return [
            0.01 + 0.11 * (i / float(n_steps - 1)) for i in range(n_steps)
        ]
    raise ValueError("unknown profile: %r" % (profile,))


def build_workflow(profile, n_steps, cfg, seed):
    """One scenario workflow with seeded step costs."""
    rng = random.Random(seed)
    costs = [round(rng.uniform(0.6, 1.6), 4) for _ in range(n_steps)]
    hazards = hazards_for_profile(profile, n_steps, rng)
    return Workflow(
        name="%s-%d" % (profile, n_steps),
        step_costs=costs,
        step_hazards=hazards,
        checkpoint_cost=cfg["checkpoint_cost"],
        penalty=cfg["penalty"],
        repair_rate=cfg["repair_rate"],
        persistence=cfg["persistence"],
        error_threshold=cfg["error_threshold"],
        severity_lo=cfg["severity_lo"],
        severity_hi=cfg["severity_hi"],
    )


def _evaluate_on_true(workflow, strategy_row):
    """Price a strategy chosen on estimated hazards on the true workflow."""
    strategy = strategy_row["strategy"]
    if strategy == "restart":
        return expected_cost_restart(workflow)
    if strategy == "repair":
        return expected_cost_repair(workflow)
    if strategy == "never":
        return expected_cost_never(workflow)
    return expected_cost_placement(workflow, strategy_row["checkpoints"])


def _pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def evaluate_scenario(profile, n_steps, cfg, seed):
    workflow = build_workflow(profile, n_steps, cfg, seed)

    restart = expected_cost_restart(workflow)
    repair = expected_cost_repair(workflow)
    never = expected_cost_never(workflow)
    every = expected_cost_every_step(workflow)
    oracle_interval, oracle_fixed = best_fixed_interval(workflow)
    oracle_points, oracle_optimal = optimal_placement(workflow)

    # The tool sees only a fault log, never the true hazards.
    counts = generate_fault_log(workflow, cfg["audit_runs"], seed + 1)
    estimated = workflow.with_hazards(estimate_hazards(counts, cfg["audit_runs"]))
    tool_points, _tool_hat_cost = optimal_placement(estimated)
    tool_optimal = expected_cost_placement(workflow, tool_points)
    tool_interval, _tool_hat_fixed = best_fixed_interval(estimated)
    tool_fixed = expected_cost_fixed_interval(workflow, tool_interval)
    tool_best = recommend(estimated)["best"]
    tool_recommendation = _evaluate_on_true(workflow, tool_best)
    oracle_recommendation = recommend(workflow)["best"]["expected_cost"]

    mc_restart = simulate_policy(workflow, "restart", cfg["mc_runs"], seed + 2)
    mc_every = simulate_policy(
        workflow,
        "rollback",
        cfg["mc_runs"],
        seed + 3,
        checkpoints=fixed_interval_checkpoints(n_steps, 1),
    )
    mc_optimal = simulate_policy(
        workflow, "rollback", cfg["mc_runs"], seed + 4, checkpoints=oracle_points
    )
    mc_repair = simulate_policy(workflow, "repair", cfg["mc_runs"], seed + 5)

    def deviation(mc, analytic):
        return abs(mc["mean_cost"] - analytic) / analytic if analytic else 0.0

    return {
        "scenario": workflow.name,
        "profile": profile,
        "n_steps": n_steps,
        "restart": restart,
        "repair": repair,
        "never": never,
        "every_step": every,
        "oracle_fixed_interval": oracle_interval,
        "oracle_fixed": oracle_fixed,
        "oracle_optimal": oracle_optimal,
        "oracle_checkpoints": oracle_points,
        "tool_fixed_interval": tool_interval,
        "tool_fixed": tool_fixed,
        "tool_optimal": tool_optimal,
        "tool_checkpoints": tool_points,
        "tool_recommendation_strategy": tool_best["strategy"],
        "tool_recommendation": tool_recommendation,
        "tool_regret_pct": _pct(tool_optimal - oracle_optimal, oracle_optimal),
        "oracle_recommendation": oracle_recommendation,
        "recommendation_regret_pct": _pct(
            tool_recommendation - oracle_recommendation, oracle_recommendation
        ),
        "interval_abs_error": abs(tool_interval - oracle_interval),
        "savings_vs_restart_pct": _pct(restart - oracle_optimal, restart),
        "savings_vs_every_step_pct": _pct(every - oracle_optimal, every),
        "fixed_vs_optimal_gap_pct": _pct(oracle_fixed - oracle_optimal, oracle_fixed),
        "mc_restart": mc_restart["mean_cost"],
        "mc_every_step": mc_every["mean_cost"],
        "mc_optimal": mc_optimal["mean_cost"],
        "mc_repair": mc_repair["mean_cost"],
        "mc_deviation_restart": deviation(mc_restart, restart),
        "mc_deviation_every_step": deviation(mc_every, every),
        "mc_deviation_optimal": deviation(mc_optimal, oracle_optimal),
        "mc_deviation_repair": deviation(mc_repair, repair),
    }


def recovery_table(cfg):
    """Analytic vs trajectory-measured fault recovery at three rates."""
    rows = []
    for rate in (0.10, 0.25, 0.50):
        measured = measure_recovery(0.8, rate, cfg["error_threshold"])
        rows.append(
            {
                "repair_rate": rate,
                "severity": 0.8,
                "analytic_half_life": measured["analytic_half_life"],
                "measured_half_life_steps": measured["half_life_steps"],
                "analytic_recovery_time": measured["analytic_recovery_time"],
                "measured_recovery_steps": measured["recovery_steps"],
            }
        )
    return rows


def run_evaluation(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    scenarios = []
    index = 0
    for length in cfg["lengths"]:
        for profile in cfg["profiles"]:
            seed = cfg["base_seed"] + 1000 * index
            scenarios.append(evaluate_scenario(profile, length, cfg, seed))
            index += 1

    def mean(key):
        return sum(row[key] for row in scenarios) / len(scenarios)

    spiky_rows = [row for row in scenarios if row["profile"] == "spiky"]
    spiky_gap = (
        sum(row["fixed_vs_optimal_gap_pct"] for row in spiky_rows)
        / len(spiky_rows)
        if spiky_rows
        else 0.0
    )

    aggregates = {
        "n_scenarios": len(scenarios),
        "mean_savings_vs_restart_pct": mean("savings_vs_restart_pct"),
        "mean_savings_vs_every_step_pct": mean("savings_vs_every_step_pct"),
        "mean_fixed_vs_optimal_gap_pct": mean("fixed_vs_optimal_gap_pct"),
        "fixed_gap_spiky_pct": spiky_gap,
        "mean_tool_regret_pct": mean("tool_regret_pct"),
        "mean_recommendation_regret_pct": mean("recommendation_regret_pct"),
        "mean_interval_abs_error": mean("interval_abs_error"),
        "mean_mc_deviation_pct": 100.0
        * (
            mean("mc_deviation_restart")
            + mean("mc_deviation_every_step")
            + mean("mc_deviation_optimal")
            + mean("mc_deviation_repair")
        )
        / 4.0,
    }
    showcase = None
    for row in scenarios:
        if row["profile"] == "spiky":
            if showcase is None or row["n_steps"] > showcase["n_steps"]:
                showcase = row
    if showcase is None and scenarios:
        showcase = scenarios[-1]
    aggregates["showcase"] = {
        "scenario": showcase["scenario"],
        "restart": showcase["restart"],
        "every_step": showcase["every_step"],
        "oracle_fixed": showcase["oracle_fixed"],
        "oracle_fixed_interval": showcase["oracle_fixed_interval"],
        "oracle_optimal": showcase["oracle_optimal"],
        "oracle_checkpoints": showcase["oracle_checkpoints"],
        "tool_optimal": showcase["tool_optimal"],
        "tool_regret_pct": showcase["tool_regret_pct"],
    }
    return {
        "config": {k: cfg[k] for k in ("checkpoint_cost", "penalty",
                                        "repair_rate", "persistence",
                                        "error_threshold",
                                        "audit_runs", "mc_runs", "base_seed")},
        "scenarios": scenarios,
        "recovery": recovery_table(cfg),
        "aggregates": aggregates,
    }


_CSV_FIELDS = [
    "scenario", "profile", "n_steps", "restart", "repair", "never",
    "every_step", "oracle_fixed_interval", "oracle_fixed", "oracle_optimal",
    "tool_fixed_interval", "tool_fixed", "tool_optimal", "tool_recommendation",
    "tool_regret_pct", "interval_abs_error", "mc_restart", "mc_every_step",
    "mc_optimal", "mc_repair",
]


def write_results(evaluation, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "evaluation.json"), "w") as handle:
        json.dump(evaluation, handle, indent=2)
    with open(os.path.join(outdir, "evaluation.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in evaluation["scenarios"]:
            writer.writerow({key: row[key] for key in _CSV_FIELDS})
    with open(os.path.join(outdir, "evaluation_summary.md"), "w") as handle:
        handle.write(summary_markdown(evaluation))
    with open(os.path.join(outdir, "HEADLINE.md"), "w") as handle:
        handle.write(headline_markdown(evaluation))


def summary_markdown(evaluation):
    agg = evaluation["aggregates"]
    cfg = evaluation["config"]
    lines = []
    lines.append("# Evaluation summary")
    lines.append("")
    lines.append(
        "Seeded evaluation: %d workflow scenarios (lengths 8-64, hazard "
        "profiles uniform_low / uniform_high / spiky / back_loaded), "
        "checkpoint cost %.1f, failure penalty %.1f, repair rate %.2f, "
        "fault persistence %.2f, error threshold %.2f. The tool "
        "estimates hazards from %d logged runs; Monte Carlo uses %d "
        "runs per policy per scenario."
        % (
            agg["n_scenarios"],
            cfg["checkpoint_cost"],
            cfg["penalty"],
            cfg["repair_rate"],
            cfg["persistence"],
            cfg["error_threshold"],
            cfg["audit_runs"],
            cfg["mc_runs"],
        )
    )
    lines.append("")
    lines.append("## Aggregates")
    lines.append("")
    lines.append(
        "- Mean saving of the optimal placement vs restart-from-scratch: "
        "%.1f%%" % agg["mean_savings_vs_restart_pct"]
    )
    lines.append(
        "- Mean saving of the optimal placement vs checkpointing every "
        "step: %.1f%%" % agg["mean_savings_vs_every_step_pct"]
    )
    lines.append(
        "- Best fixed interval costs %.1f%% more than the optimal "
        "placement on average, and %.1f%% more on spiky hazard profiles, "
        "where uniform intervals cannot follow the risk."
        % (agg["mean_fixed_vs_optimal_gap_pct"], agg["fixed_gap_spiky_pct"])
    )
    lines.append(
        "- Tool regret (placement optimised on estimated hazards, priced "
        "on true hazards): %.2f%% above the oracle placement on average."
        % agg["mean_tool_regret_pct"]
    )
    lines.append(
        "- Tool fixed-interval choice differs from the oracle interval by "
        "%.1f steps on average." % agg["mean_interval_abs_error"]
    )
    lines.append(
        "- Analytic expected costs vs seeded Monte Carlo means: %.1f%% "
        "mean absolute deviation." % agg["mean_mc_deviation_pct"]
    )
    lines.append("")
    lines.append("## Scenarios")
    lines.append("")
    lines.append(
        "| scenario | restart | every step | oracle fixed (m) | oracle "
        "optimal | tool optimal | tool regret % |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in evaluation["scenarios"]:
        lines.append(
            "| %s | %.2f | %.2f | %.2f (%d) | %.2f | %.2f | %.2f |"
            % (
                row["scenario"],
                row["restart"],
                row["every_step"],
                row["oracle_fixed"],
                row["oracle_fixed_interval"],
                row["oracle_optimal"],
                row["tool_optimal"],
                row["tool_regret_pct"],
            )
        )
    lines.append("")
    lines.append("## Fault recovery (severity 0.8, threshold %.2f)" % cfg["error_threshold"])
    lines.append("")
    lines.append(
        "| repair rate | analytic half-life | measured half-life (steps) "
        "| analytic recovery | measured recovery (steps) |"
    )
    lines.append("|---|---|---|---|---|")
    for row in evaluation["recovery"]:
        lines.append(
            "| %.2f | %.2f | %s | %.2f | %s |"
            % (
                row["repair_rate"],
                row["analytic_half_life"],
                row["measured_half_life_steps"],
                row["analytic_recovery_time"],
                row["measured_recovery_steps"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def headline_markdown(evaluation):
    agg = evaluation["aggregates"]
    show = agg["showcase"]
    mid = None
    for row in evaluation["recovery"]:
        if abs(row["repair_rate"] - 0.25) < 1e-9:
            mid = row
    lines = []
    lines.append("# Headline results")
    lines.append("")
    lines.append(
        "Checkpoint placement priced as an economic decision on %d seeded "
        "workflow scenarios (8-64 steps, four hazard profiles), with real "
        "measured expected costs and Monte Carlo cross-checks."
        % agg["n_scenarios"]
    )
    lines.append("")
    lines.append(
        "- The cost-optimal placement saves **%.1f%%** vs "
        "restart-from-scratch and **%.1f%%** vs checkpointing every step, "
        "on average." % (agg["mean_savings_vs_restart_pct"],
                         agg["mean_savings_vs_every_step_pct"])
    )
    lines.append(
        "- The best *fixed* interval costs **%.1f%%** more than the "
        "optimal placement on average and **%.1f%%** more on spiky "
        "profiles -- on uniform profiles a fixed interval is nearly "
        "optimal, and the placement gain concentrates exactly where "
        "hazards are uneven."
        % (agg["mean_fixed_vs_optimal_gap_pct"], agg["fixed_gap_spiky_pct"])
    )
    lines.append(
        "- Estimated from a %d-run fault log instead of true hazards, the "
        "tool's placement costs only **%.2f%%** more than the oracle's in "
        "true expected cost; its fixed-interval pick is off by **%.1f** "
        "steps on average." % (evaluation["config"]["audit_runs"],
                               agg["mean_tool_regret_pct"],
                               agg["mean_interval_abs_error"])
    )
    lines.append(
        "- Analytic expected costs match seeded Monte Carlo means within "
        "**%.1f%%** on average." % agg["mean_mc_deviation_pct"]
    )
    if mid is not None:
        lines.append(
            "- Recovery half-life at repair rate 0.25: a fault's error "
            "halves every **%.1f steps**; a severity-0.8 fault needs "
            "**%.1f steps** to fall below the 0.05 threshold, so faults "
            "near the end of a run cannot self-repair in time."
            % (mid["analytic_half_life"], mid["analytic_recovery_time"])
        )
    lines.append(
        "- Showcase (%s): restart %.2f, every-step %.2f, best fixed "
        "interval %d at %.2f, optimal placement %s at %.2f."
        % (
            show["scenario"],
            show["restart"],
            show["every_step"],
            show["oracle_fixed_interval"],
            show["oracle_fixed"],
            show["oracle_checkpoints"],
            show["oracle_optimal"],
        )
    )
    lines.append("")
    lines.append(
        "Caveat: all workflows are synthetic and seeded, with hazards "
        "known by construction; the experiment validates the cost model "
        "and the optimiser under that model, not any live agent system."
    )
    lines.append("")
    return "\n".join(lines)


def check_drift(stored, fresh, tol=1e-6):
    """Compare aggregate numbers of two evaluations. Returns (ok, problems)."""
    problems = []
    stored_agg = stored["aggregates"]
    fresh_agg = fresh["aggregates"]
    for key in stored_agg:
        if key == "showcase":
            continue
        stored_value = stored_agg[key]
        fresh_value = fresh_agg.get(key)
        if fresh_value is None:
            problems.append((key, stored_value, None, None))
            continue
        diff = abs(stored_value - fresh_value)
        scale = max(1.0, abs(stored_value))
        if diff / scale > tol:
            problems.append((key, stored_value, fresh_value, diff))
    return (not problems), problems
