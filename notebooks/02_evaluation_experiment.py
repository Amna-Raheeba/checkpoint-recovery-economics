"""Evaluation experiment (smoke configuration for CI).

Runs the headline evaluation on a reduced scenario set (lengths 8 and
32, fewer audit-log and Monte Carlo runs) and prints the aggregates.
The full evaluation that produced results/ is:

    python -m checkpoint_econ evaluate

Run: python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from checkpoint_econ.evaluate import SMOKE_CONFIG, run_evaluation


def main():
    evaluation = run_evaluation(SMOKE_CONFIG)
    agg = evaluation["aggregates"]
    print("Scenarios: %d" % agg["n_scenarios"])
    print("Mean saving vs restart: %.1f%%" % agg["mean_savings_vs_restart_pct"])
    print(
        "Mean saving vs every-step: %.1f%%"
        % agg["mean_savings_vs_every_step_pct"]
    )
    print(
        "Best fixed interval gap over optimal: %.1f%%"
        % agg["mean_fixed_vs_optimal_gap_pct"]
    )
    print("Mean tool regret vs oracle: %.2f%%" % agg["mean_tool_regret_pct"])
    print(
        "Mean interval error (steps): %.1f" % agg["mean_interval_abs_error"]
    )
    print(
        "Mean Monte Carlo deviation: %.1f%%" % agg["mean_mc_deviation_pct"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
