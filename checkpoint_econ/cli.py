"""Command line interface.

Usage:
    python -m checkpoint_econ demo
    python -m checkpoint_econ optimize --workflow examples/demo_workflow.json
    python -m checkpoint_econ audit --trace examples/demo_trace.json
    python -m checkpoint_econ evaluate --outdir results
"""

import argparse
import json

from .audit import audit_trace
from .evaluate import run_evaluation, write_results
from .models import Trace, Workflow
from .optimize import recommend
from .report import audit_report_markdown, optimize_report_markdown
from .simulate import measure_recovery

DEMO_WORKFLOW = Workflow(
    name="demo-research-run",
    step_costs=[1.0] * 24,
    step_hazards=[0.02] * 6 + [0.30] + [0.02] * 8 + [0.30] + [0.02] * 8,
    checkpoint_cost=3.0,
    penalty=600.0,
    repair_rate=0.25,
)

DEMO_TRACE = Trace(
    workflow=DEMO_WORKFLOW,
    checkpoints_taken=[6, 12, 18, 24],
    faults=[
        {"step": 7, "severity": 0.9},
        {"step": 16, "severity": 0.7},
        {"step": 20, "severity": 0.5},
    ],
    recovery="rollback",
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="checkpoint_econ",
        description=(
            "Price checkpoint placement for long agent runs: minimise "
            "checkpoint cost plus expected recompute after failure, and "
            "audit recorded traces for recovery waste."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Optimise and audit a built-in demo workflow")

    opt_parser = sub.add_parser(
        "optimize", help="Compute the cost-optimal checkpoint placement"
    )
    opt_parser.add_argument("--workflow", required=True, help="Workflow JSON path")

    audit_parser = sub.add_parser(
        "audit", help="Audit a recorded run trace for recovery waste"
    )
    audit_parser.add_argument("--trace", required=True, help="Trace JSON path")

    eval_parser = sub.add_parser(
        "evaluate", help="Run the headline evaluation and write results"
    )
    eval_parser.add_argument("--outdir", default="results")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        print(optimize_report_markdown(DEMO_WORKFLOW, recommend(DEMO_WORKFLOW)))
        print(audit_report_markdown(DEMO_TRACE, audit_trace(DEMO_TRACE)))
        recovery = measure_recovery(0.8, DEMO_WORKFLOW.repair_rate, 0.05)
        print(
            "Fault recovery at repair rate %.2f: half-life %.2f steps, a "
            "severity-0.8 fault needs %.2f steps to fall below 0.05."
            % (
                DEMO_WORKFLOW.repair_rate,
                recovery["analytic_half_life"],
                recovery["analytic_recovery_time"],
            )
        )
        return 0
    if args.command == "optimize":
        with open(args.workflow) as handle:
            workflow = Workflow.from_dict(json.load(handle))
        print(optimize_report_markdown(workflow, recommend(workflow)))
        return 0
    if args.command == "audit":
        with open(args.trace) as handle:
            trace = Trace.from_dict(json.load(handle))
        print(audit_report_markdown(trace, audit_trace(trace)))
        return 0
    if args.command == "evaluate":
        evaluation = run_evaluation()
        write_results(evaluation, args.outdir)
        agg = evaluation["aggregates"]
        print(
            "Wrote results to %s (optimal placement saves %.1f%% vs "
            "restart, %.1f%% vs every-step; tool regret %.2f%%)"
            % (
                args.outdir,
                agg["mean_savings_vs_restart_pct"],
                agg["mean_savings_vs_every_step_pct"],
                agg["mean_tool_regret_pct"],
            )
        )
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
