"""Demo: price the checkpoint strategies for one workflow, audit a trace.

The demo workflow is a 24-step research run with two dangerous steps
(7 and 16, hazard 0.30 against a 0.02 background). A uniform checkpoint
interval cannot see those spikes; the optimiser can.

Run: python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from checkpoint_econ.audit import audit_trace
from checkpoint_econ.cli import DEMO_TRACE, DEMO_WORKFLOW
from checkpoint_econ.optimize import recommend
from checkpoint_econ.report import audit_report_markdown, optimize_report_markdown
from checkpoint_econ.simulate import measure_recovery


def main():
    print(optimize_report_markdown(DEMO_WORKFLOW, recommend(DEMO_WORKFLOW)))
    print(audit_report_markdown(DEMO_TRACE, audit_trace(DEMO_TRACE)))
    recovery = measure_recovery(0.8, DEMO_WORKFLOW.repair_rate, 0.05)
    print(
        "Fault recovery at repair rate %.2f: analytic half-life %.2f "
        "steps (measured %s), recovery to 0.05 in %.2f steps (measured "
        "%s)."
        % (
            DEMO_WORKFLOW.repair_rate,
            recovery["analytic_half_life"],
            recovery["half_life_steps"],
            recovery["analytic_recovery_time"],
            recovery["recovery_steps"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
