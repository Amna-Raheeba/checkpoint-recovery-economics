"""checkpoint_econ: checkpoint-recovery economics for agent workflows.

Places checkpoints by minimising expected total cost (checkpoint spend
plus expected recompute after failure), measures fault recovery
half-lives under continue-and-self-repair, and audits recorded traces
for recovery waste.

Author: Amna Raheeba
"""

from .audit import AuditResult, audit_trace
from .models import Trace, Workflow
from .optimize import (
    best_fixed_interval,
    estimate_hazards,
    expected_cost_placement,
    optimal_placement,
    recommend,
)
from .simulate import analytic_half_life, measure_recovery, simulate_policy

__version__ = "0.1.0"

__all__ = [
    "AuditResult",
    "Trace",
    "Workflow",
    "analytic_half_life",
    "audit_trace",
    "best_fixed_interval",
    "estimate_hazards",
    "expected_cost_placement",
    "measure_recovery",
    "optimal_placement",
    "recommend",
    "simulate_policy",
]
