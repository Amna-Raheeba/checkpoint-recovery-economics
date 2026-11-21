"""Trace audit: replay one recorded run and price its recovery waste.

The audit takes a Trace (the workflow, the checkpoints actually taken,
the faults that actually occurred on the first pass, and the recovery
policy the run used) and decomposes what the run spent:

- base cost -- one clean pass over the workflow;
- checkpoint spend -- checkpoints taken x checkpoint cost;
- rework -- what the recovery policy re-executed because of the faults;
- penalty -- paid when, under continue-and-repair, the residual error
  at the end still exceeds the threshold.

Replay rules (deterministic, stated so the numbers can be checked by
hand): under rollback, every segment containing at least one fault is
re-executed exactly once and the retry is clean; under restart, any
fault costs one full extra pass; under repair there is no rework, only
the possible penalty, and recorded faults are replayed as the decaying
kind (whether a specific fault was persistent is a population
parameter, not something a trace records). A checkpoint is *used* when
the segment it closes
contains a fault -- it bounded that fault's rework. A checkpoint whose
segment ran clean was insurance that did not pay out on this run.

The counterfactual replays the same faults under the cost-optimal
checkpoint placement from ``optimize.optimal_placement``. The waste of
the recorded run is its actual total minus the counterfactual total,
decomposed into checkpoint-spend difference and rework difference.
"""

from dataclasses import dataclass, field

from .optimize import optimal_placement, segment_cost
from .simulate import _residual_error


@dataclass
class AuditResult:
    base_cost: float
    checkpoint_spend: float
    rework: float
    penalty_paid: bool
    penalty_amount: float
    total_actual: float
    checkpoints_taken: list
    used_checkpoints: list
    unused_checkpoints: list
    wasted_checkpoint_spend: float
    optimal_checkpoints: list = field(default_factory=list)
    counterfactual_spend: float = 0.0
    counterfactual_rework: float = 0.0
    total_counterfactual: float = 0.0
    excess_cost: float = 0.0
    spend_difference: float = 0.0
    rework_difference: float = 0.0
    recommendation: str = ""


def _segments(n_steps, checkpoints):
    """(start, end, closing_checkpoint_or_None) per segment."""
    points = sorted(set(checkpoints))
    segments = []
    start = 0
    for position in points:
        segments.append((start, position, position))
        start = position
    if start < n_steps:
        segments.append((start, n_steps, None))
    return segments


def _replay(workflow, checkpoints, faults, recovery):
    """Deterministic replay. Returns (spend, rework, penalty, used)."""
    n = workflow.n_steps
    spend = workflow.checkpoint_cost * len(set(checkpoints))
    fault_steps = [fault["step"] for fault in faults]
    if recovery == "restart":
        rework = workflow.total_cost if fault_steps else 0.0
        return spend, rework, 0.0, []
    if recovery == "repair":
        pairs = [(fault["step"], fault["severity"]) for fault in faults]
        residual = _residual_error(workflow, pairs, n)
        penalty = workflow.penalty if residual > workflow.error_threshold else 0.0
        return spend, 0.0, penalty, []
    # rollback
    rework = 0.0
    used = []
    for start, end, closing in _segments(n, checkpoints):
        if any(start < step <= end for step in fault_steps):
            rework += segment_cost(workflow, start, end)
            if closing is not None:
                used.append(closing)
    return spend, rework, 0.0, used


def audit_trace(trace):
    """Audit one recorded run and return an AuditResult."""
    workflow = trace.workflow
    spend, rework, penalty, used = _replay(
        workflow, trace.checkpoints_taken, trace.faults, trace.recovery
    )
    base = workflow.total_cost
    total_actual = base + spend + rework + penalty
    taken = sorted(set(trace.checkpoints_taken))
    unused = [position for position in taken if position not in used]
    wasted = workflow.checkpoint_cost * len(unused)

    optimal_points, _expected = optimal_placement(workflow)
    cf_spend, cf_rework, cf_penalty, _cf_used = _replay(
        workflow, optimal_points, trace.faults, "rollback"
    )
    total_cf = base + cf_spend + cf_rework + cf_penalty

    recommendation = _recommend(
        trace, unused, rework, cf_rework, total_actual, total_cf, optimal_points
    )
    return AuditResult(
        base_cost=base,
        checkpoint_spend=spend,
        rework=rework,
        penalty_paid=penalty > 0,
        penalty_amount=penalty,
        total_actual=total_actual,
        checkpoints_taken=taken,
        used_checkpoints=used,
        unused_checkpoints=unused,
        wasted_checkpoint_spend=wasted,
        optimal_checkpoints=optimal_points,
        counterfactual_spend=cf_spend,
        counterfactual_rework=cf_rework,
        total_counterfactual=total_cf,
        excess_cost=total_actual - total_cf,
        spend_difference=spend - cf_spend,
        rework_difference=rework - cf_rework,
        recommendation=recommendation,
    )


def _recommend(trace, unused, rework, cf_rework, total_actual, total_cf, optimal):
    parts = []
    if unused:
        parts.append(
            "%d of %d checkpoints taken were never used by a fault on this "
            "run (spend %.2f bought no bounded rework this time)."
            % (len(unused), len(trace.checkpoints_taken),
               trace.workflow.checkpoint_cost * len(unused))
        )
    if rework > cf_rework + 1e-9:
        parts.append(
            "Faults forced %.2f of rework; under the cost-optimal "
            "placement %s the same faults would have cost %.2f."
            % (rework, optimal, cf_rework)
        )
    if total_actual <= total_cf + 1e-9:
        parts.append(
            "On this fault pattern the recorded run cost no more than the "
            "optimal-placement counterfactual (%.2f vs %.2f) -- checkpoint "
            "insurance is priced on expectation, not on one lucky or "
            "unlucky run." % (total_actual, total_cf)
        )
    else:
        parts.append(
            "Total excess over the counterfactual: %.2f. Re-place "
            "checkpoints at %s to minimise expected cost on this workflow "
            "shape." % (total_actual - total_cf, optimal)
        )
    return " ".join(parts)
