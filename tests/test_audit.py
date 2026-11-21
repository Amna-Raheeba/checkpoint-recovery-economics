import pytest

from checkpoint_econ.audit import audit_trace
from checkpoint_econ.cli import DEMO_TRACE
from checkpoint_econ.models import Trace, Workflow


def workflow(costs, hazards, **overrides):
    kwargs = {"name": "t", "step_costs": costs, "step_hazards": hazards}
    kwargs.update(overrides)
    return Workflow(**kwargs)


def test_demo_trace_audit_numbers():
    # Pinned to the built-in demo: 24 unit steps, checkpoints every 6,
    # faults at 7, 16, 20 under rollback.
    result = audit_trace(DEMO_TRACE)
    assert result.base_cost == 24.0
    assert result.checkpoint_spend == 12.0
    assert result.rework == 18.0
    assert result.total_actual == 54.0
    assert result.used_checkpoints == [12, 18, 24]
    assert result.unused_checkpoints == [6]
    assert result.wasted_checkpoint_spend == 3.0
    assert result.optimal_checkpoints == [6, 7, 15, 16, 24]
    assert result.total_counterfactual == 49.0
    assert result.excess_cost == 5.0
    assert result.spend_difference == -3.0
    assert result.rework_difference == 8.0


def test_audit_no_faults_all_checkpoints_unused():
    wf = workflow([1.0] * 6, [0.1] * 6, checkpoint_cost=2.0)
    trace = Trace(workflow=wf, checkpoints_taken=[3, 6], faults=[],
                  recovery="rollback")
    result = audit_trace(trace)
    assert result.rework == 0.0
    assert result.used_checkpoints == []
    assert result.unused_checkpoints == [3, 6]
    assert result.wasted_checkpoint_spend == 4.0
    assert result.total_actual == 6.0 + 4.0


def test_audit_rollback_rework_per_faulted_segment():
    wf = workflow([2.0, 2.0, 2.0, 2.0], [0.1] * 4, checkpoint_cost=1.0)
    trace = Trace(
        workflow=wf,
        checkpoints_taken=[2, 4],
        faults=[{"step": 1, "severity": 0.5}, {"step": 2, "severity": 0.5},
                {"step": 4, "severity": 0.5}],
        recovery="rollback",
    )
    result = audit_trace(trace)
    # Two faults in segment (0,2] still cost one segment re-execution.
    assert result.rework == 4.0 + 4.0
    assert result.used_checkpoints == [2, 4]


def test_audit_fault_in_unvalidated_tail():
    wf = workflow([1.0, 1.0, 1.0], [0.1] * 3, checkpoint_cost=1.0)
    trace = Trace(
        workflow=wf,
        checkpoints_taken=[1],
        faults=[{"step": 3, "severity": 0.5}],
        recovery="rollback",
    )
    result = audit_trace(trace)
    # The tail segment (1,3] is re-executed under the replay rules, but
    # no checkpoint closes it, so no checkpoint is marked used.
    assert result.rework == 2.0
    assert result.used_checkpoints == []
    assert result.unused_checkpoints == [1]


def test_audit_restart_rework_is_a_full_pass():
    wf = workflow([1.0, 2.0, 3.0], [0.1] * 3)
    trace = Trace(
        workflow=wf,
        checkpoints_taken=[],
        faults=[{"step": 2, "severity": 0.5}],
        recovery="restart",
    )
    result = audit_trace(trace)
    assert result.rework == 6.0
    assert result.total_actual == 12.0


def test_audit_restart_no_faults_no_rework():
    wf = workflow([1.0, 2.0], [0.1] * 2)
    trace = Trace(workflow=wf, faults=[], recovery="restart")
    assert audit_trace(trace).rework == 0.0


def test_audit_repair_late_fault_pays_penalty():
    wf = workflow([1.0] * 4, [0.1] * 4, penalty=600.0)
    trace = Trace(
        workflow=wf,
        checkpoints_taken=[],
        faults=[{"step": 4, "severity": 0.9}],
        recovery="repair",
    )
    result = audit_trace(trace)
    assert result.penalty_paid is True
    assert result.penalty_amount == 600.0
    assert result.total_actual == 4.0 + 600.0


def test_audit_repair_early_fault_decays_away():
    wf = workflow([1.0] * 24, [0.1] * 24, penalty=600.0)
    trace = Trace(
        workflow=wf,
        checkpoints_taken=[],
        faults=[{"step": 1, "severity": 0.5}],
        recovery="repair",
    )
    result = audit_trace(trace)
    # 0.5 * 0.75**23 = 0.00067 < 0.05: healed long before the end.
    assert result.penalty_paid is False
    assert result.total_actual == 24.0


def test_audit_counterfactual_consistency():
    result = audit_trace(DEMO_TRACE)
    assert result.total_counterfactual == (
        result.base_cost
        + result.counterfactual_spend
        + result.counterfactual_rework
    )
    assert result.excess_cost == pytest.approx(
        result.total_actual - result.total_counterfactual
    )
    assert result.recommendation  # non-empty text
