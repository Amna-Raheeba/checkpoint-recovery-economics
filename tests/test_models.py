import pytest

from checkpoint_econ.models import Trace, Workflow


def make_workflow(**overrides):
    kwargs = {
        "name": "w",
        "step_costs": [1.0, 2.0, 3.0],
        "step_hazards": [0.1, 0.2, 0.3],
    }
    kwargs.update(overrides)
    return Workflow(**kwargs)


def test_workflow_basics():
    workflow = make_workflow()
    assert workflow.n_steps == 3
    assert workflow.total_cost == 6.0
    assert workflow.mean_step_cost == 2.0


def test_workflow_defaults():
    workflow = make_workflow()
    assert workflow.checkpoint_cost == 3.0
    assert workflow.penalty == 600.0
    assert workflow.repair_rate == 0.25
    assert workflow.persistence == 0.3
    assert workflow.error_threshold == 0.05


def test_workflow_rejects_length_mismatch():
    with pytest.raises(ValueError):
        make_workflow(step_hazards=[0.1])


def test_workflow_rejects_bad_hazard():
    with pytest.raises(ValueError):
        make_workflow(step_hazards=[0.1, 1.5, 0.3])
    with pytest.raises(ValueError):
        make_workflow(step_hazards=[0.1, -0.1, 0.3])


def test_workflow_rejects_nonpositive_cost():
    with pytest.raises(ValueError):
        make_workflow(step_costs=[1.0, 0.0, 3.0])


def test_workflow_rejects_bad_persistence():
    with pytest.raises(ValueError):
        make_workflow(persistence=1.5)


def test_workflow_rejects_inverted_severity():
    with pytest.raises(ValueError):
        make_workflow(severity_lo=0.9, severity_hi=0.4)


def test_workflow_round_trip():
    workflow = make_workflow()
    clone = Workflow.from_dict(workflow.to_dict())
    assert clone == workflow


def test_workflow_from_dict_defaults():
    workflow = Workflow.from_dict(
        {"name": "x", "step_costs": [1.0], "step_hazards": [0.5]}
    )
    assert workflow.penalty == 600.0
    assert workflow.persistence == 0.3


def test_with_hazards_keeps_economics():
    workflow = make_workflow()
    changed = workflow.with_hazards([0.0, 0.0, 0.0])
    assert changed.step_hazards == [0.0, 0.0, 0.0]
    assert changed.step_costs == workflow.step_costs
    assert changed.penalty == workflow.penalty
    assert changed.persistence == workflow.persistence


def test_trace_sorts_checkpoints():
    trace = Trace(
        workflow=make_workflow(),
        checkpoints_taken=[3, 1],
        faults=[{"step": 2, "severity": 0.5}],
        recovery="rollback",
    )
    assert trace.checkpoints_taken == [1, 3]


def test_trace_rejects_out_of_range_checkpoint():
    with pytest.raises(ValueError):
        Trace(workflow=make_workflow(), checkpoints_taken=[4])


def test_trace_rejects_duplicate_checkpoints():
    with pytest.raises(ValueError):
        Trace(workflow=make_workflow(), checkpoints_taken=[2, 2])


def test_trace_rejects_bad_recovery():
    with pytest.raises(ValueError):
        Trace(workflow=make_workflow(), recovery="rewind")


def test_trace_rejects_bad_fault():
    with pytest.raises(ValueError):
        Trace(workflow=make_workflow(), faults=[{"step": 9, "severity": 0.5}])
    with pytest.raises(ValueError):
        Trace(workflow=make_workflow(), faults=[{"step": 1}])


def test_trace_round_trip():
    trace = Trace(
        workflow=make_workflow(),
        checkpoints_taken=[2],
        faults=[{"step": 1, "severity": 0.7}],
        recovery="repair",
    )
    clone = Trace.from_dict(trace.to_dict())
    assert clone.checkpoints_taken == [2]
    assert clone.faults == [{"step": 1, "severity": 0.7}]
    assert clone.recovery == "repair"
    assert clone.workflow == trace.workflow
