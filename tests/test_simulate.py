import math
import random

import pytest

from checkpoint_econ.models import Workflow
from checkpoint_econ.simulate import (
    analytic_half_life,
    analytic_recovery_time,
    decay_trajectory,
    generate_fault_log,
    measure_recovery,
    simulate_policy,
    simulate_run,
)


def workflow(costs, hazards, **overrides):
    kwargs = {"name": "t", "step_costs": costs, "step_hazards": hazards}
    kwargs.update(overrides)
    return Workflow(**kwargs)


def test_restart_clean_workflow_single_pass():
    wf = workflow([2.0, 3.0], [0.0, 0.0])
    outcome = simulate_run(wf, "restart", random.Random(1))
    assert outcome["passes"] == 1
    assert outcome["total_cost"] == 5.0
    assert outcome["penalty_paid"] is False


def test_restart_certain_fault_hits_cap_and_penalty():
    wf = workflow([2.0], [1.0], penalty=600.0)
    outcome = simulate_run(wf, "restart", random.Random(1), max_passes=5)
    assert outcome["passes"] == 5
    assert outcome["penalty_paid"] is True
    assert outcome["total_cost"] == 5 * 2.0 + 600.0


def test_repair_clean_workflow():
    wf = workflow([2.0, 3.0], [0.0, 0.0])
    outcome = simulate_run(wf, "repair", random.Random(1))
    assert outcome["total_cost"] == 5.0
    assert outcome["final_error"] == 0.0
    assert outcome["penalty_paid"] is False


def test_repair_persistent_fault_always_penalised():
    # hazard 1.0 + persistence 1.0: the fault never decays, severity
    # >= 0.4 > threshold, so the penalty is certain.
    wf = workflow([1.0] * 4, [1.0, 0.0, 0.0, 0.0], persistence=1.0)
    outcome = simulate_run(wf, "repair", random.Random(3))
    assert outcome["penalty_paid"] is True
    assert outcome["total_cost"] == 4.0 + wf.penalty


def test_rollback_clean_workflow_pays_checkpoints_only():
    wf = workflow([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], checkpoint_cost=3.0)
    outcome = simulate_run(wf, "rollback", random.Random(1), checkpoints=[3])
    assert outcome["total_cost"] == 3.0 + 3.0
    assert outcome["checkpoint_spend"] == 3.0
    assert outcome["segment_attempts"] == 1


def test_rollback_retries_faulted_segment():
    # Step 1 faults with probability 1 on the first attempt in
    # expectation terms -- use a seeded run and only assert the cost
    # accounting identity, which holds for every outcome.
    wf = workflow([1.0, 1.0], [0.9, 0.0], checkpoint_cost=2.0)
    outcome = simulate_run(wf, "rollback", random.Random(11), checkpoints=[1, 2])
    assert outcome["total_cost"] == (
        outcome["execution_cost"]
        + outcome["checkpoint_spend"]
        + (wf.penalty if outcome["penalty_paid"] else 0.0)
    )
    assert outcome["checkpoint_spend"] == 4.0
    assert outcome["execution_cost"] >= 2.0


def test_simulate_run_rejects_unknown_policy():
    wf = workflow([1.0], [0.0])
    with pytest.raises(ValueError):
        simulate_run(wf, "rewind", random.Random(1))


def test_simulate_policy_deterministic():
    wf = workflow([1.0] * 8, [0.1] * 8)
    first = simulate_policy(wf, "repair", 50, 42)
    second = simulate_policy(wf, "repair", 50, 42)
    assert first == second


def test_simulate_policy_aggregates_shape():
    wf = workflow([1.0] * 4, [0.2] * 4)
    result = simulate_policy(wf, "restart", 30, 7)
    assert result["runs"] == 30
    assert result["min_cost"] <= result["mean_cost"] <= result["max_cost"]
    assert 0.0 <= result["penalty_rate"] <= 1.0
    assert result["mean_faults"] >= 0.0


def test_monte_carlo_tracks_analytic_restart():
    # Seeded, so this is a fixed check: 400 restarts of a 4-step
    # workflow must land near the analytic expectation.
    from checkpoint_econ.optimize import expected_cost_restart

    wf = workflow([1.0] * 4, [0.2] * 4)
    result = simulate_policy(wf, "restart", 400, 123)
    analytic = expected_cost_restart(wf)
    assert abs(result["mean_cost"] - analytic) / analytic < 0.15


def test_monte_carlo_tracks_analytic_repair():
    from checkpoint_econ.optimize import expected_cost_repair

    wf = workflow([1.0] * 6, [0.15] * 6)
    result = simulate_policy(wf, "repair", 600, 321)
    analytic = expected_cost_repair(wf)
    assert abs(result["mean_cost"] - analytic) / analytic < 0.15


def test_generate_fault_log_deterministic_and_bounded():
    wf = workflow([1.0] * 5, [0.5] * 5)
    first = generate_fault_log(wf, 40, 9)
    second = generate_fault_log(wf, 40, 9)
    assert first == second
    assert all(0 <= count <= 40 for count in first)


def test_generate_fault_log_zero_hazard():
    wf = workflow([1.0] * 3, [0.0] * 3)
    assert generate_fault_log(wf, 10, 1) == [0, 0, 0]


def test_decay_trajectory():
    assert decay_trajectory(1.0, 0.5, 3) == [1.0, 0.5, 0.25, 0.125]
    assert decay_trajectory(0.8, 0.0, 2) == [0.8, 0.8, 0.8]
    with pytest.raises(ValueError):
        decay_trajectory(-1.0, 0.5, 3)


def test_analytic_half_life():
    assert analytic_half_life(0.25) == pytest.approx(2.4094, abs=1e-3)
    assert analytic_half_life(0.0) == math.inf
    assert analytic_half_life(1.0) == 0.0
    assert analytic_half_life(0.5) == pytest.approx(1.0)


def test_analytic_recovery_time():
    assert analytic_recovery_time(0.8, 0.25, 0.05) == pytest.approx(9.6377, abs=1e-3)
    assert analytic_recovery_time(0.04, 0.25, 0.05) == 0.0
    assert analytic_recovery_time(0.8, 0.0, 0.05) == math.inf


def test_measure_recovery_matches_trajectory():
    measured = measure_recovery(0.8, 0.25, 0.05)
    # 0.8 * 0.75**3 = 0.3375 <= 0.4 (first halving), and
    # 0.8 * 0.75**10 = 0.0451 <= 0.05 (first crossing of threshold).
    assert measured["half_life_steps"] == 3
    assert measured["recovery_steps"] == 10
    assert measured["analytic_half_life"] == pytest.approx(2.4094, abs=1e-3)


def test_measure_recovery_no_repair_never_recovers():
    measured = measure_recovery(0.8, 0.0, 0.05, max_steps=50)
    assert measured["recovery_steps"] is None
    assert measured["half_life_steps"] is None
