import math

import pytest

from checkpoint_econ.models import Workflow
from checkpoint_econ.optimize import (
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
    segment_cost,
    segment_fault_prob,
    strategy_table,
    unrecovered_prob,
    validated_segment_cost,
)


def workflow(costs, hazards, **overrides):
    kwargs = {"name": "t", "step_costs": costs, "step_hazards": hazards}
    kwargs.update(overrides)
    return Workflow(**kwargs)


def test_segment_cost_and_fault_prob():
    wf = workflow([1.0, 2.0, 3.0], [0.5, 0.5, 0.0])
    assert segment_cost(wf, 0, 2) == 3.0
    assert segment_cost(wf, 1, 3) == 5.0
    assert segment_fault_prob(wf, 0, 2) == 0.75
    assert segment_fault_prob(wf, 2, 3) == 0.0


def test_validated_segment_cost():
    wf = workflow([2.0], [0.5])
    assert validated_segment_cost(wf, 0, 1) == 4.0
    wf_clean = workflow([2.0], [0.0])
    assert validated_segment_cost(wf_clean, 0, 1) == 2.0


def test_validated_segment_cost_certain_fault_is_infinite():
    wf = workflow([2.0], [1.0])
    assert validated_segment_cost(wf, 0, 1) == math.inf


def test_unrecovered_prob_zero_hazard():
    wf = workflow([1.0, 1.0], [0.0, 0.0])
    assert unrecovered_prob(wf, 0, 2) == 0.0


def test_unrecovered_prob_last_step_fault():
    # A fault on the final step has no time to decay and every severity
    # in [0.4, 1.0] exceeds the 0.05 threshold, so the probability is
    # exactly the step hazard.
    wf = workflow([1.0, 1.0], [0.0, 0.25])
    assert unrecovered_prob(wf, 0, 2) == pytest.approx(0.25)


def test_unrecovered_prob_persistent_fault_anywhere():
    # With persistence 1.0 a fault anywhere is unrecovered (severities
    # always exceed the threshold), so the probability equals the
    # probability of at least one fault in the range.
    wf = workflow([1.0, 1.0, 1.0], [0.1, 0.2, 0.3], persistence=1.0)
    expected = 1.0 - 0.9 * 0.8 * 0.7
    assert unrecovered_prob(wf, 0, 3) == pytest.approx(expected)


def test_unrecovered_prob_full_repair_kills_old_faults():
    # repair_rate 1.0: a decaying fault one step before the end is
    # fully repaired; only the persistent fraction survives.
    wf = workflow([1.0, 1.0], [0.5, 0.0], repair_rate=1.0, persistence=0.3)
    assert unrecovered_prob(wf, 0, 2) == pytest.approx(0.5 * 0.3)


def test_unrecovered_prob_rejects_beyond_end():
    wf = workflow([1.0], [0.1])
    with pytest.raises(ValueError):
        unrecovered_prob(wf, 0, 2)


def test_expected_cost_never_clean_workflow():
    wf = workflow([1.0, 2.0], [0.0, 0.0])
    assert expected_cost_never(wf) == 3.0
    assert expected_cost_repair(wf) == 3.0


def test_expected_cost_placement_manual():
    wf = workflow([1.0, 1.0], [0.5, 0.5], checkpoint_cost=3.0)
    # Checkpoint at 2: one validated segment, q = 0.75 -> 2 / 0.25 = 8 + 3.
    assert expected_cost_placement(wf, [2]) == pytest.approx(11.0)
    # Checkpoints at 1 and 2: each segment q = 0.5 -> 1 / 0.5 = 2, twice,
    # plus two checkpoints.
    assert expected_cost_placement(wf, [1, 2]) == pytest.approx(10.0)


def test_expected_cost_placement_rejects_out_of_range():
    wf = workflow([1.0], [0.1])
    with pytest.raises(ValueError):
        expected_cost_placement(wf, [2])


def test_fixed_interval_checkpoints():
    assert fixed_interval_checkpoints(10, 3) == [3, 6, 9, 10]
    assert fixed_interval_checkpoints(10, 5) == [5, 10]
    assert fixed_interval_checkpoints(8, 8) == [8]
    assert fixed_interval_checkpoints(4, 1) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        fixed_interval_checkpoints(10, 0)


def test_every_step_matches_interval_one():
    wf = workflow([1.0] * 5, [0.1] * 5)
    assert expected_cost_every_step(wf) == expected_cost_fixed_interval(wf, 1)


def test_restart_clean_workflow():
    wf = workflow([2.0, 3.0], [0.0, 0.0])
    assert expected_cost_restart(wf) == 5.0


def test_restart_geometric():
    wf = workflow([2.0], [0.5], penalty=600.0)
    expected = 2.0 * (1.0 - 0.5 ** 50) / 0.5 + (0.5 ** 50) * 600.0
    assert expected_cost_restart(wf, max_passes=50) == pytest.approx(expected)


def test_optimal_placement_clean_workflow_buys_nothing():
    wf = workflow([1.0] * 6, [0.0] * 6)
    points, cost = optimal_placement(wf)
    assert points == []
    assert cost == pytest.approx(6.0)


def test_optimal_placement_never_worse_than_baselines():
    wf = workflow(
        [1.0] * 16,
        [0.02] * 5 + [0.35] + [0.02] * 4 + [0.35] + [0.02] * 5,
    )
    points, cost = optimal_placement(wf)
    assert cost <= expected_cost_every_step(wf) + 1e-9
    assert cost <= expected_cost_never(wf) + 1e-9
    _interval, fixed = best_fixed_interval(wf)
    assert cost <= fixed + 1e-9
    # The placement prices itself consistently through the public API.
    assert expected_cost_placement(wf, points) == pytest.approx(cost)


def test_optimal_placement_covers_spike():
    # One dangerous step (step 4, hazard 0.4): the optimiser puts a
    # checkpoint immediately after it so a fault there is caught and
    # rolled back instead of reaching the end unvalidated, and the
    # total cost collapses vs never checkpointing (272.46).
    wf = workflow([1.0] * 8, [0.01] * 3 + [0.4] + [0.01] * 4)
    points, cost = optimal_placement(wf)
    assert 4 in points
    assert cost == pytest.approx(17.03, abs=0.01)
    assert cost < 0.1 * expected_cost_never(wf)


def test_best_fixed_interval_grid():
    wf = workflow([1.0] * 8, [0.08] * 8)
    interval, cost = best_fixed_interval(wf)
    assert 1 <= interval <= 8
    assert cost == pytest.approx(expected_cost_fixed_interval(wf, interval))
    for other in range(1, 9):
        assert cost <= expected_cost_fixed_interval(wf, other) + 1e-9


def test_estimate_hazards_smoothing():
    assert estimate_hazards([0, 10], 10) == [1.0 / 12.0, 11.0 / 12.0]
    with pytest.raises(ValueError):
        estimate_hazards([11], 10)
    with pytest.raises(ValueError):
        estimate_hazards([0], 0)


def test_strategy_table_and_recommend():
    wf = workflow([1.0] * 12, [0.05] * 12)
    table = strategy_table(wf)
    strategies = [row["strategy"] for row in table]
    assert strategies == [
        "restart", "repair", "never", "every_step", "best_fixed", "optimal",
    ]
    result = recommend(wf)
    assert result["best"]["expected_cost"] == min(
        row["expected_cost"] for row in table
    )


def test_recommend_can_choose_restart_for_short_safe_runs():
    wf = workflow([1.0] * 8, [0.02] * 8)
    result = recommend(wf)
    assert result["best"]["strategy"] == "restart"
