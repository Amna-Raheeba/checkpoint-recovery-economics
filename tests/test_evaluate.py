import random

import pytest

from checkpoint_econ.evaluate import (
    DEFAULT_CONFIG,
    SMOKE_CONFIG,
    build_workflow,
    check_drift,
    hazards_for_profile,
    run_evaluation,
)


def test_hazards_profiles():
    rng = random.Random(0)
    assert hazards_for_profile("uniform_low", 5, rng) == [0.02] * 5
    assert hazards_for_profile("uniform_high", 5, rng) == [0.08] * 5
    back = hazards_for_profile("back_loaded", 5, rng)
    assert back[0] == pytest.approx(0.01)
    assert back[-1] == pytest.approx(0.12)
    assert back == sorted(back)
    with pytest.raises(ValueError):
        hazards_for_profile("nope", 5, rng)


def test_hazards_spiky_counts():
    rng = random.Random(1)
    hazards = hazards_for_profile("spiky", 64, rng)
    assert sum(1 for h in hazards if h == 0.35) == 10
    assert sum(1 for h in hazards if h == 0.01) == 54


def test_build_workflow_deterministic():
    first = build_workflow("spiky", 16, DEFAULT_CONFIG, 1234)
    second = build_workflow("spiky", 16, DEFAULT_CONFIG, 1234)
    assert first == second
    assert first.name == "spiky-16"
    assert first.penalty == DEFAULT_CONFIG["penalty"]
    assert first.persistence == DEFAULT_CONFIG["persistence"]


def test_smoke_evaluation_shape():
    evaluation = run_evaluation(SMOKE_CONFIG)
    assert evaluation["aggregates"]["n_scenarios"] == 8
    assert len(evaluation["scenarios"]) == 8
    assert len(evaluation["recovery"]) == 3
    for row in evaluation["scenarios"]:
        assert row["oracle_optimal"] <= row["every_step"] + 1e-9
        assert row["oracle_optimal"] <= row["oracle_fixed"] + 1e-9
        assert row["tool_regret_pct"] > -1.0  # estimation noise only
        assert row["recommendation_regret_pct"] >= -1e-9


def test_evaluation_aggregates_present():
    evaluation = run_evaluation(SMOKE_CONFIG)
    agg = evaluation["aggregates"]
    for key in (
        "mean_savings_vs_restart_pct",
        "mean_savings_vs_every_step_pct",
        "mean_fixed_vs_optimal_gap_pct",
        "fixed_gap_spiky_pct",
        "mean_tool_regret_pct",
        "mean_recommendation_regret_pct",
        "mean_interval_abs_error",
        "mean_mc_deviation_pct",
        "showcase",
    ):
        assert key in agg
    assert agg["showcase"]["scenario"] == "spiky-32"


def test_check_drift_identical():
    evaluation = run_evaluation(SMOKE_CONFIG)
    ok, problems = check_drift(evaluation, evaluation)
    assert ok
    assert problems == []


def test_check_drift_detects_change():
    evaluation = run_evaluation(SMOKE_CONFIG)
    tampered = {"aggregates": dict(evaluation["aggregates"])}
    tampered["aggregates"]["mean_tool_regret_pct"] += 5.0
    ok, problems = check_drift(evaluation, tampered)
    assert not ok
    assert problems[0][0] == "mean_tool_regret_pct"
