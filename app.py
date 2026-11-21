"""Streamlit dashboard for checkpoint-recovery-economics.

Three tabs:
1. Optimise -- build a workflow shape and price every checkpoint
   strategy: restart, repair, never, every step, best fixed interval,
   and the DP-optimal placement.
2. Recovery -- fault recovery dynamics: how a planted fault's error
   decays under continue-and-self-repair, its half-life, and how many
   steps it needs to fall below threshold.
3. Evaluation -- the stored headline results (results/evaluation.json).

Run: streamlit run app.py
"""

import json
import os
import random

import streamlit as st

from checkpoint_econ.evaluate import hazards_for_profile
from checkpoint_econ.models import Workflow
from checkpoint_econ.optimize import recommend
from checkpoint_econ.simulate import (
    analytic_half_life,
    analytic_recovery_time,
    decay_trajectory,
    measure_recovery,
)

st.set_page_config(page_title="Checkpoint Recovery Economics", layout="wide")

PROFILES = ["uniform_low", "uniform_high", "spiky", "back_loaded"]


def _workflow(profile, n_steps, checkpoint_cost, penalty, repair_rate):
    rng = random.Random(99)
    costs = [round(rng.uniform(0.6, 1.6), 4) for _ in range(n_steps)]
    hazards = hazards_for_profile(profile, n_steps, rng)
    return Workflow(
        name="%s-%d" % (profile, n_steps),
        step_costs=costs,
        step_hazards=hazards,
        checkpoint_cost=checkpoint_cost,
        penalty=penalty,
        repair_rate=repair_rate,
    )


def tab_optimise():
    st.header("Price a checkpoint strategy")
    st.write(
        "A checkpoint is insurance: it costs every run and pays out only "
        "when a fault lands in the segment it closes. The optimiser "
        "places checkpoints where the expected payout beats the premium."
    )
    col1, col2, col3 = st.columns(3)
    profile = col1.selectbox("Hazard profile", PROFILES, index=2)
    n_steps = col2.slider("Workflow length (steps)", 4, 64, 32, 4)
    repair_rate = col3.slider("Repair rate", 0.0, 0.9, 0.25, 0.05)
    col4, col5 = st.columns(2)
    checkpoint_cost = col4.slider("Checkpoint cost", 0.5, 20.0, 3.0, 0.5)
    penalty = col5.slider("Failure penalty", 0.0, 2000.0, 600.0, 25.0)
    workflow = _workflow(
        profile, int(n_steps), float(checkpoint_cost), float(penalty),
        float(repair_rate),
    )
    result = recommend(workflow)
    best = result["best"]
    st.write(
        "**Recommendation: %s** at expected cost %.2f (checkpoints: %s)."
        % (
            best["strategy"],
            best["expected_cost"],
            best["checkpoints"] if best["checkpoints"] else "none",
        )
    )
    rows = []
    for row in result["table"]:
        rows.append(
            {
                "strategy": row["strategy"],
                "checkpoints": (
                    ", ".join(str(p) for p in row["checkpoints"])
                    if row["checkpoints"]
                    else "-"
                ),
                "expected cost": round(row["expected_cost"], 2),
            }
        )
    st.dataframe(rows, use_container_width=True)
    st.bar_chart(
        {"expected cost": [row["expected cost"] for row in rows]},
    )
    st.caption(
        "Bars in table order: restart, repair, never, every step, best "
        "fixed interval, optimal placement. Base cost of one clean pass: "
        "%.2f." % workflow.total_cost
    )


def tab_recovery():
    st.header("How fast does a fault heal itself?")
    st.write(
        "Under continue-and-self-repair a fault's error decays by the "
        "repair rate each step. Its half-life decides whether a late "
        "fault can heal before the run ends -- if it cannot, only a "
        "checkpoint or a restart saves the run."
    )
    col1, col2, col3 = st.columns(3)
    repair_rate = col1.slider("Repair rate", 0.02, 0.95, 0.25, 0.01)
    severity = col2.slider("Fault severity", 0.1, 1.0, 0.8, 0.05)
    threshold = col3.slider("Error threshold", 0.01, 0.30, 0.05, 0.01)
    measured = measure_recovery(float(severity), float(repair_rate),
                                float(threshold))
    m1, m2 = st.columns(2)
    m1.metric(
        "Half-life (steps)",
        "%.2f analytic / %s measured"
        % (analytic_half_life(float(repair_rate)),
           measured["half_life_steps"]),
    )
    m2.metric(
        "Recovery time to threshold (steps)",
        "%.2f analytic / %s measured"
        % (
            analytic_recovery_time(float(severity), float(repair_rate),
                                   float(threshold)),
            measured["recovery_steps"],
        ),
    )
    horizon = 40
    trajectory = decay_trajectory(float(severity), float(repair_rate), horizon)
    st.line_chart({"state error": trajectory})
    st.caption(
        "Error of one fault over the %d steps after injection. A fault "
        "injected fewer steps than the recovery time before the end of "
        "a run finishes above threshold and pays the penalty."
        % horizon
    )


def tab_evaluation():
    st.header("Headline evaluation (stored results)")
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "evaluation.json"
    )
    if not os.path.exists(path):
        st.info("No stored results yet. Run: python -m checkpoint_econ evaluate")
        return
    with open(path) as handle:
        evaluation = json.load(handle)
    agg = evaluation["aggregates"]
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Saving vs restart", "%.1f%%" % agg["mean_savings_vs_restart_pct"]
    )
    m2.metric(
        "Saving vs every-step", "%.1f%%" % agg["mean_savings_vs_every_step_pct"]
    )
    m3.metric("Tool regret vs oracle", "%.2f%%" % agg["mean_tool_regret_pct"])
    rows = []
    for scenario in evaluation["scenarios"]:
        rows.append(
            {
                "scenario": scenario["scenario"],
                "restart": round(scenario["restart"], 2),
                "every step": round(scenario["every_step"], 2),
                "oracle fixed": round(scenario["oracle_fixed"], 2),
                "oracle optimal": round(scenario["oracle_optimal"], 2),
                "tool optimal": round(scenario["tool_optimal"], 2),
                "tool regret %": round(scenario["tool_regret_pct"], 2),
            }
        )
    st.dataframe(rows, use_container_width=True)
    show = agg["showcase"]
    st.write(
        "Showcase (%s): restart %.2f, every-step %.2f, best fixed "
        "interval %d at %.2f, optimal placement %s at %.2f."
        % (
            show["scenario"],
            show["restart"],
            show["every_step"],
            show["oracle_fixed_interval"],
            show["oracle_fixed"],
            show["oracle_checkpoints"],
            show["oracle_optimal"],
        )
    )
    st.caption(
        "All workflows are synthetic and seeded, with hazards known by "
        "construction; see results/evaluation_summary.md."
    )


tab1, tab2, tab3 = st.tabs(["Optimise", "Recovery", "Evaluation"])
with tab1:
    tab_optimise()
with tab2:
    tab_recovery()
with tab3:
    tab_evaluation()
