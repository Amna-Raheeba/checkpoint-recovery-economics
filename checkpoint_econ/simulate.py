"""Seeded Monte Carlo simulation of workflow runs under recovery policies.

The simulator is the ground-truth check on the analytic cost model in
``optimize``: it executes workflows step by step with a seeded RNG,
injects faults at the declared per-step hazards, applies one recovery
policy, and records what the run actually cost.

Policies:

- "restart": no checkpoints. A pass that injects any fault is discarded
  and the workflow restarts from step 0, up to ``max_passes`` passes;
  if every pass faults, the failure penalty is paid.
- "rollback": checkpoints at the given positions. A segment that
  injects a fault is re-executed until it runs clean (capped at
  ``max_segment_attempts`` attempts, after which the run gives up on
  the segment and pays the penalty -- the analytic cost of such a
  segment is infinite anyway). A trailing unvalidated segment is
  executed once and its faults decay under the repair dynamics; the
  penalty is paid if the residual error at the end exceeds the
  threshold.
- "repair": no checkpoints and no rework. Faults inject error, each
  subsequent step repairs a fraction of it, and the penalty is paid if
  the residual error at the end exceeds the threshold.

It also generates first-pass fault logs (for hazard estimation) and
measures fault recovery: how many steps a single planted fault's error
needs to decay to half (its half-life) and below threshold.
"""

import math
import random

POLICIES = ("restart", "rollback", "repair")


def _draw_faults(workflow, a, b, rng):
    """Faults injected by one pass over steps a+1 .. b.

    Returns a list of (step, severity, persistent) triples; a fault is
    persistent with probability ``workflow.persistence``.
    """
    faults = []
    for step in range(a + 1, b + 1):
        if rng.random() < workflow.step_hazards[step - 1]:
            severity = rng.uniform(workflow.severity_lo, workflow.severity_hi)
            persistent = rng.random() < workflow.persistence
            faults.append((step, severity, persistent))
    return faults


def _residual_error(workflow, faults, end):
    """Error remaining at position ``end`` from faults at or before it.

    ``faults`` holds (step, severity) pairs or (step, severity,
    persistent) triples. Persistent faults never decay.
    """
    decay = 1.0 - workflow.repair_rate
    error = 0.0
    for fault in faults:
        step, severity = fault[0], fault[1]
        persistent = fault[2] if len(fault) > 2 else False
        if step > end:
            continue
        if persistent:
            error = max(error, severity)
            continue
        remaining = end - step
        if decay <= 0.0:
            factor = 1.0 if remaining == 0 else 0.0
        else:
            factor = decay ** remaining
        error = max(error, severity * factor)
    return error


def simulate_run(workflow, policy, rng, checkpoints=(), max_passes=50,
               max_segment_attempts=1000):
    """Simulate one run. Returns a dict of measured outcomes.

    Keys: total_cost, execution_cost, checkpoint_spend, penalty_paid
    (bool), passes (workflow passes started), segment_attempts,
    faults_seen, final_error.
    """
    if policy not in POLICIES:
        raise ValueError("unknown policy: %r" % (policy,))
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")
    n = workflow.n_steps
    base = workflow.total_cost

    if policy == "restart":
        passes = 0
        faults_seen = 0
        penalty_paid = False
        while True:
            passes += 1
            faults = _draw_faults(workflow, 0, n, rng)
            faults_seen += len(faults)
            if not faults:
                break
            if passes >= max_passes:
                penalty_paid = True
                break
        total = base * passes + (workflow.penalty if penalty_paid else 0.0)
        return {
            "total_cost": total,
            "execution_cost": base * passes,
            "checkpoint_spend": 0.0,
            "penalty_paid": penalty_paid,
            "passes": passes,
            "segment_attempts": passes,
            "faults_seen": faults_seen,
            "final_error": 0.0,
        }

    if policy == "repair":
        error = 0.0
        permanent = 0.0
        faults_seen = 0
        decay = 1.0 - workflow.repair_rate
        for step in range(1, n + 1):
            error *= decay
            if rng.random() < workflow.step_hazards[step - 1]:
                severity = rng.uniform(workflow.severity_lo, workflow.severity_hi)
                if rng.random() < workflow.persistence:
                    permanent = max(permanent, severity)
                else:
                    error = max(error, severity)
                faults_seen += 1
        error = max(error, permanent)
        penalty_paid = error > workflow.error_threshold
        total = base + (workflow.penalty if penalty_paid else 0.0)
        return {
            "total_cost": total,
            "execution_cost": base,
            "checkpoint_spend": 0.0,
            "penalty_paid": penalty_paid,
            "passes": 1,
            "segment_attempts": 1,
            "faults_seen": faults_seen,
            "final_error": error,
        }

    # rollback
    points = sorted(set(checkpoints))
    for position in points:
        if position < 1 or position > n:
            raise ValueError("checkpoint position out of range: %r" % (position,))
    execution = 0.0
    attempts = 0
    faults_seen = 0
    penalty_paid = False
    start = 0
    for position in points:
        segment_attempts = 0
        while True:
            attempts += 1
            segment_attempts += 1
            execution += sum(workflow.step_costs[start:position])
            faults = _draw_faults(workflow, start, position, rng)
            faults_seen += len(faults)
            if not faults:
                break
            if segment_attempts >= max_segment_attempts:
                penalty_paid = True
                break
        start = position
    final_error = 0.0
    if start < n:
        execution += sum(workflow.step_costs[start:n])
        faults = _draw_faults(workflow, start, n, rng)
        faults_seen += len(faults)
        attempts += 1
        final_error = _residual_error(workflow, faults, n)
        penalty_paid = penalty_paid or (final_error > workflow.error_threshold)
    spend = workflow.checkpoint_cost * len(points)
    total = execution + spend + (workflow.penalty if penalty_paid else 0.0)
    return {
        "total_cost": total,
        "execution_cost": execution,
        "checkpoint_spend": spend,
        "penalty_paid": penalty_paid,
        "passes": 1,
        "segment_attempts": attempts,
        "faults_seen": faults_seen,
        "final_error": final_error,
    }


def simulate_policy(workflow, policy, runs, seed, checkpoints=(), max_passes=50):
    """Aggregate ``runs`` seeded simulations of one policy.

    Returns a dict: runs, mean_cost, min_cost, max_cost, penalty_rate,
    mean_faults.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    rng = random.Random(seed)
    total = 0.0
    min_cost = math.inf
    max_cost = 0.0
    penalties = 0
    faults = 0
    for _ in range(runs):
        outcome = simulate_run(workflow, policy, rng, checkpoints, max_passes)
        total += outcome["total_cost"]
        min_cost = min(min_cost, outcome["total_cost"])
        max_cost = max(max_cost, outcome["total_cost"])
        penalties += 1 if outcome["penalty_paid"] else 0
        faults += outcome["faults_seen"]
    return {
        "runs": runs,
        "mean_cost": total / runs,
        "min_cost": min_cost,
        "max_cost": max_cost,
        "penalty_rate": penalties / float(runs),
        "mean_faults": faults / float(runs),
    }


def generate_fault_log(workflow, runs, seed):
    """First-pass fault counts per step over ``runs`` logged runs.

    Returns a list of length n_steps: entry i counts the runs in which
    step i+1 injected a fault on the first pass. This is the raw
    material a team would have from its own run logs.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    rng = random.Random(seed)
    counts = [0] * workflow.n_steps
    for _ in range(runs):
        for step in range(1, workflow.n_steps + 1):
            if rng.random() < workflow.step_hazards[step - 1]:
                counts[step - 1] += 1
    return counts


def decay_trajectory(severity, repair_rate, steps):
    """Error after each of the next ``steps`` steps for one fault.

    Entry t is the error t steps after injection (t from 0 to steps).
    """
    if severity < 0:
        raise ValueError("severity must be non-negative")
    if not 0.0 <= repair_rate <= 1.0:
        raise ValueError("repair_rate must lie in [0, 1]")
    decay = 1.0 - repair_rate
    return [severity * (decay ** t) for t in range(steps + 1)]


def analytic_half_life(repair_rate):
    """Steps for a fault's error to halve: ln(0.5) / ln(1 - r)."""
    if repair_rate <= 0.0:
        return math.inf
    if repair_rate >= 1.0:
        return 0.0
    return math.log(0.5) / math.log(1.0 - repair_rate)


def analytic_recovery_time(severity, repair_rate, threshold):
    """Steps until a fault's error decays to ``threshold`` (0 if already)."""
    if severity <= threshold:
        return 0.0
    if repair_rate <= 0.0:
        return math.inf
    if repair_rate >= 1.0:
        return 1.0
    return math.log(threshold / severity) / math.log(1.0 - repair_rate)


def measure_recovery(severity, repair_rate, threshold, max_steps=100000):
    """Empirical recovery of one planted fault, from the decay trajectory.

    Returns a dict: half_life_steps (first t with error <= severity/2),
    recovery_steps (first t with error <= threshold, or None when the
    fault never recovers within max_steps), analytic_half_life,
    analytic_recovery_time.
    """
    trajectory = decay_trajectory(severity, repair_rate, max_steps)
    half = None
    recovery = None
    for t, error in enumerate(trajectory):
        if half is None and error <= severity / 2.0:
            half = t
        if recovery is None and error <= threshold:
            recovery = t
        if half is not None and recovery is not None:
            break
    return {
        "half_life_steps": half,
        "recovery_steps": recovery,
        "analytic_half_life": analytic_half_life(repair_rate),
        "analytic_recovery_time": analytic_recovery_time(
            severity, repair_rate, threshold
        ),
    }
