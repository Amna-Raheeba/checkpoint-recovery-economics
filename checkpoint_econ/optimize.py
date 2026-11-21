"""Analytic expected costs and the checkpoint-placement optimiser.

Positions are step counts: position 0 is the start, position b is the
state after step b. A segment (a, b] is the run of steps a+1 .. b.

Cost model (all quantities in the workflow's cost unit):

- A *validated* segment ends at a checkpoint. Faults inside it are
  detected at the checkpoint and the segment is re-executed until it
  runs clean, so its expected execution cost is
  ``segment_cost / (1 - q)`` where ``q`` is the probability that one
  pass over the segment injects at least one fault. The checkpoint
  itself costs ``checkpoint_cost``.
- An *unvalidated* final segment is executed once. A fault inside it is
  never rolled back; it only matters if its error contribution has not
  decayed below the error threshold by the end of the workflow, in
  which case the failure penalty is paid. The decay is the repair
  dynamics: a decaying fault of severity s at step k contributes
  ``s * (1 - repair_rate) ** (N - k)`` at the end, while a persistent
  fault (probability ``persistence``) contributes its full severity
  forever -- it corrupted a downstream dependency.
- *Restart* recovery uses no checkpoints: the whole workflow is
  re-executed until a pass is clean (capped at ``max_passes`` passes,
  after which the penalty is paid).
- *Repair* recovery never re-executes anything: expected cost is the
  base cost plus the penalty times the probability that some fault's
  contribution survives to the end above threshold.

The optimiser searches checkpoint placements by dynamic programming and
also scores the standard baselines (never checkpoint, every step, and
every fixed interval) so a placement decision is always reported next
to the policies teams actually use.
"""

import math

_EPS = 1e-12


def segment_cost(workflow, a, b):
    """Execution cost of one pass over steps a+1 .. b."""
    return sum(workflow.step_costs[a:b])


def segment_fault_prob(workflow, a, b):
    """Probability that one pass over steps a+1 .. b injects a fault."""
    survival = 1.0
    for hazard in workflow.step_hazards[a:b]:
        survival *= 1.0 - hazard
    return 1.0 - survival


def validated_segment_cost(workflow, a, b):
    """Expected execution cost of segment (a, b] with rollback at b."""
    q = segment_fault_prob(workflow, a, b)
    if q >= 1.0 - _EPS:
        return math.inf
    return segment_cost(workflow, a, b) / (1.0 - q)


def _severity_exceedance(workflow, critical):
    """P(severity > critical) for severity ~ U[severity_lo, severity_hi]."""
    lo = workflow.severity_lo
    hi = workflow.severity_hi
    if hi <= lo:
        return 1.0 if lo > critical else 0.0
    prob = (hi - critical) / (hi - lo)
    return min(1.0, max(0.0, prob))


def unrecovered_prob(workflow, a, end):
    """P(some fault in steps a+1 .. end survives above threshold at end).

    A fault at step k is unrecovered in two ways: with probability
    ``persistence`` it is the persistent kind, which never decays and is
    unrecovered whenever its severity exceeds the threshold at all; or
    it is the decaying kind and its severity exceeds
    ``threshold / (1 - repair_rate) ** (end - k)``. Fault contributions
    combine by maximum, so the survival events multiply.
    """
    n = workflow.n_steps
    if end > n:
        raise ValueError("end position beyond workflow length")
    decay = 1.0 - workflow.repair_rate
    p_persistent_bad = _severity_exceedance(workflow, workflow.error_threshold)
    survival = 1.0
    for k in range(a + 1, end + 1):
        remaining = end - k
        if decay <= 0.0:
            factor = 1.0 if remaining == 0 else 0.0
        else:
            factor = decay ** remaining
        if factor <= 0.0:
            critical = math.inf
        else:
            critical = workflow.error_threshold / factor
        p_exceed = (
            workflow.persistence * p_persistent_bad
            + (1.0 - workflow.persistence)
            * _severity_exceedance(workflow, critical)
        )
        p_bad = workflow.step_hazards[k - 1] * p_exceed
        survival *= 1.0 - p_bad
    return 1.0 - survival


def expected_cost_placement(workflow, checkpoints):
    """Expected total cost of rollback recovery with these checkpoints.

    ``checkpoints`` is an iterable of positions in 1..N. Segments closed
    by a checkpoint are validated; a trailing segment after the last
    checkpoint (when N is not checkpointed) is unvalidated.
    """
    n = workflow.n_steps
    points = sorted(set(checkpoints))
    for position in points:
        if position < 1 or position > n:
            raise ValueError("checkpoint position out of range: %r" % (position,))
    total = 0.0
    start = 0
    for position in points:
        total += workflow.checkpoint_cost
        total += validated_segment_cost(workflow, start, position)
        start = position
    if start < n:
        total += segment_cost(workflow, start, n)
        total += workflow.penalty * unrecovered_prob(workflow, start, n)
    return total


def fixed_interval_checkpoints(n_steps, interval):
    """Checkpoint positions for a fixed interval: every interval, plus N."""
    if interval < 1:
        raise ValueError("interval must be at least 1")
    points = list(range(interval, n_steps, interval))
    if not points or points[-1] != n_steps:
        points.append(n_steps)
    return points


def expected_cost_fixed_interval(workflow, interval):
    points = fixed_interval_checkpoints(workflow.n_steps, interval)
    return expected_cost_placement(workflow, points)


def expected_cost_never(workflow):
    """No checkpoints at all: one pass, unvalidated to the end."""
    return expected_cost_placement(workflow, [])


def expected_cost_every_step(workflow):
    return expected_cost_fixed_interval(workflow, 1)


def expected_cost_restart(workflow, max_passes=50):
    """Restart-from-scratch: repeat whole passes until one is clean."""
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")
    q = segment_fault_prob(workflow, 0, workflow.n_steps)
    if q <= 0.0:
        return workflow.total_cost
    if q >= 1.0 - _EPS:
        return max_passes * workflow.total_cost + workflow.penalty
    passes = (1.0 - q ** max_passes) / (1.0 - q)
    return workflow.total_cost * passes + (q ** max_passes) * workflow.penalty


def expected_cost_repair(workflow):
    """Continue-and-self-repair: no rework, penalty if error survives."""
    return workflow.total_cost + workflow.penalty * unrecovered_prob(
        workflow, 0, workflow.n_steps
    )


def optimal_placement(workflow):
    """Cost-optimal checkpoint placement by dynamic programming.

    Returns (checkpoints, expected_cost). The DP builds validated
    segments left to right, then chooses the cheaper of closing the
    workflow with a validated final segment or leaving the tail
    unvalidated.
    """
    n = workflow.n_steps
    prefix_cost = [0.0]
    prefix_log_survival = [0.0]
    for i in range(n):
        prefix_cost.append(prefix_cost[-1] + workflow.step_costs[i])
        hazard = workflow.step_hazards[i]
        # Clamp instead of log(0): a hazard-1 step makes any segment
        # containing it infinitely expensive while leaving the prefix
        # differences of later segments meaningful.
        prefix_log_survival.append(
            prefix_log_survival[-1] + math.log(max(1.0 - hazard, 1e-15))
        )

    def seg_cost(a, b):
        return prefix_cost[b] - prefix_cost[a]

    def seg_validated(a, b):
        log_survival = prefix_log_survival[b] - prefix_log_survival[a]
        q = 1.0 - math.exp(log_survival)
        if q >= 1.0 - _EPS:
            return math.inf
        return seg_cost(a, b) / (1.0 - q)

    infinity = math.inf
    dp = [infinity] * (n + 1)
    parent = [-1] * (n + 1)
    dp[0] = 0.0
    for j in range(1, n + 1):
        for i in range(0, j):
            if dp[i] == infinity:
                continue
            candidate = dp[i] + workflow.checkpoint_cost + seg_validated(i, j)
            if candidate < dp[j]:
                dp[j] = candidate
                parent[j] = i

    best_cost = dp[n]
    best_tail_start = None
    for i in range(0, n):
        if dp[i] == infinity:
            continue
        candidate = (
            dp[i]
            + seg_cost(i, n)
            + workflow.penalty * unrecovered_prob(workflow, i, n)
        )
        if candidate < best_cost:
            best_cost = candidate
            best_tail_start = i

    # Reconstruct the checkpoint chain ending at the last validated
    # position (n when the tail is validated, best_tail_start otherwise).
    checkpoints = []
    cursor = n if best_tail_start is None else best_tail_start
    while cursor > 0:
        checkpoints.append(cursor)
        cursor = parent[cursor]
    checkpoints.reverse()
    return checkpoints, best_cost


def best_fixed_interval(workflow):
    """Best fixed checkpoint interval by exhaustive grid search.

    Returns (interval, expected_cost) -- the oracle fixed interval when
    evaluated on the true workflow.
    """
    best_interval = None
    best_cost = math.inf
    for interval in range(1, workflow.n_steps + 1):
        cost = expected_cost_fixed_interval(workflow, interval)
        if cost < best_cost:
            best_cost = cost
            best_interval = interval
    return best_interval, best_cost


def estimate_hazards(fault_counts, runs):
    """Laplace-smoothed per-step hazard estimates from an audit log.

    ``fault_counts[i]`` is the number of logged runs in which step i+1
    faulted on its first pass. Smoothing (count + 1) / (runs + 2) keeps
    unobserved steps at a small positive hazard instead of exactly 0.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    estimates = []
    for count in fault_counts:
        if count < 0 or count > runs:
            raise ValueError("fault count out of range: %r" % (count,))
        estimates.append((count + 1.0) / (runs + 2.0))
    return estimates


def strategy_table(workflow):
    """Expected costs of every strategy on this workflow.

    Returns a list of dicts with keys: strategy, checkpoints,
    expected_cost. Strategies: restart, repair, never, every_step,
    best_fixed (with its interval), optimal (DP placement).
    """
    interval, fixed_cost = best_fixed_interval(workflow)
    opt_points, opt_cost = optimal_placement(workflow)
    return [
        {
            "strategy": "restart",
            "checkpoints": [],
            "expected_cost": expected_cost_restart(workflow),
        },
        {
            "strategy": "repair",
            "checkpoints": [],
            "expected_cost": expected_cost_repair(workflow),
        },
        {
            "strategy": "never",
            "checkpoints": [],
            "expected_cost": expected_cost_never(workflow),
        },
        {
            "strategy": "every_step",
            "checkpoints": fixed_interval_checkpoints(workflow.n_steps, 1),
            "expected_cost": expected_cost_every_step(workflow),
        },
        {
            "strategy": "best_fixed",
            "checkpoints": fixed_interval_checkpoints(workflow.n_steps, interval),
            "interval": interval,
            "expected_cost": fixed_cost,
        },
        {
            "strategy": "optimal",
            "checkpoints": opt_points,
            "expected_cost": opt_cost,
        },
    ]


def recommend(workflow):
    """The cheapest strategy overall, with the full comparison table."""
    table = strategy_table(workflow)
    best = min(table, key=lambda row: row["expected_cost"])
    return {"table": table, "best": best}
