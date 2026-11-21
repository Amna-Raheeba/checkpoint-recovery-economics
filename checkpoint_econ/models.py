"""Core data models for checkpoint-recovery-economics.

A Workflow is a sequence of steps. Each step has an execution cost and a
fault hazard (the probability that executing the step injects a fault into
the workflow state). A fault injects a state error with a severity drawn
from a declared range. A fraction of faults (the persistence) corrupt a
downstream dependency and never self-repair; the rest decay geometrically
while the run continues, each further step repairing a fixed fraction of
the outstanding error (the repair rate).

All money-like quantities (step costs, checkpoint cost, failure penalty)
share one unit -- read it as dollars, tokens, or seconds; the economics
only depend on the ratios.
"""

from dataclasses import dataclass, field


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class Workflow:
    """A step sequence with per-step costs and fault hazards."""

    name: str
    step_costs: list
    step_hazards: list
    checkpoint_cost: float = 3.0
    penalty: float = 600.0
    repair_rate: float = 0.25
    persistence: float = 0.3
    error_threshold: float = 0.05
    severity_lo: float = 0.4
    severity_hi: float = 1.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("workflow name must be a non-empty string")
        if not self.step_costs:
            raise ValueError("a workflow needs at least one step")
        if len(self.step_costs) != len(self.step_hazards):
            raise ValueError("step_costs and step_hazards must have equal length")
        for cost in self.step_costs:
            if not _is_number(cost) or cost <= 0:
                raise ValueError("every step cost must be a positive number")
        for hazard in self.step_hazards:
            if not _is_number(hazard) or hazard < 0.0 or hazard > 1.0:
                raise ValueError("every step hazard must lie in [0, 1]")
        if not _is_number(self.checkpoint_cost) or self.checkpoint_cost < 0:
            raise ValueError("checkpoint_cost must be a non-negative number")
        if not _is_number(self.penalty) or self.penalty < 0:
            raise ValueError("penalty must be a non-negative number")
        if not _is_number(self.repair_rate) or not 0.0 <= self.repair_rate <= 1.0:
            raise ValueError("repair_rate must lie in [0, 1]")
        if not _is_number(self.persistence) or not 0.0 <= self.persistence <= 1.0:
            raise ValueError("persistence must lie in [0, 1]")
        if not _is_number(self.error_threshold) or self.error_threshold <= 0:
            raise ValueError("error_threshold must be positive")
        if not _is_number(self.severity_lo) or not _is_number(self.severity_hi):
            raise ValueError("severity bounds must be numbers")
        if self.severity_lo < 0 or self.severity_hi < self.severity_lo:
            raise ValueError("severity range must satisfy 0 <= lo <= hi")
        self.step_costs = [float(c) for c in self.step_costs]
        self.step_hazards = [float(h) for h in self.step_hazards]

    @property
    def n_steps(self):
        return len(self.step_costs)

    @property
    def total_cost(self):
        return sum(self.step_costs)

    @property
    def mean_step_cost(self):
        return self.total_cost / self.n_steps

    def with_hazards(self, hazards, name=None):
        """Copy of this workflow with a different hazard profile."""
        return Workflow(
            name=name or self.name,
            step_costs=list(self.step_costs),
            step_hazards=list(hazards),
            checkpoint_cost=self.checkpoint_cost,
            penalty=self.penalty,
            repair_rate=self.repair_rate,
            persistence=self.persistence,
            error_threshold=self.error_threshold,
            severity_lo=self.severity_lo,
            severity_hi=self.severity_hi,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "step_costs": list(self.step_costs),
            "step_hazards": list(self.step_hazards),
            "checkpoint_cost": self.checkpoint_cost,
            "penalty": self.penalty,
            "repair_rate": self.repair_rate,
            "persistence": self.persistence,
            "error_threshold": self.error_threshold,
            "severity_lo": self.severity_lo,
            "severity_hi": self.severity_hi,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            step_costs=list(data["step_costs"]),
            step_hazards=list(data["step_hazards"]),
            checkpoint_cost=data.get("checkpoint_cost", 3.0),
            penalty=data.get("penalty", 600.0),
            repair_rate=data.get("repair_rate", 0.25),
            persistence=data.get("persistence", 0.3),
            error_threshold=data.get("error_threshold", 0.05),
            severity_lo=data.get("severity_lo", 0.4),
            severity_hi=data.get("severity_hi", 1.0),
        )


@dataclass
class Trace:
    """One recorded run: where checkpoints were taken, where faults hit.

    Steps are 1-based. ``faults`` is a list of dicts with keys ``step``
    and ``severity`` -- the faults observed on the first pass through the
    workflow. ``recovery`` is the policy the run actually used:
    "rollback", "restart", or "repair".
    """

    workflow: Workflow
    checkpoints_taken: list = field(default_factory=list)
    faults: list = field(default_factory=list)
    recovery: str = "rollback"

    def __post_init__(self):
        n = self.workflow.n_steps
        for position in self.checkpoints_taken:
            if not isinstance(position, int) or isinstance(position, bool):
                raise ValueError("checkpoint positions must be integers")
            if position < 1 or position > n:
                raise ValueError("checkpoint position out of range: %r" % (position,))
        if len(set(self.checkpoints_taken)) != len(self.checkpoints_taken):
            raise ValueError("checkpoint positions must be unique")
        self.checkpoints_taken = sorted(self.checkpoints_taken)
        for fault in self.faults:
            if "step" not in fault or "severity" not in fault:
                raise ValueError("each fault needs 'step' and 'severity'")
            step = fault["step"]
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError("fault step must be an integer")
            if step < 1 or step > n:
                raise ValueError("fault step out of range: %r" % (step,))
            severity = fault["severity"]
            if not _is_number(severity) or severity < 0:
                raise ValueError("fault severity must be a non-negative number")
        if self.recovery not in ("rollback", "restart", "repair"):
            raise ValueError("recovery must be rollback, restart, or repair")

    def to_dict(self):
        return {
            "workflow": self.workflow.to_dict(),
            "checkpoints_taken": list(self.checkpoints_taken),
            "faults": [dict(f) for f in self.faults],
            "recovery": self.recovery,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            workflow=Workflow.from_dict(data["workflow"]),
            checkpoints_taken=list(data.get("checkpoints_taken", [])),
            faults=[dict(f) for f in data.get("faults", [])],
            recovery=data.get("recovery", "rollback"),
        )
