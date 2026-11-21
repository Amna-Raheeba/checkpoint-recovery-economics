"""Markdown reports for optimisation and audit results."""


def optimize_report_markdown(workflow, recommendation):
    lines = []
    lines.append("# Checkpoint plan for %s" % workflow.name)
    lines.append("")
    lines.append(
        "%d steps, base cost %.2f, checkpoint cost %.2f, failure penalty "
        "%.2f, repair rate %.2f, error threshold %.2f."
        % (
            workflow.n_steps,
            workflow.total_cost,
            workflow.checkpoint_cost,
            workflow.penalty,
            workflow.repair_rate,
            workflow.error_threshold,
        )
    )
    lines.append("")
    lines.append("## Expected cost by strategy")
    lines.append("")
    lines.append("| strategy | checkpoints | expected cost |")
    lines.append("|---|---|---|")
    for row in recommendation["table"]:
        points = row["checkpoints"]
        label = ", ".join(str(p) for p in points) if points else "-"
        if row["strategy"] == "best_fixed":
            label = "every %d steps (%s)" % (row["interval"], label)
        lines.append(
            "| %s | %s | %.2f |" % (row["strategy"], label, row["expected_cost"])
        )
    lines.append("")
    best = recommendation["best"]
    lines.append(
        "**Recommendation: %s** at expected cost %.2f (checkpoints: %s)."
        % (
            best["strategy"],
            best["expected_cost"],
            best["checkpoints"] if best["checkpoints"] else "none",
        )
    )
    lines.append("")
    return "\n".join(lines)


def audit_report_markdown(trace, result):
    lines = []
    lines.append("# Recovery audit for one recorded run")
    lines.append("")
    lines.append(
        "Workflow %s (%d steps), recovery policy used: %s."
        % (trace.workflow.name, trace.workflow.n_steps, trace.recovery)
    )
    lines.append("")
    lines.append("## What the run spent")
    lines.append("")
    lines.append("| component | cost |")
    lines.append("|---|---|")
    lines.append("| base (one clean pass) | %.2f |" % result.base_cost)
    lines.append("| checkpoint spend | %.2f |" % result.checkpoint_spend)
    lines.append("| rework | %.2f |" % result.rework)
    lines.append("| penalty | %.2f |" % result.penalty_amount)
    lines.append("| **total** | **%.2f** |" % result.total_actual)
    lines.append("")
    lines.append("## Checkpoint use")
    lines.append("")
    lines.append(
        "Taken: %s. Used by a fault: %s. Never used: %s (wasted spend "
        "%.2f on this run)."
        % (
            result.checkpoints_taken if result.checkpoints_taken else "none",
            result.used_checkpoints if result.used_checkpoints else "none",
            result.unused_checkpoints if result.unused_checkpoints else "none",
            result.wasted_checkpoint_spend,
        )
    )
    lines.append("")
    lines.append("## Counterfactual (same faults, optimal placement)")
    lines.append("")
    lines.append(
        "Optimal checkpoints %s would have spent %.2f on checkpoints and "
        "%.2f on rework, total %.2f. Excess of the recorded run: %.2f "
        "(spend difference %+.2f, rework difference %+.2f)."
        % (
            result.optimal_checkpoints,
            result.counterfactual_spend,
            result.counterfactual_rework,
            result.total_counterfactual,
            result.excess_cost,
            result.spend_difference,
            result.rework_difference,
        )
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(result.recommendation)
    lines.append("")
    return "\n".join(lines)
