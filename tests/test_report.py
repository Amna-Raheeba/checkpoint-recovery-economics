from checkpoint_econ.audit import audit_trace
from checkpoint_econ.cli import DEMO_TRACE, DEMO_WORKFLOW
from checkpoint_econ.optimize import recommend
from checkpoint_econ.report import audit_report_markdown, optimize_report_markdown


def test_optimize_report_contents():
    text = optimize_report_markdown(DEMO_WORKFLOW, recommend(DEMO_WORKFLOW))
    assert "# Checkpoint plan for demo-research-run" in text
    assert "Expected cost by strategy" in text
    assert "optimal" in text
    assert "Recommendation" in text


def test_audit_report_contents():
    text = audit_report_markdown(DEMO_TRACE, audit_trace(DEMO_TRACE))
    assert "# Recovery audit" in text
    assert "What the run spent" in text
    assert "Checkpoint use" in text
    assert "Counterfactual" in text
    assert "Recommendation" in text


def test_optimize_report_shows_all_strategies():
    text = optimize_report_markdown(DEMO_WORKFLOW, recommend(DEMO_WORKFLOW))
    for strategy in ("restart", "repair", "never", "every_step",
                     "best_fixed", "optimal"):
        assert strategy in text
