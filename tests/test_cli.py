import json
import os

from checkpoint_econ.cli import main


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Checkpoint plan for demo-research-run" in out
    assert "Recovery audit" in out
    assert "half-life" in out


def test_cli_optimize(capsys):
    path = os.path.join(
        os.path.dirname(__file__), "..", "examples", "demo_workflow.json"
    )
    assert main(["optimize", "--workflow", path]) == 0
    out = capsys.readouterr().out
    assert "Checkpoint plan" in out


def test_cli_audit(capsys):
    path = os.path.join(
        os.path.dirname(__file__), "..", "examples", "demo_trace.json"
    )
    assert main(["audit", "--trace", path]) == 0
    out = capsys.readouterr().out
    assert "Recovery audit" in out


def test_cli_evaluate(tmp_path, capsys):
    assert main(["evaluate", "--outdir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Wrote results" in out
    for name in ("evaluation.json", "evaluation.csv",
                 "evaluation_summary.md", "HEADLINE.md"):
        assert os.path.exists(os.path.join(str(tmp_path), name))
    with open(os.path.join(str(tmp_path), "evaluation.json")) as handle:
        evaluation = json.load(handle)
    assert evaluation["aggregates"]["n_scenarios"] == 16


def test_cli_no_command(capsys):
    assert main([]) == 2
