"""Generate the stored case data.

Writes:
- data/workflows.json: the workflow of every evaluation scenario, using
  the same seed scheme as checkpoint_econ.evaluate, so the stored
  workflows are exactly the ones the headline evaluation prices.
- examples/demo_workflow.json and examples/demo_trace.json: the demo
  workflow and recorded trace used by the CLI demo and the examples.

Run: python scripts/build_cases.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from checkpoint_econ.cli import DEMO_TRACE, DEMO_WORKFLOW
from checkpoint_econ.evaluate import DEFAULT_CONFIG, build_workflow


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = DEFAULT_CONFIG
    scenarios = []
    index = 0
    for length in cfg["lengths"]:
        for profile in cfg["profiles"]:
            seed = cfg["base_seed"] + 1000 * index
            workflow = build_workflow(profile, length, cfg, seed)
            scenarios.append({"scenario": workflow.name,
                              "workflow": workflow.to_dict()})
            index += 1
    data_path = os.path.join(root, "data", "workflows.json")
    with open(data_path, "w") as handle:
        json.dump({"scenarios": scenarios}, handle, indent=2)
    print("Wrote %s (%d scenarios)" % (data_path, len(scenarios)))

    workflow_path = os.path.join(root, "examples", "demo_workflow.json")
    with open(workflow_path, "w") as handle:
        json.dump(DEMO_WORKFLOW.to_dict(), handle, indent=2)
    print("Wrote %s" % workflow_path)

    trace_path = os.path.join(root, "examples", "demo_trace.json")
    with open(trace_path, "w") as handle:
        json.dump(DEMO_TRACE.to_dict(), handle, indent=2)
    print("Wrote %s" % trace_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
