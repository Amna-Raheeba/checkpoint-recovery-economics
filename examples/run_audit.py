"""Run the audit on the bundled demo trace and print the report.

Run: python examples/run_audit.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from checkpoint_econ.audit import audit_trace
from checkpoint_econ.models import Trace
from checkpoint_econ.report import audit_report_markdown


def main():
    path = os.path.join(os.path.dirname(__file__), "demo_trace.json")
    with open(path) as handle:
        trace = Trace.from_dict(json.load(handle))
    print(audit_report_markdown(trace, audit_trace(trace)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
