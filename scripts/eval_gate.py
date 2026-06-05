"""Fail the build if an eval report regresses below committed thresholds.

    python scripts/eval_gate.py eval/results/latest.json eval/thresholds.json \
        [--system agent] [--summary "$GITHUB_STEP_SUMMARY"]

Prints a pass/fail table, optionally appends it to a summary file (the GitHub
Actions step-summary), and exits non-zero when any gated metric is below its floor
(or the error budget is exceeded). That non-zero exit is what blocks the PR.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None) -> int:
    _utf8()
    p = argparse.ArgumentParser(description="Gate a PR on eval metrics vs committed thresholds.")
    p.add_argument("results", help="eval results JSON (e.g. eval/results/latest.json)")
    p.add_argument("thresholds", help="thresholds JSON (e.g. eval/thresholds.json)")
    p.add_argument(
        "--system", default=None, help="which system to gate (default: thresholds.system)"
    )
    p.add_argument("--summary", default=None, help="append the Markdown table to this file")
    args = p.parse_args(argv)

    from agentic_rag.eval.gate import check_thresholds, render_gate_summary

    report = json.loads(Path(args.results).read_text(encoding="utf-8"))
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    system = args.system or thresholds.get("system", "agent")

    passed, rows = check_thresholds(report, thresholds, system)
    md = render_gate_summary(rows, passed, system=system, subset=thresholds.get("subset", ""))
    print(md)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    if not passed:
        print("\nEval gate FAILED — a metric dropped below its committed floor.", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
