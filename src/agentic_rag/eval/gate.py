"""Compare an eval report's aggregate metrics against committed thresholds.

Pure logic over the report dict ``run_eval`` produces (``aggregates[system][metric]``)
plus a thresholds config, so it unit-tests offline with no API or Qdrant. The CI
gate (``scripts/eval_gate.py``) uses this to FAIL the build when the shipped system
regresses below a committed floor — turning answer quality into a versioned,
enforced signal instead of a vibe.
"""

from __future__ import annotations

# Human labels for the metric keys (mirrors runner.METRIC_KEYS).
_LABELS = {
    "recall_at_k": "Recall@k",
    "mrr": "MRR",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer rel.",
    "context_precision": "Ctx prec.",
    "context_recall": "Ctx recall",
    "judge_overall_norm": "Judge",
}


def check_thresholds(report: dict, thresholds: dict, system: str | None = None):
    """Compare ``report``'s aggregates for ``system`` to ``thresholds``.

    Returns ``(passed, rows)`` where each row is
    ``{metric, label, value, floor, status}`` (status ∈ pass|fail|skip) plus a
    trailing error-budget row. A metric that's absent or ``None`` in the report
    (e.g. the judge on a retrieval-only run) is **skipped**, not failed — so the
    same thresholds work for the free retrieval-only fallback.
    """
    system = system or thresholds.get("system", "agent")
    agg = (report.get("aggregates") or {}).get(system, {}) or {}
    rows = []
    passed = True

    for metric, floor in (thresholds.get("metrics") or {}).items():
        value = agg.get(metric)
        if value is None:
            status = "skip"
        elif value + 1e-9 >= floor:
            status = "pass"
        else:
            status = "fail"
            passed = False
        rows.append(
            {
                "metric": metric,
                "label": _LABELS.get(metric, metric),
                "value": value,
                "floor": floor,
                "status": status,
            }
        )

    # Error budget: any system error on a gated question is a hard fail by default.
    max_errors = thresholds.get("max_errors", 0)
    n_errors = agg.get("n_errors", 0)
    err_ok = n_errors <= max_errors
    if not err_ok:
        passed = False
    rows.append(
        {
            "metric": "n_errors",
            "label": "Errors",
            "value": n_errors,
            "floor": max_errors,
            "status": "pass" if err_ok else "fail",
            "is_errors": True,
        }
    )
    return passed, rows


def render_gate_summary(
    rows: list[dict], passed: bool, system: str = "agent", subset: str = ""
) -> str:
    """Render the gate result as a Markdown table (for the CI step summary / PR comment)."""
    icon = {"pass": "✅", "fail": "❌", "skip": "➖"}
    where = f" on the **{subset}** subset" if subset else ""
    lines = [
        f"## Eval gate: {'✅ PASS' if passed else '❌ FAIL'}",
        f"System **{system}**{where} — metrics vs committed floors (`eval/thresholds.json`).",
        "",
        "| Metric | Value | Floor | |",
        "|---|---|---|---|",
    ]
    for r in rows:
        if r.get("is_errors"):
            value_txt, floor_txt = str(r["value"]), f"≤ {r['floor']}"
        else:
            value_txt = "n/a" if r["value"] is None else f"{r['value']:.3f}"
            floor_txt = f"≥ {r['floor']:.2f}"
        lines.append(f"| {r['label']} | {value_txt} | {floor_txt} | {icon.get(r['status'], '')} |")
    return "\n".join(lines)
