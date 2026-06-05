"""Tests for the CI eval gate (pure threshold comparison, no API/Qdrant)."""

from __future__ import annotations

from agentic_rag.eval.gate import check_thresholds, render_gate_summary

THRESHOLDS = {
    "system": "agent",
    "subset": "seed",
    "max_errors": 0,
    "metrics": {"recall_at_k": 0.90, "faithfulness": 0.60, "judge_overall_norm": 0.80},
}


def _report(agg: dict) -> dict:
    return {"aggregates": {"agent": agg}}


def test_all_metrics_above_floor_passes():
    passed, rows = check_thresholds(
        _report(
            {"recall_at_k": 1.0, "faithfulness": 0.76, "judge_overall_norm": 0.95, "n_errors": 0}
        ),
        THRESHOLDS,
    )
    assert passed
    assert all(r["status"] in {"pass"} for r in rows)


def test_one_metric_below_floor_fails():
    passed, rows = check_thresholds(
        _report(
            {"recall_at_k": 0.50, "faithfulness": 0.76, "judge_overall_norm": 0.95, "n_errors": 0}
        ),
        THRESHOLDS,
    )
    assert not passed
    recall = next(r for r in rows if r["metric"] == "recall_at_k")
    assert recall["status"] == "fail"


def test_none_metric_is_skipped_not_failed():
    # Retrieval-only run: faithfulness/judge are None -> skipped, gate still passes.
    passed, rows = check_thresholds(
        _report(
            {"recall_at_k": 1.0, "faithfulness": None, "judge_overall_norm": None, "n_errors": 0}
        ),
        THRESHOLDS,
    )
    assert passed
    assert {r["metric"]: r["status"] for r in rows}["faithfulness"] == "skip"


def test_error_budget_exceeded_fails():
    passed, _ = check_thresholds(
        _report(
            {"recall_at_k": 1.0, "faithfulness": 0.76, "judge_overall_norm": 0.95, "n_errors": 1}
        ),
        THRESHOLDS,
    )
    assert not passed


def test_value_exactly_at_floor_passes():
    passed, _ = check_thresholds(
        _report(
            {"recall_at_k": 0.90, "faithfulness": 0.60, "judge_overall_norm": 0.80, "n_errors": 0}
        ),
        THRESHOLDS,
    )
    assert passed


def test_render_summary_reflects_status():
    passed, rows = check_thresholds(
        _report(
            {"recall_at_k": 0.5, "faithfulness": 0.76, "judge_overall_norm": 0.95, "n_errors": 0}
        ),
        THRESHOLDS,
    )
    md = render_gate_summary(rows, passed, system="agent", subset="seed")
    assert "❌ FAIL" in md
    assert "Recall@k" in md and "seed" in md
