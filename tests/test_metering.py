"""Tests for LLM call metering (cost / latency / cache aggregation, no API)."""

from __future__ import annotations

import pytest

from agentic_rag.llm.metering import CallRecord, meter, percentile, record_call, summarize


def _rec(cost=0.001, lat=100.0, cached=False):
    return CallRecord(role="grade", model="m", cost_usd=cost, latency_ms=lat, cached=cached)


def test_meter_collects_records_in_scope():
    with meter() as sink:
        record_call(_rec())
        record_call(_rec(cached=True))
    assert len(sink) == 2


def test_record_call_is_noop_without_scope():
    record_call(_rec())  # no active meter() -> silently dropped, no error


def test_summarize_cost_and_hit_rate():
    s = summarize([_rec(cost=0.001, cached=False), _rec(cost=0.0, cached=True)])
    assert s["n_calls"] == 2
    assert s["cost_usd"] == pytest.approx(0.001)
    assert s["cache_hits"] == 1
    assert s["hit_rate"] == 0.5


def test_summarize_percentiles():
    s = summarize([_rec(lat=val) for val in (100, 200, 300, 400)])
    assert s["p50_ms"] == pytest.approx(250.0)
    assert s["p95_ms"] == pytest.approx(385.0)
    assert s["latency_ms_total"] == pytest.approx(1000.0)


def test_summarize_empty():
    s = summarize([])
    assert s["n_calls"] == 0 and s["cost_usd"] == 0.0 and s["hit_rate"] == 0.0


def test_percentile_helpers():
    assert percentile([10, 20, 30, 40], 50) == pytest.approx(25.0)
    assert percentile([], 95) == 0.0
    assert percentile([5], 95) == 5.0
