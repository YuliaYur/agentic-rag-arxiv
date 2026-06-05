"""Per-call cost / latency / cache metering.

Every ``LLMClient.structured`` call appends a ``CallRecord`` to whatever metering
scope is active (a ``contextvar``, so it's thread/async-safe and a no-op when no
scope is open). ``run_agent`` opens a scope around one agent run to get
cost-per-query; the before/after harness opens one per config. Pure data — no API.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

_records: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "llm_call_records", default=None
)


@dataclass
class CallRecord:
    role: str | None
    model: str
    cost_usd: float
    latency_ms: float
    cached: bool
    input_tokens: int = 0
    output_tokens: int = 0


def record_call(rec: CallRecord) -> None:
    """Append a record to the active metering scope (no-op if none is open)."""
    sink = _records.get()
    if sink is not None:
        sink.append(rec)


@contextmanager
def meter() -> Iterator[list[CallRecord]]:
    """Collect the ``CallRecord``s made inside this block."""
    sink: list[CallRecord] = []
    token = _records.set(sink)
    try:
        yield sink
    finally:
        _records.reset(token)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated p-th percentile of ``values`` (p in [0,100])."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def summarize(records: list[CallRecord]) -> dict:
    """Aggregate a batch of call records: cost, latency percentiles, cache hit-rate."""
    n = len(records)
    if n == 0:
        return {
            "n_calls": 0,
            "cost_usd": 0.0,
            "latency_ms_total": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "cache_hits": 0,
            "hit_rate": 0.0,
        }
    latencies = [r.latency_ms for r in records]
    hits = sum(1 for r in records if r.cached)
    return {
        "n_calls": n,
        "cost_usd": sum(r.cost_usd for r in records),
        "latency_ms_total": sum(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "cache_hits": hits,
        "hit_rate": hits / n,
    }
