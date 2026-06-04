"""Evaluation harness: golden set + RAGAS-style metrics + LLM-judge, run over the
single-shot baseline and the agent graph, producing a comparison report.

See ADR-0011 for why the metrics are implemented natively rather than via the
``ragas`` package. The runner and metrics take injected LLM/systems so they're
unit-tested offline with fakes; the real run is driven by ``scripts/eval_run.py``.
"""

from .dataset import GoldenItem, load_golden_set
from .runner import run_eval
from .systems import AgentSystem, BaselineSystem, SystemResult

__all__ = [
    "GoldenItem",
    "load_golden_set",
    "run_eval",
    "BaselineSystem",
    "AgentSystem",
    "SystemResult",
]
