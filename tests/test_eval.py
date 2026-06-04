"""Offline tests for the eval harness: dataset, retrieval metrics, RAGAS-style
metrics, judge, runner, and report. A scripted FakeLLM stands in for the API."""

from __future__ import annotations

import pytest

from agentic_rag.eval import report as report_mod
from agentic_rag.eval.dataset import load_golden_set
from agentic_rag.eval.judge import JudgeVerdict, judge_answer, normalized_overall
from agentic_rag.eval.ragas_metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from agentic_rag.eval.retrieval_metrics import mrr, recall_at_k
from agentic_rag.eval.runner import run_eval
from agentic_rag.eval.systems import SystemResult

# --- dataset ----------------------------------------------------------------


def test_golden_set_loads_and_is_valid():
    items = load_golden_set()
    assert len(items) >= 25  # the curated/draft mix
    assert all(it.id and it.question and it.reference_answer for it in items)
    assert all(it.type in {"factual", "comparative", "multi-hop"} for it in items)
    assert all(it.expected_arxiv_ids for it in items)
    # a real mix of single-hop and cross-paper questions
    assert any(not it.is_multihop for it in items)
    assert any(it.is_multihop for it in items)


def test_dataset_rejects_missing_field(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "question": "q?", "type": "factual"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing/empty required field"):
        load_golden_set(bad)


def test_dataset_rejects_bad_type(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"id":"x","question":"q?","type":"weird","expected_arxiv_ids":["1"],'
        '"reference_answer":"a"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="type must be one of"):
        load_golden_set(bad)


# --- retrieval metrics ------------------------------------------------------


def test_recall_at_k():
    assert recall_at_k(["A", "B", "C"], ["A", "B"], k=3) == 1.0
    assert recall_at_k(["A", "X", "Y"], ["A", "B"], k=3) == 0.5  # only 1 of 2
    assert recall_at_k(["X", "Y", "A"], ["A"], k=2) == 0.0  # A is below k


def test_mrr():
    assert mrr(["A", "B"], ["B"]) == 0.5  # first relevant at rank 2
    assert mrr(["A", "B"], ["A"]) == 1.0
    assert mrr(["X", "Y"], ["A"]) == 0.0


# --- scripted fake LLM ------------------------------------------------------


class FakeLLM:
    """Returns a scripted schema instance per schema name."""

    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def structured(self, system, user, schema):
        self.calls.append(schema.__name__)
        return schema(**self._responses[schema.__name__])


def _metric_llm():
    return FakeLLM(
        {
            "_Statements": {"statements": ["s1", "s2"]},
            "_Verdicts": {
                "verdicts": [
                    {"statement": "s1", "supported": True},
                    {"statement": "s2", "supported": False},
                ]
            },
            "_Relevances": {"relevant": [True, False, True]},
            "_GenQuestions": {"questions": ["g1", "g2"], "noncommittal": False},
            "_Similarities": {"scores": [1.0, 0.5]},
            "JudgeVerdict": {
                "correctness": 4,
                "completeness": 4,
                "relevance": 5,
                "overall": 4,
                "rationale": "good",
            },
        }
    )


# --- RAGAS-style metrics ----------------------------------------------------


def test_faithfulness_supported_fraction():
    # 2 statements, 1 supported -> 0.5
    assert faithfulness(_metric_llm(), "ans", ["ctx"]) == 0.5


def test_faithfulness_none_without_context():
    assert faithfulness(_metric_llm(), "ans", []) is None


def test_faithfulness_none_for_refusal_without_calling_llm():
    # A refusal makes no factual claim -> not applicable; must short-circuit (no LLM).
    class Boom:
        def structured(self, *a, **k):
            raise AssertionError("LLM should not be called for a refusal")

    val = faithfulness(Boom(), "I don't have enough information in the provided sources.", ["ctx"])
    assert val is None


def test_aggregate_skips_none_metric():
    from agentic_rag.eval.runner import aggregate

    pq = [
        {"systems": {"baseline": {"scores": {"faithfulness": None, "mrr": 1.0}}}},
        {"systems": {"baseline": {"scores": {"faithfulness": 0.5, "mrr": 0.5}}}},
    ]
    agg = aggregate(pq, ["baseline"])
    assert agg["baseline"]["faithfulness"] == 0.5  # the None row is skipped
    assert agg["baseline"]["mrr"] == 0.75


def test_context_recall_supported_fraction():
    assert context_recall(_metric_llm(), "reference", ["ctx"]) == 0.5


def test_context_precision_rank_weighted():
    # relevant=[T, F, T] over 3 contexts -> AP = (1/1 + 2/3) / 2 = 0.8333...
    val = context_precision(_metric_llm(), "q?", "ref", ["c1", "c2", "c3"])
    assert round(val, 4) == 0.8333


def test_answer_relevancy_mean_similarity():
    assert answer_relevancy(_metric_llm(), "q?", "ans") == 0.75  # mean(1.0, 0.5)


def test_answer_relevancy_noncommittal_is_zero():
    llm = FakeLLM({"_GenQuestions": {"questions": [], "noncommittal": True}})
    assert answer_relevancy(llm, "q?", "I don't know") == 0.0


# --- judge ------------------------------------------------------------------


def test_judge_and_normalization():
    verdict = judge_answer(_metric_llm(), "q?", "ans", "ref")
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.overall == 4
    assert normalized_overall(verdict) == 0.75  # (4-1)/4


# --- runner + report --------------------------------------------------------


class FakeSystem:
    def __init__(self, arxiv_ids):
        self._ids = arxiv_ids

    def run(self, question):
        return SystemResult(
            answer="some grounded answer [S1].",
            contexts=["chunk text about transformers"],
            retrieved_arxiv_ids=self._ids,
            cited_arxiv_ids=self._ids[:1],
        )


def test_run_eval_aggregates_and_report():
    items = load_golden_set()[:2]
    systems = {
        "baseline": FakeSystem(["X"]),  # retrieves wrong paper
        "agent": FakeSystem(items[0].expected_arxiv_ids),  # retrieves the right one
    }
    rep = run_eval(items, systems, _metric_llm(), k=5, use_ragas=True, use_judge=True)
    assert rep["systems"] == ["baseline", "agent"]
    assert len(rep["per_question"]) == 2
    agg = rep["aggregates"]
    # agent retrieved an expected id for item[0]; baseline retrieved none -> better recall
    assert agg["agent"]["recall_at_k"] >= agg["baseline"]["recall_at_k"]
    for name in ("baseline", "agent"):
        assert agg[name]["faithfulness"] == 0.5
        assert agg[name]["n_errors"] == 0
    # report renders without error and shows both systems
    md = report_mod.render_markdown(rep)
    assert "baseline" in md and "agent" in md and "Faithfulness" in md


def test_runner_captures_system_errors():
    class Boom:
        def run(self, q):
            raise RuntimeError("kaboom")

    items = load_golden_set()[:1]
    rep = run_eval(items, {"baseline": Boom()}, _metric_llm(), use_ragas=False, use_judge=False)
    assert rep["aggregates"]["baseline"]["n_errors"] == 1
    assert "kaboom" in rep["per_question"][0]["systems"]["baseline"]["error"]
