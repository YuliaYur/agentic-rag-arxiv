"""Run every system over the golden set and compute all metrics per question.

Pure orchestration over injected pieces (systems, a metric LLM): no API or Qdrant
needed to unit-test it with fakes. Produces a plain-dict report that ``report.py``
renders and that we serialize to ``eval/results/``.
"""

from __future__ import annotations

from statistics import mean

from .dataset import GoldenItem
from .judge import judge_answer, normalized_overall
from .ragas_metrics import compute_ragas_metrics
from .retrieval_metrics import mrr, recall_at_k

# Metrics that make up the per-system aggregate table, in display order.
METRIC_KEYS = [
    "recall_at_k",
    "mrr",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "judge_overall_norm",
]


def evaluate_item(
    item: GoldenItem,
    systems: dict,
    metric_llm,
    k: int = 5,
    use_ragas: bool = True,
    use_judge: bool = True,
) -> dict:
    """Run each system on one question and score it. Errors are captured per system."""
    out: dict = {
        "id": item.id,
        "question": item.question,
        "type": item.type,
        "status": item.status,
        "expected_arxiv_ids": item.expected_arxiv_ids,
        "systems": {},
    }
    for name, system in systems.items():
        try:
            res = system.run(item.question)
            scores = {
                "recall_at_k": recall_at_k(res.retrieved_arxiv_ids, item.expected_arxiv_ids, k),
                "mrr": mrr(res.retrieved_arxiv_ids, item.expected_arxiv_ids),
            }
            if use_ragas:
                scores.update(
                    compute_ragas_metrics(
                        metric_llm, item.question, res.answer, res.contexts, item.reference_answer
                    )
                )
            entry = {
                "answer": res.answer,
                "retrieved_arxiv_ids": res.retrieved_arxiv_ids,
                "cited_arxiv_ids": res.cited_arxiv_ids,
                "scores": scores,
                "extra": res.extra,
            }
            if use_judge:
                verdict = judge_answer(metric_llm, item.question, res.answer, item.reference_answer)
                scores["judge_overall_norm"] = normalized_overall(verdict)
                entry["judge"] = verdict.model_dump()
            out["systems"][name] = entry
        except Exception as exc:  # one bad question shouldn't sink the whole run
            out["systems"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def aggregate(per_question: list[dict], system_names: list[str]) -> dict:
    """Mean of each metric per system across all questions (skipping errors)."""
    agg: dict = {}
    for name in system_names:
        agg[name] = {}
        for key in METRIC_KEYS:
            vals = [
                q["systems"][name]["scores"][key]
                for q in per_question
                if name in q["systems"]
                and "scores" in q["systems"][name]
                and key in q["systems"][name]["scores"]
            ]
            agg[name][key] = mean(vals) if vals else None
        errs = sum(1 for q in per_question if q["systems"].get(name, {}).get("error"))
        agg[name]["n_errors"] = errs
    return agg


def run_eval(
    items: list[GoldenItem],
    systems: dict,
    metric_llm,
    k: int = 5,
    use_ragas: bool = True,
    use_judge: bool = True,
    progress=None,
) -> dict:
    """Evaluate all items against all systems; return a serializable report dict."""
    per_question = []
    for i, item in enumerate(items, start=1):
        if progress:
            progress(i, len(items), item.id)
        per_question.append(evaluate_item(item, systems, metric_llm, k, use_ragas, use_judge))
    names = list(systems.keys())
    return {
        "config": {"k": k, "use_ragas": use_ragas, "use_judge": use_judge, "n_items": len(items)},
        "systems": names,
        "per_question": per_question,
        "aggregates": aggregate(per_question, names),
    }
