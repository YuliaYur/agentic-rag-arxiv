"""Retrieval-quality metrics over ``expected_arxiv_ids`` — no LLM, fully offline.

These score the *retriever* (did the right paper(s) come back?), independent of the
generated answer. Cheap and deterministic, so they run on every question.
"""

from __future__ import annotations


def recall_at_k(retrieved_arxiv_ids: list[str], expected_arxiv_ids: list[str], k: int) -> float:
    """Fraction of expected papers that appear in the top-k retrieved papers.

    1.0 means every paper the question needs was retrieved within k. For a
    multi-hop question expecting 2 papers, retrieving only 1 scores 0.5.
    """
    if not expected_arxiv_ids:
        return 0.0
    top = set(retrieved_arxiv_ids[:k])
    hit = sum(1 for e in set(expected_arxiv_ids) if e in top)
    return hit / len(set(expected_arxiv_ids))


def mrr(retrieved_arxiv_ids: list[str], expected_arxiv_ids: list[str]) -> float:
    """Reciprocal rank of the FIRST expected paper (1/rank); 0 if none retrieved.

    Rewards getting a relevant paper high in the list: rank 1 -> 1.0, rank 2 -> 0.5.
    """
    expected = set(expected_arxiv_ids)
    for rank, aid in enumerate(retrieved_arxiv_ids, start=1):
        if aid in expected:
            return 1.0 / rank
    return 0.0
