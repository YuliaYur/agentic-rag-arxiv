"""Reciprocal Rank Fusion (RRF).

Why RRF (and not score-based fusion): dense cosine scores live in ~[0, 1] while
BM25 scores are unbounded and corpus-dependent. Normalizing them to combine is
fiddly and brittle. RRF sidesteps that entirely by fusing on **rank position**,
not score:

    rrf_score(d) = sum over each ranked list of  1 / (k + rank(d))

A document ranked highly by *both* retrievers accumulates score from both lists
and rises to the top — which is exactly the hybrid behavior we want. The
constant k (default 60, from the original paper) flattens the curve so the #1
slot doesn't completely dominate #2, #3, ...

Pure function, fully unit-tested.
"""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked id-lists into one ranking.

    Args:
        rankings: each element is a list of document ids in descending relevance.
        k: RRF damping constant.

    Returns:
        (doc_id, fused_score) pairs sorted by fused score descending. Ties are
        broken by id for deterministic output.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
