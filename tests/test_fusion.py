"""Tests for Reciprocal Rank Fusion (pure logic)."""

from __future__ import annotations

from agentic_rag.retrieve.fusion import reciprocal_rank_fusion


def test_doc_high_in_both_lists_wins():
    # "b" is 2nd in both lists; "a" is 1st in one but absent from the other.
    dense = ["a", "b", "c"]
    bm25 = ["d", "b", "e"]
    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    order = [doc for doc, _ in fused]
    # b appears in both -> accumulates from both -> beats single-list leaders.
    assert order[0] == "b"


def test_rrf_score_formula():
    fused = dict(reciprocal_rank_fusion([["x", "y"]], k=60))
    assert abs(fused["x"] - 1.0 / 61) < 1e-9  # rank 1 -> 1/(60+1)
    assert abs(fused["y"] - 1.0 / 62) < 1e-9  # rank 2 -> 1/(60+2)


def test_scores_sum_across_lists():
    fused = dict(reciprocal_rank_fusion([["x"], ["x"]], k=60))
    assert abs(fused["x"] - 2.0 / 61) < 1e-9  # rank 1 in both lists


def test_deterministic_tie_break_by_id():
    # Two docs each rank-1 in one list -> equal score -> sorted by id.
    fused = reciprocal_rank_fusion([["b"], ["a"]], k=60)
    assert [doc for doc, _ in fused] == ["a", "b"]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
