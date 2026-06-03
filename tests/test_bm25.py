"""Tests for in-memory BM25 search (the keyword half of hybrid)."""

from __future__ import annotations

from agentic_rag.retrieve.bm25 import BM25Index, tokenize


def test_tokenize_keeps_alnum_terms_lowercased():
    assert tokenize("RoBERTa uses the GLUE benchmark.") == [
        "roberta",
        "uses",
        "the",
        "glue",
        "benchmark",
    ]
    assert tokenize("O(n^2)") == ["o", "n", "2"]


def _index():
    ids = ["d1", "d2", "d3"]
    texts = [
        "the transformer uses multi head self attention",
        "roberta is evaluated on the glue benchmark",
        "vision transformers split an image into patches",
    ]
    return BM25Index(ids, texts)


def test_exact_term_retrieves_right_doc():
    # "glue" appears only in d2 -> d2 must rank first.
    hits = _index().search("glue benchmark", limit=3)
    assert hits[0][0] == "d2"


def test_only_positive_scores_returned():
    # A query with no shared terms returns nothing (no score > 0).
    hits = _index().search("quantum chromodynamics", limit=3)
    assert hits == []


def test_empty_query_returns_nothing():
    assert _index().search("", limit=5) == []


def test_limit_is_respected():
    hits = _index().search("transformer", limit=1)
    assert len(hits) <= 1
