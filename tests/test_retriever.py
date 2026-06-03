"""Tests for HybridRetriever orchestration, using fakes (no Qdrant, no model)."""

from __future__ import annotations

from agentic_rag.retrieve.bm25 import BM25Index
from agentic_rag.retrieve.config import RetrieveConfig
from agentic_rag.retrieve.models import RetrievedChunk
from agentic_rag.retrieve.retriever import HybridRetriever


def _chunk(i: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=i,
        text=text,
        arxiv_id=f"id-{i}",
        title=f"Title {i}",
        slug=i,
        section=f"Section {i}",
        page=1,
        page_end=1,
        chunk_index=int(i[-1]),
    )


CHUNKS = [
    _chunk("d1", "the transformer uses multi head self attention"),
    _chunk("d2", "roberta is evaluated on the glue benchmark"),
    _chunk("d3", "vision transformers split an image into patches"),
    _chunk("d4", "scaling laws relate model size to loss"),
]


class FakeDense:
    """Returns a preset ranked list, ignoring the query text."""

    def __init__(self, ranked_ids):
        self._ranked = ranked_ids

    def search(self, query, limit):
        return [(doc_id, 1.0 - i * 0.01) for i, doc_id in enumerate(self._ranked)][:limit]


class FakeReranker:
    """Scores by a fixed id->score lookup so we can assert reordering."""

    def __init__(self, scores):
        self._scores = scores
        self.called_with = None

    def score(self, query, texts):
        self.called_with = (query, texts)
        # Map each text back to its chunk score via the corpus.
        return [self._scores[t] for t in texts]


def _bm25():
    return BM25Index([c.id for c in CHUNKS], [c.text for c in CHUNKS])


def test_metadata_intact_in_results():
    r = HybridRetriever(CHUNKS, FakeDense(["d1", "d2"]), _bm25(), reranker=None)
    hits = r.retrieve("self attention", k=2)
    top = hits[0]
    assert top.title.startswith("Title")
    assert top.arxiv_id.startswith("id-")
    assert top.section.startswith("Section")
    assert "§" in top.citation()


def test_fusion_promotes_doc_strong_in_both():
    # Dense favors d1; BM25 (via the query "glue benchmark") favors d2.
    # d2 is rank-1 in BM25 and present in dense -> should win the fusion.
    r = HybridRetriever(
        CHUNKS,
        FakeDense(["d2", "d1", "d3"]),
        _bm25(),
        reranker=None,
        config=RetrieveConfig(use_reranker=False),
    )
    hits = r.retrieve("glue benchmark", k=4)
    assert hits[0].id == "d2"
    # Provenance is recorded.
    assert hits[0].bm25_rank == 1
    assert hits[0].dense_rank == 1


def test_reranker_reorders_results():
    # Fusion would put d1 first, but the reranker prefers d3 -> d3 must end up first.
    scores = {c.text: 0.0 for c in CHUNKS}
    scores[CHUNKS[2].text] = 9.9  # d3 gets the top rerank score
    rr = FakeReranker(scores)
    cfg = RetrieveConfig(use_reranker=True, rerank_candidates=10)
    r = HybridRetriever(
        CHUNKS, FakeDense(["d1", "d2", "d3", "d4"]), _bm25(), reranker=rr, config=cfg
    )
    hits = r.retrieve("transformer", k=4)
    assert hits[0].id == "d3"
    assert hits[0].rerank_score == 9.9
    assert rr.called_with is not None  # reranker actually invoked


def test_k_limit_respected():
    r = HybridRetriever(CHUNKS, FakeDense(["d1", "d2", "d3", "d4"]), _bm25(), reranker=None)
    assert len(r.retrieve("transformer", k=2)) == 2
