"""Tests for HybridRetriever orchestration, using fakes (no Qdrant, no model)."""

from __future__ import annotations

from agentic_rag.retrieve.bm25 import BM25Index
from agentic_rag.retrieve.config import RetrieveConfig
from agentic_rag.retrieve.models import RetrievedChunk
from agentic_rag.retrieve.retriever import (
    HybridRetriever,
    anchor_query_to_title,
    detect_named_papers,
    round_robin_merge,
)


def _chunk(i: str, text: str, arxiv_id: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id=i,
        text=text,
        arxiv_id=arxiv_id or f"id-{i}",
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


def test_reranker_blends_with_fusion():
    # Reranker strongly prefers d3 (a mid-fusion doc): the blend promotes d3 into the
    # top results, while the fusion-#1 doc (d1) is NOT buried -> both sit at the top.
    scores = {c.text: 0.0 for c in CHUNKS}
    scores[CHUNKS[2].text] = 9.9  # d3 gets the top rerank score
    rr = FakeReranker(scores)
    cfg = RetrieveConfig(use_reranker=True, rerank_candidates=10)
    r = HybridRetriever(
        CHUNKS, FakeDense(["d1", "d2", "d3", "d4"]), _bm25(), reranker=rr, config=cfg
    )
    hits = r.retrieve("transformer", k=4)
    top2 = {hits[0].id, hits[1].id}
    assert "d3" in top2  # reranker promoted d3 from mid-fusion
    assert "d1" in top2  # but the fusion-strong doc is not discarded
    assert rr.called_with is not None  # reranker actually invoked


def test_fusion_strong_survives_a_bad_reranker():
    # The q-0006 failure mode: a doc strong in fusion (d2, forced to fusion #1) but
    # which the reranker hates, while the reranker loves a fusion-weak doc (d4).
    # Pure rerank-sort would bury d2 at the bottom; the blend keeps it in the top-k.
    scores = {c.text: 0.0 for c in CHUNKS}
    scores[CHUNKS[1].text] = -9.9  # d2: reranker hates the fusion-strong doc
    scores[CHUNKS[3].text] = 9.9  # d4: reranker loves a fusion-weak doc
    rr = FakeReranker(scores)
    cfg = RetrieveConfig(use_reranker=True, rerank_candidates=10)
    r = HybridRetriever(
        CHUNKS, FakeDense(["d2", "d1", "d3", "d4"]), _bm25(), reranker=rr, config=cfg
    )
    hits = r.retrieve("glue benchmark", k=2)  # bm25 also favors d2 -> d2 is fusion #1
    ids = [h.id for h in hits]
    assert "d2" in ids  # not buried by the reranker that hated it


def test_k_limit_respected():
    r = HybridRetriever(CHUNKS, FakeDense(["d1", "d2", "d3", "d4"]), _bm25(), reranker=None)
    assert len(r.retrieve("transformer", k=2)) == 2


# --- paper-diversity cap (q-0005: one paper monopolizing the top-k) ----------


def _same_paper_chunks():
    # Four chunks: p1 has three (c1,c2,c3), p2 has one (c4). p1 dominates ranking.
    return [
        _chunk("c1", "alpha one", arxiv_id="p1"),
        _chunk("c2", "alpha two", arxiv_id="p1"),
        _chunk("c3", "alpha three", arxiv_id="p1"),
        _chunk("c4", "beta four", arxiv_id="p2"),
    ]


def test_cap_makes_room_for_a_second_paper():
    chunks = _same_paper_chunks()
    cfg = RetrieveConfig(use_reranker=False, max_per_paper=2)
    # Dense order puts all three p1 chunks first; without the cap p2 is buried.
    r = HybridRetriever(chunks, FakeDense(["c1", "c2", "c3", "c4"]), _bm25_for(chunks), config=cfg)
    hits = r.retrieve("alpha", k=3)
    papers = [h.arxiv_id for h in hits]
    assert papers.count("p1") == 2  # capped at 2
    assert "p2" in papers  # the second paper got a slot it otherwise wouldn't


def test_cap_backfills_when_too_few_papers():
    # Single-paper question: only p1 chunks exist. The cap must NOT shrink the
    # result below k -- it backfills the capped chunks rather than returning short.
    chunks = [_chunk(f"c{i}", f"alpha {i}", arxiv_id="p1") for i in range(1, 5)]
    cfg = RetrieveConfig(use_reranker=False, max_per_paper=2)
    r = HybridRetriever(chunks, FakeDense(["c1", "c2", "c3", "c4"]), _bm25_for(chunks), config=cfg)
    hits = r.retrieve("alpha", k=3)
    assert len(hits) == 3  # not truncated to the cap of 2
    assert {h.arxiv_id for h in hits} == {"p1"}


def test_cap_none_disables():
    chunks = _same_paper_chunks()
    cfg = RetrieveConfig(use_reranker=False, max_per_paper=None)
    r = HybridRetriever(chunks, FakeDense(["c1", "c2", "c3", "c4"]), _bm25_for(chunks), config=cfg)
    hits = r.retrieve("alpha", k=3)
    assert [h.arxiv_id for h in hits] == ["p1", "p1", "p1"]  # no diversification


def _bm25_for(chunks):
    return BM25Index([c.id for c in chunks], [c.text for c in chunks])


# --- round-robin merge (decomposed per-side retrieval) -----------------------


def test_round_robin_alternates_sides():
    # Two per-side result lists; merge must interleave so BOTH sides appear early.
    side_a = [_chunk("a1", "x", "A"), _chunk("a2", "x", "A"), _chunk("a3", "x", "A")]
    side_b = [_chunk("b1", "y", "B"), _chunk("b2", "y", "B")]
    merged = round_robin_merge([side_a, side_b], k=4)
    assert [c.id for c in merged] == ["a1", "b1", "a2", "b2"]  # rank-1s before rank-2s


def test_round_robin_dedupes_shared_chunks():
    shared = _chunk("s1", "shared", "A")
    side_a = [shared, _chunk("a2", "x", "A")]
    side_b = [shared, _chunk("b2", "y", "B")]
    merged = round_robin_merge([side_a, side_b], k=4)
    ids = [c.id for c in merged]
    assert ids.count("s1") == 1  # deduped
    assert set(ids) == {"s1", "a2", "b2"}


def test_round_robin_respects_k():
    side_a = [_chunk(f"a{i}", "x", "A") for i in range(5)]
    side_b = [_chunk(f"b{i}", "y", "B") for i in range(5)]
    assert len(round_robin_merge([side_a, side_b], k=3)) == 3


# --- title-anchoring of decomposed sub-queries -------------------------------

_TITLES = {
    "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "1907.11692": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "2003.10555": "ELECTRA: Pre-training Text Encoders as Discriminators",
    "1706.03762": "Attention Is All You Need",
    "2010.11929": "An Image is Worth 16x16 Words: Transformers for Image Recognition",
}


def test_anchor_locks_onto_the_named_paper_not_a_citer():
    # "BERT masked language modeling" shares "bert" with BOTH the BERT and RoBERTa
    # titles; the anchor must pick the actual BERT paper (more distinctive overlap)
    # and prepend its title terms.
    out = anchor_query_to_title("BERT masked language modeling objective", _TITLES)
    assert "bidirectional" in out.lower()  # BERT's distinctive title word was prepended
    assert out.lower().endswith("bert masked language modeling objective")


def test_anchor_alias_handles_name_not_equal_title():
    # The original Transformer's title is "Attention Is All You Need" -- a query
    # that says "original transformer" shares no title token, so only the alias
    # can anchor it.
    aliases = {"original transformer": "1706.03762"}
    out = anchor_query_to_title("original transformer token embeddings", _TITLES, aliases)
    assert "attention" in out.lower() and "need" in out.lower()


def test_anchor_is_noop_without_a_strong_match():
    # A query naming no corpus paper distinctively is returned unchanged.
    q = "general question about optimization tricks"
    assert anchor_query_to_title(q, _TITLES) == q


def test_anchor_noop_on_empty_titles():
    assert anchor_query_to_title("anything", {}) == "anything"


# --- named-paper detection (deterministic decomposition trigger) -------------

_NAMES = (
    ("bert", "1810.04805"),
    ("roberta", "1907.11692"),
    ("electra", "2003.10555"),
    ("masked autoencoder", "2111.06377"),
    ("original transformer", "1706.03762"),
    ("vit", "2010.11929"),
)


def test_detect_names_each_compared_subject():
    named = detect_named_papers(
        "How does ELECTRA's pre-training objective differ from BERT's masked language modeling?",
        _NAMES,
    )
    assert named == {"2003.10555", "1810.04805"}  # ELECTRA + BERT


def test_detect_does_not_confuse_bert_for_roberta():
    # "BERT" must NOT also flag RoBERTa (whose title merely contains "BERT").
    named = detect_named_papers("What are BERT's two pre-training objectives?", _NAMES)
    assert named == {"1810.04805"}


def test_detect_masked_word_does_not_flag_mae():
    # "masked language modeling" must not match MAE ("masked autoencoder").
    named = detect_named_papers("Explain masked language modeling in BERT", _NAMES)
    assert named == {"1810.04805"}


def test_detect_multiword_alias_for_name_not_in_title():
    named = detect_named_papers(
        "How does ViT relate to the original Transformer's token embeddings?", _NAMES
    )
    assert named == {"2010.11929", "1706.03762"}


def test_detect_alias_requires_contiguous_phrase():
    # "Swin Transformer ... the original ViT": both "original" and "transformer"
    # appear, but NOT as the phrase "original transformer" -- it must not mis-fire
    # the Transformer paper (the q-0026 false positive).
    named = detect_named_papers(
        "How does Swin Transformer's hierarchy differ from the original ViT?", _NAMES
    )
    assert "1706.03762" not in named
    assert named == {"2010.11929"}  # only ViT (Swin isn't in this mini-registry)
