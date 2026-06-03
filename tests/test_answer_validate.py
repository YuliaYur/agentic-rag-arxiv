"""Tests for grounding validation — the cite-or-abstain enforcement (no LLM)."""

from __future__ import annotations

from agentic_rag.answer.schemas import Citation, CitedAnswer
from agentic_rag.answer.validate import validate_cited_answer
from agentic_rag.retrieve.models import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    # S1 -> ELECTRA, S2 -> BERT
    return [
        RetrievedChunk(
            id="a",
            text="...",
            arxiv_id="2003.10555",
            title="ELECTRA",
            slug="electra",
            section="3 Method",
            page=4,
            page_end=4,
            chunk_index=0,
        ),
        RetrievedChunk(
            id="b",
            text="...",
            arxiv_id="1810.04805",
            title="BERT",
            slug="bert",
            section="3.1 Pre-training",
            page=4,
            page_end=5,
            chunk_index=1,
        ),
    ]


def test_grounded_citation_uses_authoritative_metadata():
    # Model supplies WRONG arxiv_id/section/page but a valid source_id.
    raw = CitedAnswer(
        answer="ELECTRA uses replaced-token detection [S1].",
        citations=[Citation(source_id="S1", arxiv_id="WRONG", section="WRONG", page=999)],
        insufficient_context=False,
    )
    v = validate_cited_answer("q", raw, _chunks())
    assert v.is_grounded
    assert len(v.citations) == 1
    c = v.citations[0]
    # Authoritative values come from the chunk, not the model's copy.
    assert c.arxiv_id == "2003.10555"
    assert c.section == "3 Method"
    assert c.page == 4


def test_unknown_citation_source_is_violation():
    raw = CitedAnswer(
        answer="claim [S1].",
        citations=[Citation(source_id="S9", arxiv_id="x", section="y", page=1)],
        insufficient_context=False,
    )
    v = validate_cited_answer("q", raw, _chunks())
    assert not v.is_grounded
    assert any("unknown source" in m for m in v.violations)


def test_inline_marker_is_grounded_and_added_to_citations():
    # Model cited [S2] inline but listed no structured citations.
    raw = CitedAnswer(answer="BERT masks tokens [S2].", citations=[], insufficient_context=False)
    v = validate_cited_answer("q", raw, _chunks())
    assert v.is_grounded
    assert [c.source_id for c in v.citations] == ["S2"]
    assert v.citations[0].arxiv_id == "1810.04805"


def test_unknown_inline_marker_is_violation():
    raw = CitedAnswer(
        answer="a claim [S5].",
        citations=[Citation(source_id="S1", arxiv_id="2003.10555", section="3 Method", page=4)],
        insufficient_context=False,
    )
    v = validate_cited_answer("q", raw, _chunks())
    assert not v.is_grounded
    assert any("[S5]" in m for m in v.violations)


def test_insufficient_context_path_is_grounded():
    raw = CitedAnswer(
        answer="I don't have enough information in the provided sources.",
        citations=[],
        insufficient_context=True,
    )
    v = validate_cited_answer("q", raw, _chunks())
    assert v.is_grounded and v.insufficient_context and not v.citations


def test_insufficient_context_with_citations_is_violation():
    raw = CitedAnswer(
        answer="...",
        citations=[Citation(source_id="S1", arxiv_id="2003.10555", section="3 Method", page=4)],
        insufficient_context=True,
    )
    v = validate_cited_answer("q", raw, _chunks())
    assert not v.is_grounded


def test_claims_without_any_citation_is_violation():
    raw = CitedAnswer(
        answer="ELECTRA is great and BERT is also great.",
        citations=[],
        insufficient_context=False,
    )
    v = validate_cited_answer("q", raw, _chunks())
    assert not v.is_grounded
    assert any("cites no grounded sources" in m for m in v.violations)


def test_duplicate_citations_deduped():
    raw = CitedAnswer(
        answer="a [S1] and again [S1].",
        citations=[
            Citation(source_id="S1", arxiv_id="2003.10555", section="3 Method", page=4),
            Citation(source_id="S1", arxiv_id="2003.10555", section="3 Method", page=4),
        ],
        insufficient_context=False,
    )
    v = validate_cited_answer("q", raw, _chunks())
    assert len(v.citations) == 1
