"""Tests for prompt / context construction (no LLM, no network)."""

from __future__ import annotations

from agentic_rag.answer.prompt import build_user_prompt, format_sources
from agentic_rag.retrieve.models import RetrievedChunk


def _chunk(i: int, **kw) -> RetrievedChunk:
    defaults = {
        "id": str(i),
        "text": "some   text\nwith   messy   whitespace",
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "slug": "transformer",
        "section": "6.1 Machine Translation",
        "page": 8,
        "page_end": 8,
        "chunk_index": i,
    }
    defaults.update(kw)
    return RetrievedChunk(**defaults)


def test_sources_are_labeled_and_carry_metadata():
    out = format_sources([_chunk(0), _chunk(1)])
    assert "[S1]" in out and "[S2]" in out
    assert "arXiv:1706.03762" in out
    assert "§6.1 Machine Translation" in out
    assert "p.8" in out


def test_source_text_whitespace_collapsed():
    out = format_sources([_chunk(0)])
    assert "some text with messy whitespace" in out


def test_page_range_rendered_when_spanning():
    out = format_sources([_chunk(0, page=7, page_end=8)])
    assert "p.7-8" in out


def test_user_prompt_has_question_and_sources():
    up = build_user_prompt("Why self-attention?", [_chunk(0)])
    assert "QUESTION: Why self-attention?" in up
    assert "[S1]" in up
