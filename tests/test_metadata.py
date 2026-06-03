"""Tests for chunk metadata: citation fields, page ranges, ids, payload."""

from __future__ import annotations

import hashlib

from agentic_rag.ingest.chunk import chunk_blocks
from agentic_rag.ingest.config import ChunkConfig
from agentic_rag.ingest.parse import Block

CFG = ChunkConfig(target_tokens=40, overlap_tokens=0, min_tokens=4, max_tokens=60)


def _sentence(word: str, n: int) -> str:
    return ("The " + (word + " ") * n).strip() + "."


def _make_chunks(**kw):
    blocks = [
        Block(
            text=" ".join(_sentence("alpha", 6) for _ in range(6)),
            page=1,
            section="Introduction",
            is_heading=False,
        ),
        Block(
            text=" ".join(_sentence("beta", 6) for _ in range(6)),
            page=2,
            section="Introduction",
            is_heading=False,
        ),
    ]
    defaults = {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "slug": "transformer",
        "config": CFG,
    }
    defaults.update(kw)
    return chunk_blocks(blocks, **defaults)


def test_citation_fields_propagate():
    chunks = _make_chunks()
    for c in chunks:
        assert c.arxiv_id == "1706.03762"
        assert c.title == "Attention Is All You Need"
        assert c.slug == "transformer"
        assert c.section == "Introduction"


def test_chunk_index_is_sequential():
    chunks = _make_chunks()
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_range_spans_source_pages():
    # A chunk built from text on pages 1 and 2 must report page=1, page_end=2.
    chunks = _make_chunks()
    spanning = [c for c in chunks if c.page != c.page_end]
    assert spanning, "expected at least one chunk to span the page boundary"
    for c in spanning:
        assert c.page == 1 and c.page_end == 2
    # Every chunk's page <= page_end and within the source pages.
    for c in chunks:
        assert 1 <= c.page <= c.page_end <= 2


def test_content_hash_matches_text():
    for c in _make_chunks():
        assert c.content_hash == hashlib.sha1(c.text.encode("utf-8")).hexdigest()


def test_point_id_is_deterministic_and_unique():
    a = _make_chunks()
    b = _make_chunks()
    # Stable across runs (idempotent upsert relies on this).
    assert [c.point_id() for c in a] == [c.point_id() for c in b]
    # Unique per chunk index.
    assert len({c.point_id() for c in a}) == len(a)


def test_payload_has_all_citation_keys():
    c = _make_chunks()[0]
    payload = c.payload()
    for key in (
        "text",
        "arxiv_id",
        "title",
        "slug",
        "section",
        "page",
        "page_end",
        "chunk_index",
        "n_tokens",
        "content_hash",
    ):
        assert key in payload
    assert payload["text"] == c.text


def test_embed_text_includes_context_header_when_enabled():
    c = _make_chunks()[0]  # CFG has prepend_context=True by default
    assert c.embed_text.startswith("Attention Is All You Need > Introduction\n")
    assert c.embed_text.endswith(c.text)


def test_embed_text_is_plain_when_context_disabled():
    import dataclasses

    cfg = dataclasses.replace(CFG, prepend_context=False)
    c = _make_chunks(config=cfg)[0]
    assert c.embed_text == c.text
