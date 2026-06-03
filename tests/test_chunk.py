"""Tests for structure-aware chunking logic (offline, no model / no network)."""

from __future__ import annotations

import dataclasses

from agentic_rag.ingest.chunk import (
    chunk_blocks,
    count_tokens,
    split_sentences,
)
from agentic_rag.ingest.config import ChunkConfig
from agentic_rag.ingest.parse import Block

CFG = ChunkConfig(target_tokens=40, overlap_tokens=10, min_tokens=8, max_tokens=60)


def _sentence(word: str, n: int) -> str:
    # A capitalized sentence ~ n words long, all sharing a marker word.
    return ("The " + (word + " ") * n).strip() + "."


def _blocks(section: str, body: str, page: int = 1) -> Block:
    return Block(text=body, page=page, section=section, is_heading=False)


def test_split_sentences_basic():
    text = "First sentence here. Second one follows! Third? Yes."
    assert split_sentences(text) == [
        "First sentence here.",
        "Second one follows!",
        "Third?",
        "Yes.",
    ]


def test_chunks_never_cross_sections():
    blocks = [
        _blocks("Introduction", " ".join(_sentence("alpha", 6) for _ in range(8))),
        _blocks("Methods", " ".join(_sentence("beta", 6) for _ in range(8))),
    ]
    chunks = chunk_blocks(blocks, arxiv_id="1", title="T", slug="t", config=CFG)

    assert {c.section for c in chunks} == {"Introduction", "Methods"}
    for c in chunks:
        if c.section == "Introduction":
            assert "beta" not in c.text
        if c.section == "Methods":
            assert "alpha" not in c.text


def test_chunk_sizes_within_hard_max():
    body = " ".join(_sentence("alpha", 6) for _ in range(20))
    chunks = chunk_blocks([_blocks("S", body)], arxiv_id="1", title="T", slug="t", config=CFG)
    assert len(chunks) > 1  # body should split into several chunks
    for c in chunks:
        assert c.n_tokens <= CFG.max_tokens


def test_overlap_between_consecutive_chunks():
    body = " ".join(_sentence(f"w{i}", 5) for i in range(20))
    chunks = chunk_blocks([_blocks("S", body)], arxiv_id="1", title="T", slug="t", config=CFG)
    assert len(chunks) >= 2
    # With overlap > 0, each chunk after the first must share some text with its
    # predecessor's tail (sentence-level carry-over).
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_sents = set(split_sentences(prev.text))
        nxt_sents = set(split_sentences(nxt.text))
        assert prev_sents & nxt_sents, "expected overlapping sentence(s)"


def test_no_overlap_when_disabled():
    cfg = dataclasses.replace(CFG, overlap_tokens=0)
    body = " ".join(_sentence(f"w{i}", 5) for i in range(20))
    chunks = chunk_blocks([_blocks("S", body)], arxiv_id="1", title="T", slug="t", config=cfg)
    seen: set[str] = set()
    for c in chunks:
        sents = split_sentences(c.text)
        assert not (set(sents) & seen), "no sentence should repeat without overlap"
        seen.update(sents)


def test_oversize_single_sentence_is_split():
    # One 200-word sentence with no internal punctuation -> must be word-split.
    giant = "Begin " + "blah " * 200 + "end."
    chunks = chunk_blocks([_blocks("S", giant)], arxiv_id="1", title="T", slug="t", config=CFG)
    assert len(chunks) > 1
    for c in chunks:
        assert c.n_tokens <= CFG.max_tokens


def test_small_trailing_fragment_is_merged():
    # Several full sentences plus one tiny trailing sentence in the same section.
    body = " ".join(_sentence("alpha", 6) for _ in range(10)) + " Tiny."
    chunks = chunk_blocks([_blocks("S", body)], arxiv_id="1", title="T", slug="t", config=CFG)
    assert len(chunks) >= 2
    # No chunk below the minimum once merging has happened.
    assert all(c.n_tokens >= CFG.min_tokens for c in chunks)


def test_noise_fragments_are_dropped():
    # A lone tiny block in its own section (e.g. a stray page number) is dropped.
    blocks = [
        _blocks("1 Introduction", " ".join(_sentence("alpha", 6) for _ in range(6))),
        _blocks("Footer", "6", page=2),  # noise: a single token
    ]
    chunks = chunk_blocks(blocks, arxiv_id="1", title="T", slug="t", config=CFG)
    assert all(c.section != "Footer" for c in chunks)
    assert all(c.n_tokens >= 5 for c in chunks)


def test_count_tokens_is_deterministic():
    assert count_tokens("The cat sat on the mat.") == 7
    assert count_tokens("") == 0
