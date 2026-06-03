"""Structure-aware chunking + chunk metadata.

Design (rationale and tuning guidance in DECISIONS.md):

* Chunks **never cross a section boundary** — a chunk is always "about" one
  section, which keeps embeddings coherent and lets us cite an exact section.
* Packing works at **sentence granularity** so boundaries fall between
  sentences, not mid-thought, and so overlap is a clean sentence carry-over.
* Each chunk carries full citation metadata: arxiv_id, title, slug, section,
  page (start) + page_end, plus a content hash and a deterministic point id for
  idempotent indexing.

Token counting is injected (``token_counter``). The real pipeline passes the
embedding model's own tokenizer so we never exceed its context window; tests and
the offline default use a fast regex word/punctuation counter.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from .config import ChunkConfig
from .parse import Block

# Deterministic namespace for chunk point ids (so re-runs upsert, never duplicate).
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

# Chunks below this many tokens are PDF noise (stray page numbers, figure
# labels, lone symbols) rather than content, and are dropped.
_NOISE_TOKEN_FLOOR = 5

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
# Sentence boundary: punctuation followed by whitespace and a capital / opener.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")

TokenCounter = Callable[[str], int]


def count_tokens(text: str) -> int:
    """Fast, offline, deterministic token approximation (words + punctuation)."""
    return len(_TOKEN_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    """Split a block of text into sentences. Crude but offline and deterministic.

    Imperfect on abbreviations / inline equations; acceptable because chunk
    boundaries only need to be *reasonable*, not linguistically perfect.
    """
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


@dataclass(frozen=True)
class Chunk:
    """A retrievable chunk plus everything needed to cite and index it."""

    text: str  # clean text shown to the user / returned by retrieval
    embed_text: str  # what actually gets embedded (may include a context header)
    arxiv_id: str
    title: str
    slug: str
    section: str
    page: int  # first page the chunk touches (1-based)
    page_end: int  # last page the chunk touches
    chunk_index: int  # position within the paper
    n_tokens: int
    content_hash: str

    def point_id(self) -> str:
        """Deterministic id: same (paper, index) -> same id -> idempotent upsert."""
        return str(uuid.uuid5(_NAMESPACE, f"{self.arxiv_id}:{self.chunk_index}"))

    def payload(self) -> dict:
        """Qdrant payload: the metadata we store alongside the vector for citation."""
        return {
            "text": self.text,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "slug": self.slug,
            "section": self.section,
            "page": self.page,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
            "n_tokens": self.n_tokens,
            "content_hash": self.content_hash,
        }


@dataclass
class _Unit:
    """A sentence (or word-split fragment) tagged with its source page."""

    text: str
    page: int


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _group_sections(blocks: Iterable[Block], respect: bool) -> Iterator[tuple[str, list[Block]]]:
    """Yield (section, blocks) groups. If not respecting sections, one big group."""
    blocks = list(blocks)
    if not respect:
        yield ("", blocks)
        return
    group: list[Block] = []
    current: str | None = None
    for blk in blocks:
        if current is None:
            current = blk.section
        if blk.section != current:
            yield (current, group)
            group, current = [], blk.section
        group.append(blk)
    if group:
        yield (current or "", group)


def _to_units(blocks: list[Block], cfg: ChunkConfig, count: TokenCounter) -> list[_Unit]:
    """Flatten body blocks into page-tagged sentence units, hard-splitting any
    sentence longer than max_tokens (e.g. a runaway equation line)."""
    units: list[_Unit] = []
    for blk in blocks:
        if blk.is_heading:
            continue  # the heading is the section label, not body content
        for sent in split_sentences(blk.text):
            if count(sent) <= cfg.max_tokens:
                units.append(_Unit(sent, blk.page))
                continue
            # Oversize single sentence: split on words into target-sized pieces.
            words, buf = sent.split(), []
            for w in words:
                buf.append(w)
                if count(" ".join(buf)) >= cfg.target_tokens:
                    units.append(_Unit(" ".join(buf), blk.page))
                    buf = []
            if buf:
                units.append(_Unit(" ".join(buf), blk.page))
    return units


def _overlap_tail(units: list[_Unit], overlap_tokens: int, count: TokenCounter) -> list[_Unit]:
    """Trailing units whose token sum stays within overlap_tokens (contiguous tail).

    Crucially never carries a unit larger than the overlap budget, so a big
    word-split fragment can't smuggle itself into the next chunk and blow the
    hard ceiling.
    """
    if overlap_tokens <= 0:
        return []
    tail: list[_Unit] = []
    total = 0
    for u in reversed(units):
        t = count(u.text)
        if total + t > overlap_tokens:
            break  # stop at the first unit that doesn't fit (keep the tail contiguous)
        tail.insert(0, u)
        total += t
    return tail


def _emit(units: list[_Unit], count: TokenCounter) -> tuple[str, int, int, int]:
    """Build (text, page_start, page_end, n_tokens) from a unit list."""
    text = " ".join(u.text for u in units)
    pages = [u.page for u in units]
    return text, min(pages), max(pages), count(text)


def _pack_section(
    units: list[_Unit], cfg: ChunkConfig, count: TokenCounter
) -> list[tuple[str, int, int, int]]:
    """Greedily pack units into overlapping chunks within a single section."""
    raw: list[tuple[str, int, int, int]] = []
    cur: list[_Unit] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur, cur_tokens
        raw.append(_emit(cur, count))
        cur = _overlap_tail(cur, cfg.overlap_tokens, count)
        cur_tokens = sum(count(x.text) for x in cur)

    for u in units:
        t = count(u.text)
        if cur and cur_tokens + t > cfg.max_tokens:
            # Hard ceiling: never let a chunk exceed the model window, regardless
            # of the min-size gate below.
            flush()
        elif cur and cur_tokens + t > cfg.target_tokens and cur_tokens >= cfg.min_tokens:
            # Soft target: close at a sentence boundary once we're big enough.
            flush()
        cur.append(u)
        cur_tokens += t
        # A single unit that alone exceeds the ceiling is emitted solo (no overlap).
        if len(cur) == 1 and cur_tokens > cfg.max_tokens:
            raw.append(_emit(cur, count))
            cur, cur_tokens = [], 0

    if cur and cur_tokens > 0:
        raw.append(_emit(cur, count))

    # Merge a too-small trailing fragment into the previous chunk (same section).
    if len(raw) >= 2 and raw[-1][3] < cfg.min_tokens:
        prev, last = raw[-2], raw[-1]
        merged_text = prev[0] + " " + last[0]
        raw[-2] = (merged_text, min(prev[1], last[1]), max(prev[2], last[2]), count(merged_text))
        raw.pop()
    return raw


def chunk_blocks(
    blocks: list[Block],
    *,
    arxiv_id: str,
    title: str,
    slug: str,
    config: ChunkConfig | None = None,
    token_counter: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Turn parsed blocks into citation-ready, section-aware chunks."""
    cfg = config or ChunkConfig()
    chunks: list[Chunk] = []
    index = 0

    for section, group in _group_sections(blocks, cfg.respect_sections):
        units = _to_units(group, cfg, token_counter)
        for text, page_start, page_end, n_tokens in _pack_section(units, cfg, token_counter):
            if not text.strip() or n_tokens < _NOISE_TOKEN_FLOOR:
                continue  # drop empty/noise fragments (e.g. a lone page number)
            header = f"{title} > {section}\n" if (cfg.prepend_context and section) else ""
            chunks.append(
                Chunk(
                    text=text,
                    embed_text=header + text,
                    arxiv_id=arxiv_id,
                    title=title,
                    slug=slug,
                    section=section,
                    page=page_start,
                    page_end=page_end,
                    chunk_index=index,
                    n_tokens=n_tokens,
                    content_hash=_content_hash(text),
                )
            )
            index += 1
    return chunks
