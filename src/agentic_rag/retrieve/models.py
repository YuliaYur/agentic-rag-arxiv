"""The retrieval-layer result type.

Mirrors the ingestion ``Chunk``'s citation metadata, but adds the scores and
rank provenance produced by retrieval. Kept separate from ingestion's ``Chunk``
on purpose: ingestion builds chunks (with embed_text, hashes); retrieval returns
chunks *with relevance scores*. Different concerns, different type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with its source metadata intact + scores."""

    id: str            # Qdrant point id (deterministic uuid5 from ingestion)
    text: str
    arxiv_id: str
    title: str
    slug: str
    section: str
    page: int
    page_end: int
    chunk_index: int

    # Scores / provenance (populated during retrieval; None when not applicable).
    score: float = 0.0              # the fused (RRF) score
    dense_rank: int | None = None   # 1-based rank in dense results (None if absent)
    bm25_rank: int | None = None    # 1-based rank in BM25 results (None if absent)
    rerank_score: float | None = None  # cross-encoder score (None if not reranked)

    def citation(self) -> str:
        pages = f"p.{self.page}" if self.page == self.page_end else f"p.{self.page}-{self.page_end}"
        return f"{self.title} ({self.arxiv_id}) §{self.section} {pages}"
