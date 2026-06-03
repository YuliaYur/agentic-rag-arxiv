"""Dense (vector) search against Qdrant, plus loading chunks for BM25.

These are the two pieces that touch Qdrant/the embedding model; everything else
in the retrieval layer is pure logic. Kept thin so the orchestrator can be
tested with fakes.
"""

from __future__ import annotations

from .config import RetrieveConfig
from .models import RetrievedChunk


class QdrantDenseSearcher:
    """Embeds the query (with the bge query prefix) and runs cosine search."""

    def __init__(self, embedder, client, collection: str, config: RetrieveConfig) -> None:
        self._embedder = embedder
        self._client = client
        self._collection = collection
        self._cfg = config

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        qvec = self._embedder.encode([self._cfg.query_prefix + query])[0]
        hits = self._client.query_points(
            self._collection, query=qvec, limit=limit, with_payload=False
        ).points
        return [(str(h.id), float(h.score)) for h in hits]


def load_chunks(client, collection: str) -> list[RetrievedChunk]:
    """Pull every chunk's metadata from Qdrant once, to build the BM25 index and
    to resolve ids -> chunks at query time."""
    chunks: list[RetrievedChunk] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=1000, offset=offset, with_payload=True, with_vectors=False
        )
        for p in points:
            pl = p.payload
            chunks.append(
                RetrievedChunk(
                    id=str(p.id),
                    text=pl["text"],
                    arxiv_id=pl["arxiv_id"],
                    title=pl["title"],
                    slug=pl["slug"],
                    section=pl["section"],
                    page=pl["page"],
                    page_end=pl["page_end"],
                    chunk_index=pl["chunk_index"],
                )
            )
        if offset is None:
            break
    return chunks
