"""Qdrant vector index: idempotent upsert of chunks + their citation metadata.

Idempotency strategy:
* Point ids are deterministic (uuid5 of "arxiv_id:chunk_index"), so re-running
  overwrites the same points instead of creating duplicates.
* After upserting a paper's chunks we prune any stale points for that paper whose
  chunk_index is beyond the new count (handles a paper that now yields fewer
  chunks, e.g. after a chunking-parameter change).
"""

from __future__ import annotations

from .chunk import Chunk
from .config import QdrantConfig


class VectorIndex:
    def __init__(self, config: QdrantConfig | None = None) -> None:
        from qdrant_client import QdrantClient

        self.config = config or QdrantConfig()
        # check_compatibility=False: silence the client/server minor-version
        # warning; the REST API surface we use is stable across these versions.
        self.client = QdrantClient(
            host=self.config.host, port=self.config.port, check_compatibility=False
        )

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        from qdrant_client import models

        exists = self.client.collection_exists(self.config.collection)
        if recreate and exists:
            self.client.delete_collection(self.config.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.config.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            # Index arxiv_id so per-paper count/prune/filter is fast.
            self.client.create_payload_index(
                collection_name=self.config.collection,
                field_name="arxiv_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _paper_filter(self, arxiv_id: str):
        from qdrant_client import models

        return models.Filter(
            must=[models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id))]
        )

    def count_for_paper(self, arxiv_id: str) -> int:
        return self.client.count(
            collection_name=self.config.collection,
            count_filter=self._paper_filter(arxiv_id),
            exact=True,
        ).count

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(id=c.point_id(), vector=v, payload=c.payload())
            for c, v in zip(chunks, vectors)
        ]
        bs = self.config.upsert_batch
        for i in range(0, len(points), bs):
            self.client.upsert(collection_name=self.config.collection, points=points[i : i + bs])

    def prune_stale(self, arxiv_id: str, keep_count: int) -> None:
        """Delete points for a paper whose chunk_index >= keep_count.

        Handles the case where re-chunking now yields fewer chunks than a prior
        run, so no orphaned points linger past the new tail.
        """
        from qdrant_client import models

        self.client.delete(
            collection_name=self.config.collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id)),
                    models.FieldCondition(key="chunk_index", range=models.Range(gte=keep_count)),
                ]
            ),
        )

    def sample(self, limit: int = 5):
        """Return a few stored points (payloads) for showing example chunks."""
        points, _ = self.client.scroll(
            collection_name=self.config.collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return points
