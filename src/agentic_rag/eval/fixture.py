"""Freeze / restore the Qdrant index as a committed fixture.

CI can't rebuild the index from source — the corpus PDFs (``data/raw/``) and the
Qdrant volume are git-ignored, and ``rag-ingest`` downloads models and parses
PDFs. So we freeze a known-good index as a small committed artifact
(``eval/fixtures/index.jsonl.gz``: one ``{"id","vector","payload"}`` object per
chunk, gzipped) and reload it into a fresh Qdrant in CI. Deterministic and
arXiv-independent.

``load_fixture`` mirrors the collection ``ingest/index.py`` builds (vectors
``size=384, distance=COSINE`` plus the ``arxiv_id`` keyword index) so the
retriever behaves identically against the restored collection.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator

from ..ingest.config import EmbedConfig


def iter_fixture(path) -> Iterator[dict]:
    """Yield the {id, vector, payload} records from a gzipped-JSONL fixture."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_fixture(path, records: Iterable[dict]) -> int:
    """Write records to a gzipped-JSONL fixture; return how many were written."""
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_fixture(
    client, path, collection: str, vector_size: int | None = None, batch: int = 256
) -> int:
    """Recreate ``collection`` from the fixture and upsert every point.

    Drops any existing collection of that name first, then recreates it with the
    same vector params + payload index as ingestion, and upserts the frozen points
    (ids/vectors/payloads preserved, so retrieval is identical). Returns the count.
    """
    from qdrant_client import models

    size = vector_size or EmbedConfig().dim
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="arxiv_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    total = 0
    buf: list = []
    for rec in iter_fixture(path):
        buf.append(models.PointStruct(id=rec["id"], vector=rec["vector"], payload=rec["payload"]))
        if len(buf) >= batch:
            client.upsert(collection_name=collection, points=buf)
            total += len(buf)
            buf = []
    if buf:
        client.upsert(collection_name=collection, points=buf)
        total += len(buf)
    return total
