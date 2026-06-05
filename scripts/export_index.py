"""Export the live Qdrant index to the committed CI fixture.

    python scripts/export_index.py            # -> eval/fixtures/index.jsonl.gz

Run this locally whenever the index changes (after `rag-ingest`), then commit the
fixture. CI loads it with `scripts/load_index_fixture.py` instead of fetching the
corpus and re-ingesting. Needs Qdrant up with the built collection.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "index.jsonl.gz"


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None) -> int:
    _utf8()
    p = argparse.ArgumentParser(description="Freeze the Qdrant index to a committed fixture.")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output .jsonl.gz path")
    p.add_argument("--collection", default=None, help="override the collection name")
    args = p.parse_args(argv)

    from qdrant_client import QdrantClient

    from agentic_rag.eval.fixture import write_fixture
    from agentic_rag.ingest.config import QdrantConfig

    cfg = QdrantConfig()
    collection = args.collection or cfg.collection
    client = QdrantClient(host=cfg.host, port=cfg.port, check_compatibility=False)
    if not client.collection_exists(collection):
        print(
            f"collection {collection!r} not found — build it first (rag-ingest).", file=sys.stderr
        )
        return 2

    def records():
        offset = None
        while True:
            points, offset = client.scroll(
                collection, limit=256, offset=offset, with_payload=True, with_vectors=True
            )
            for pt in points:
                yield {"id": str(pt.id), "vector": pt.vector, "payload": pt.payload}
            if offset is None:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_fixture(out, records())
    size_kb = out.stat().st_size / 1024
    print(f"exported {n} points from '{collection}' -> {out}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
