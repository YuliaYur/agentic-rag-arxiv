"""Load the committed index fixture into Qdrant (the CI path; replaces ingest).

    python scripts/load_index_fixture.py

Recreates the collection from `eval/fixtures/index.jsonl.gz` and upserts every
point. Needs Qdrant reachable (e.g. the CI service container, or local Docker).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "index.jsonl.gz"


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None) -> int:
    _utf8()
    p = argparse.ArgumentParser(description="Load the frozen index fixture into Qdrant.")
    p.add_argument("--path", default=str(DEFAULT_FIXTURE), help="fixture .jsonl.gz path")
    p.add_argument("--collection", default=None, help="override the collection name")
    args = p.parse_args(argv)

    if not Path(args.path).exists():
        print(f"fixture not found: {args.path}", file=sys.stderr)
        return 2

    from qdrant_client import QdrantClient

    from agentic_rag.eval.fixture import load_fixture
    from agentic_rag.ingest.config import QdrantConfig

    cfg = QdrantConfig()
    collection = args.collection or cfg.collection
    client = QdrantClient(host=cfg.host, port=cfg.port, check_compatibility=False)
    n = load_fixture(client, args.path, collection)
    print(f"loaded {n} points into '{collection}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
