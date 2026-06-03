"""Inspect the Qdrant index: point count, token-size stats, and a sample query.

Doubles as a retrieval smoke test that the stored metadata is enough to cite a
source (title + section + page).

    python scripts/inspect_index.py
    python scripts/inspect_index.py --query "how is the quadratic cost of attention reduced?"
"""

from __future__ import annotations

import argparse
import contextlib
import sys

from agentic_rag.ingest.config import IngestConfig


def main() -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser()
    p.add_argument("--query", default="What is the computational complexity of self-attention?")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()

    cfg = IngestConfig()
    from qdrant_client import QdrantClient

    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, check_compatibility=False)

    total = client.count(cfg.qdrant.collection, exact=True).count
    print(f"Collection '{cfg.qdrant.collection}': {total} points (dim={cfg.embed.dim}, cosine)\n")

    # Token-size sanity: scroll all payloads and summarize n_tokens.
    tokens: list[int] = []
    offset = None
    while True:
        points, offset = client.scroll(
            cfg.qdrant.collection,
            limit=1000,
            offset=offset,
            with_payload=["n_tokens"],
            with_vectors=False,
        )
        tokens.extend(pt.payload["n_tokens"] for pt in points)
        if offset is None:
            break
    tokens.sort()
    n = len(tokens)
    over = sum(1 for t in tokens if t > 512)
    print(
        f"chunk tokens (model tokenizer): min={tokens[0]} median={tokens[n // 2]} "
        f"max={tokens[-1]} mean={sum(tokens) / n:.0f}"
    )
    print(f"chunks over 512 (model window): {over}\n")

    # Sample semantic query -> show citations.
    from agentic_rag.ingest.embed import Embedder

    embedder = Embedder(cfg.embed)
    # bge-v1.5 wants this instruction prefix on the QUERY side only.
    q = "Represent this sentence for searching relevant passages: " + args.query
    qvec = embedder.encode([q])[0]
    hits = client.query_points(
        cfg.qdrant.collection, query=qvec, limit=args.top, with_payload=True
    ).points

    print(f'Query: "{args.query}"\n' + "-" * 78)
    for h in hits:
        pl = h.payload
        pages = (
            f"p.{pl['page']}"
            if pl["page"] == pl["page_end"]
            else f"p.{pl['page']}-{pl['page_end']}"
        )
        print(f"[{h.score:.3f}] {pl['title']} ({pl['arxiv_id']}) — {pl['section']} {pages}")
        print(f"        {pl['text'][:160].strip()}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
