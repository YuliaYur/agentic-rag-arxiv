"""CLI entry point for the ingestion pipeline.

    rag-ingest                      # parse+chunk+embed+index everything (idempotent)
    rag-ingest --dry-run            # parse+chunk only, print stats + examples (offline)
    rag-ingest --papers 1706.03762  # one paper (by arxiv_id or slug)
    rag-ingest --force              # re-embed + re-index even if up-to-date
    rag-ingest --recreate           # drop and rebuild the Qdrant collection first

Also runnable without install as: python -m agentic_rag.ingest.cli
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import textwrap

from .config import IngestConfig
from .pipeline import run_ingest


def _force_utf8_stdout() -> None:
    """Windows consoles default to a legacy codepage (cp1251 here); chunk text
    contains math symbols and other non-Latin characters. Force UTF-8 so
    printing never crashes."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _build_config(args: argparse.Namespace) -> IngestConfig:
    base = IngestConfig()
    chunk = dataclasses.replace(
        base.chunk,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    embed = dataclasses.replace(base.embed, model_name=args.model)
    qdrant = dataclasses.replace(
        base.qdrant, host=args.host, port=args.port, collection=args.collection
    )
    return dataclasses.replace(base, chunk=chunk, embed=embed, qdrant=qdrant)


def _print_examples(examples, log=print) -> None:
    if not examples:
        return
    log("\n" + "=" * 78)
    log(f"EXAMPLE CHUNKS ({len(examples)})")
    log("=" * 78)
    for c in examples:
        pages = f"p.{c.page}" if c.page == c.page_end else f"p.{c.page}-{c.page_end}"
        log(f"\n[chunk {c.chunk_index}]  {c.title}  ({c.arxiv_id})")
        log(f"  section: {c.section!r}   {pages}   {c.n_tokens} tokens   id={c.point_id()[:8]}...")
        log(f"  hash: {c.content_hash[:12]}...")
        body = textwrap.fill(c.text, width=74, initial_indent="  > ", subsequent_indent="  > ")
        log(body[:600] + ("..." if len(body) > 600 else ""))


def main(argv: list[str] | None = None) -> int:
    base = IngestConfig()
    p = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Ingest arXiv PDFs into a local Qdrant vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--papers", nargs="+", metavar="ID_OR_SLUG",
                   help="limit to these papers (arxiv_id or slug)")
    p.add_argument("--limit", type=int, default=None, help="process at most N papers")
    p.add_argument("--force", action="store_true", help="re-embed + re-index even if up-to-date")
    p.add_argument("--recreate", action="store_true", help="drop and rebuild the collection first")
    p.add_argument("--dry-run", action="store_true",
                   help="parse + chunk only; no model, no Qdrant (offline)")
    p.add_argument("--examples", type=int, default=3, help="how many example chunks to print")
    p.add_argument("--model", default=base.embed.model_name, help="sentence-transformers model")
    p.add_argument("--collection", default=base.qdrant.collection, help="Qdrant collection name")
    p.add_argument("--host", default=base.qdrant.host, help="Qdrant host")
    p.add_argument("--port", type=int, default=base.qdrant.port, help="Qdrant port")
    p.add_argument("--target-tokens", type=int, default=base.chunk.target_tokens,
                   help="target chunk size in tokens")
    p.add_argument("--overlap-tokens", type=int, default=base.chunk.overlap_tokens,
                   help="overlap between chunks in tokens")
    args = p.parse_args(argv)
    _force_utf8_stdout()

    config = _build_config(args)
    result = run_ingest(
        config,
        papers=args.papers,
        limit=args.limit,
        force=args.force,
        recreate=args.recreate,
        do_index=not args.dry_run,
        n_examples=args.examples,
    )

    print("\n" + "-" * 78)
    print(f"Papers processed: {result.papers_processed}  "
          f"skipped(up-to-date): {result.papers_skipped}  missing: {result.papers_missing}")
    print(f"Total chunks: {result.total_chunks}  newly indexed: {result.indexed_chunks}")
    _print_examples(result.examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
