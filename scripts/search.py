"""Run a hybrid-retrieval query from the CLI and see ranked results + sources.

    python scripts/search.py "how is the quadratic cost of attention reduced?"
    python scripts/search.py "GLUE benchmark" --k 8
    python scripts/search.py "LoRA rank" --no-rerank        # see fusion-only order
    python scripts/search.py "..." --compare                # dense-only vs hybrid

Needs the Qdrant index running (docker compose up -d) and built (rag-ingest).
"""

from __future__ import annotations

import argparse
import sys
import textwrap


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _print_hits(title, hits, scored=True):
    print(f"\n{title}\n" + "-" * 78)
    for i, h in enumerate(hits, start=1):
        if scored:
            rr = f"  rerank={h.rerank_score:.3f}" if h.rerank_score is not None else ""
            prov = f"  (dense#{h.dense_rank or '-'} bm25#{h.bm25_rank or '-'})"
            head = f"{i}. [rrf={h.score:.4f}{rr}]{prov}  {h.citation()}"
        else:
            head = f"{i}. {h.citation()}"
        print(head)
        snippet = " ".join(h.text.split())[:200]
        print(textwrap.fill(snippet + "...", width=74, initial_indent="     ", subsequent_indent="     "))


def main(argv=None) -> int:
    _utf8()
    p = argparse.ArgumentParser(description="Hybrid retrieval over the arXiv index.")
    p.add_argument("query", help="the search query")
    p.add_argument("--k", type=int, default=5, help="number of results to return")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rerank")
    p.add_argument("--compare", action="store_true",
                   help="also show dense-only results, to contrast with hybrid")
    args = p.parse_args(argv)

    from agentic_rag.retrieve.config import RetrieveConfig
    from agentic_rag.retrieve.retriever import build_retriever

    cfg = RetrieveConfig(use_reranker=not args.no_rerank)
    print(f'Building retriever (query: "{args.query}") ...')
    retriever = build_retriever(retrieve_config=cfg)

    if args.compare:
        # Dense-only: ask the dense searcher directly, resolve ids -> chunks.
        dense_hits = retriever._dense.search(args.query, args.k)  # noqa: SLF001 (demo)
        dense_chunks = [retriever._by_id[i] for i, _ in dense_hits]  # noqa: SLF001
        _print_hits(f"DENSE-ONLY (top {args.k})", dense_chunks, scored=False)

    hits = retriever.retrieve(args.query, k=args.k)
    label = "HYBRID + RERANK" if not args.no_rerank else "HYBRID (fusion only)"
    _print_hits(f"{label} (top {args.k})", hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
