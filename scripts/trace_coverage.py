"""Diagnose multi-hop coverage for a comparison question.

Traces one query through every retrieval stage (dense, BM25, RRF fusion, the
rerank/fusion blend) and reports, for each *expected* paper, the best (shallowest)
rank at which a chunk of that paper appears. This distinguishes two failure modes
for a two-paper comparison stuck at recall 0.50:

  * COVERAGE gap  -- one paper is buried deep (or absent) in the candidate pool,
    because a single embedding of "How does A differ from B?" pulls toward one
    side. Fix: query decomposition / per-side retrieval.
  * RANKING gap   -- both papers are near the top of fusion but the reranker/blend
    pushes the second one just past final_k. Fix: blend tuning, not decomposition.

Usage:
    python scripts/trace_coverage.py q-0001
    python scripts/trace_coverage.py "How does ELECTRA differ from BERT?" \
        --expect 2003.10555 1810.04805

Needs the Qdrant index running (docker compose up -d) and built (rag-ingest).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden_set.jsonl"


def _utf8():
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")


def _load_golden(qid: str) -> dict:
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("id") == qid:
            return row
    raise SystemExit(f"question id {qid!r} not found in {GOLDEN}")


def _first_rank(ranked_ids, by_id, target_arxiv) -> int | None:
    """1-based rank of the first chunk belonging to ``target_arxiv`` (None if absent)."""
    for r, doc_id in enumerate(ranked_ids, start=1):
        chunk = by_id.get(doc_id)
        if chunk is not None and chunk.arxiv_id == target_arxiv:
            return r
    return None


def _fmt(rank, k) -> str:
    if rank is None:
        return "absent"
    flag = "" if rank <= k else "  <-- past final_k"
    return f"rank {rank}{flag}"


def main(argv=None) -> int:
    _utf8()
    p = argparse.ArgumentParser(description="Trace multi-hop coverage through retrieval stages.")
    p.add_argument("query_or_id", help="a golden-set id (q-0001) or a raw query string")
    p.add_argument("--expect", nargs="*", default=None, help="expected arxiv ids (if raw query)")
    p.add_argument("--k", type=int, default=5, help="final_k to judge against")
    p.add_argument(
        "--sub",
        nargs="*",
        default=None,
        help="per-side sub-queries: retrieve each, round-robin merge (simulates the "
        "agent's decomposed re-retrieval) and report coverage instead of single-query.",
    )
    args = p.parse_args(argv)

    if args.query_or_id.startswith("q-") and args.expect is None:
        row = _load_golden(args.query_or_id)
        query = row["question"]
        expected = row["expected_arxiv_ids"]
        print(f"# {row['id']} ({row['type']}): {query}")
    else:
        query = args.query_or_id
        expected = args.expect or []
        print(f"# raw query: {query}")
    print(f"# expected papers: {', '.join(expected)}\n")

    from agentic_rag.retrieve.config import RetrieveConfig
    from agentic_rag.retrieve.fusion import reciprocal_rank_fusion
    from agentic_rag.retrieve.retriever import build_retriever, round_robin_merge

    cfg = RetrieveConfig()
    retriever = build_retriever(retrieve_config=cfg)
    by_id = retriever._by_id  # noqa: SLF001 (diagnostic)

    # --- decomposed mode: per-side retrieve + round-robin merge ---------------
    if args.sub:
        print(f"# decomposed into {len(args.sub)} sub-queries: {args.sub}\n")
        per_side = [retriever.retrieve(q, args.k) for q in args.sub]
        for q, side in zip(args.sub, per_side, strict=True):
            print(f"  [{q}] -> {sorted({c.arxiv_id for c in side})}")
        merged = round_robin_merge(per_side, args.k)
        covered = {c.arxiv_id for c in merged} & set(expected)
        print(f"\nmerged top-{args.k} papers: {sorted({c.arxiv_id for c in merged})}")
        print(
            f"recall@{args.k}: {len(covered)}/{len(expected)} = {len(covered) / len(expected):.3f}"
        )
        print(f"\n--- merged top-{args.k} chunks ---")
        for i, c in enumerate(merged, start=1):
            marker = "*" if c.arxiv_id in expected else " "
            print(f"{marker}{i}. [{c.arxiv_id}] {c.title[:40]} §{c.section}")
        return 0

    # --- recompute each stage's ranking explicitly (mirrors retriever.retrieve) ---
    dense = retriever._dense.search(query, cfg.dense_candidates)  # noqa: SLF001
    bm25 = retriever._bm25.search(query, cfg.bm25_candidates)  # noqa: SLF001
    dense_ids = [d for d, _ in dense]
    bm25_ids = [d for d, _ in bm25]
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=cfg.rrf_k)
    fused_ids = [d for d, _ in fused]

    final = retriever.retrieve(query, k=cfg.dense_candidates)  # full blended order
    final_ids = [c.id for c in final]

    stages = [
        ("dense", dense_ids),
        ("bm25", bm25_ids),
        ("fusion (RRF)", fused_ids),
        ("rerank-blend (final)", final_ids),
    ]

    # title lookup per expected paper
    titles = {}
    for arx in expected:
        for c in by_id.values():
            if c.arxiv_id == arx:
                titles[arx] = c.title
                break

    print(f"{'stage':<24}" + "".join(f"{titles.get(a, a)[:26]:<28}" for a in expected))
    print("-" * (24 + 28 * len(expected)))
    for name, ids in stages:
        cells = "".join(f"{_fmt(_first_rank(ids, by_id, a), args.k):<28}" for a in expected)
        print(f"{name:<24}{cells}")

    # --- verdict on the final top-k ---
    topk = final[: args.k]
    covered = {c.arxiv_id for c in topk} & set(expected)
    print(f"\ntop-{args.k} papers: {sorted({c.arxiv_id for c in topk})}")
    print(f"recall@{args.k}: {len(covered)}/{len(expected)} = {len(covered) / len(expected):.3f}")

    print(f"\n--- top-{args.k} chunks (final blended order) ---")
    for i, c in enumerate(topk, start=1):
        marker = "*" if c.arxiv_id in expected else " "
        print(
            f"{marker}{i}. [{c.arxiv_id}] {c.title[:40]} §{c.section} (d#{c.dense_rank} b#{c.bm25_rank})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
