"""Before/after benchmark: LLM model routing + semantic caching.

Runs a small query set through the agent under three configs and prints a
cost + latency (p50/p95) + cache-hit table:

  A  uniform-strong   strong model everywhere, no cache      (the naive baseline)
  B  routed           strong synthesis + cheap grade/critic, no cache
  C  routed + cache   same as B with the semantic cache on; measured on a WARM pass
                      (a cold pass populates the cache first)

Reuses one retriever across configs (models/index load once). Needs OPENAI_API_KEY;
config C needs Redis (`docker compose up -d redis`). Makes real calls at the strong
model's price, so keep the set small.

    python scripts/bench_routing_cache.py
    python scripts/bench_routing_cache.py --strong gpt-4o --cheap gpt-4o-mini -n 2
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import sys
import time

DEFAULT_QUERIES = [
    "How does ELECTRA's pre-training objective differ from BERT's masked language modeling?",
    "What changes did RoBERTa make to BERT's pre-training recipe?",
    "What optimizer and learning-rate schedule does the original Transformer use?",
]


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")


def _run_set(app, run_agent, queries):
    """Run each query once; return per-query (wall_ms, cost, n_calls, cache_hits)."""
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        final = run_agent(app, q)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        m = final.get("metering") or {}
        rows.append(
            {
                "wall_ms": wall_ms,
                "cost": m.get("cost_usd", 0.0),
                "calls": m.get("n_calls", 0),
                "hits": m.get("cache_hits", 0),
            }
        )
    return rows


def _summary(name, rows):
    from agentic_rag.llm.metering import percentile

    n = len(rows)
    cost = sum(r["cost"] for r in rows)
    calls = sum(r["calls"] for r in rows)
    hits = sum(r["hits"] for r in rows)
    walls = [r["wall_ms"] for r in rows]
    return {
        "name": name,
        "queries": n,
        "cost_total": cost,
        "cost_per_query": cost / n if n else 0.0,
        "p50_ms": percentile(walls, 50),
        "p95_ms": percentile(walls, 95),
        "hit_rate": hits / calls if calls else 0.0,
    }


def main(argv=None) -> int:
    _utf8()
    p = argparse.ArgumentParser(description="Benchmark routing + semantic caching.")
    p.add_argument("--strong", default="gpt-4o", help="strong model (synthesis / baseline)")
    p.add_argument("--cheap", default="gpt-4o-mini", help="cheap model (grade + critic)")
    p.add_argument("-n", "--num", type=int, default=len(DEFAULT_QUERIES), help="how many queries")
    args = p.parse_args(argv)

    from dotenv import load_dotenv

    from agentic_rag.ingest.config import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env", override=True)

    from agentic_rag.agent.config import AgentConfig
    from agentic_rag.agent.graph import build_graph, run_agent
    from agentic_rag.guardrails import Guardrails
    from agentic_rag.llm.cache import CacheConfig, configure_cache
    from agentic_rag.llm.client import LLMClient
    from agentic_rag.llm.config import LLMConfig
    from agentic_rag.retrieve.retriever import build_retriever

    queries = DEFAULT_QUERIES[: args.num]
    print(f"Loading retriever (shared) ... evaluating {len(queries)} queries.")
    retriever = build_retriever()

    def make_app(llm_config):
        return build_graph(retriever, LLMClient(llm_config), AgentConfig(), Guardrails())

    results = []

    # A — uniform strong, no cache
    configure_cache(CacheConfig(enabled=False))
    print(f"\n[A] uniform-strong ({args.strong}), no cache ...")
    results.append(
        _summary(
            "A uniform-strong (no cache)",
            _run_set(make_app(LLMConfig.uniform(args.strong)), run_agent, queries),
        )
    )

    # B — routed, no cache
    routed = LLMConfig.routed(synthesis=args.strong, cheap=args.cheap)
    print(f"[B] routed ({args.strong} synth + {args.cheap} grade/critic), no cache ...")
    results.append(_summary("B routed (no cache)", _run_set(make_app(routed), run_agent, queries)))

    # C — routed + semantic cache (cold pass to populate, measure the warm pass)
    cache_cfg = dataclasses.replace(CacheConfig.from_env(), enabled=True)
    on = configure_cache(cache_cfg)
    if on:
        app_c = make_app(routed)
        print("[C] routed + cache: cold pass (populating cache) ...")
        _run_set(app_c, run_agent, queries)
        print("[C] routed + cache: warm pass (measured) ...")
        results.append(_summary("C routed + cache (warm)", _run_set(app_c, run_agent, queries)))
        configure_cache(CacheConfig(enabled=False))
    else:
        print("[C] SKIPPED — semantic cache unavailable (start Redis: docker compose up -d redis).")

    # --- table ---
    print("\n## Routing + caching: before/after\n")
    print(
        "| Config | Queries | Total cost | $/query | p50 latency | p95 latency | Cache hit-rate |"
    )
    print("|---|---|---|---|---|---|---|")
    base = results[0]["cost_per_query"] or 1e-9
    for r in results:
        delta = f" ({(r['cost_per_query'] / base - 1) * 100:+.0f}%)" if r is not results[0] else ""
        print(
            f"| {r['name']} | {r['queries']} | ${r['cost_total']:.4f} | "
            f"${r['cost_per_query']:.4f}{delta} | {r['p50_ms'] / 1000:.2f}s | "
            f"{r['p95_ms'] / 1000:.2f}s | {r['hit_rate'] * 100:.0f}% |"
        )
    print("\n_Cost from LiteLLM's per-call accounting; latency is end-to-end wall time per query._")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
