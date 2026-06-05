"""Run the full agentic answer graph on a question.

    python scripts/agent_ask.py "How does ELECTRA's objective differ from BERT and RoBERTa?"
    python scripts/agent_ask.py "..." --k 6 --max-retrieval-rounds 3 --max-revision-rounds 2

Needs Qdrant + index + OPENAI_API_KEY in .env. Makes SEVERAL paid LLM calls
(grade + generate + critic, times any loops) — still fractions of a cent on gpt-4o-mini.
"""

from __future__ import annotations

import argparse
import contextlib
import sys


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")


def _print_trace(trace: list[dict]) -> None:
    print("CONTROL FLOW (per node):")
    for i, e in enumerate(trace, start=1):
        node = e["node"]
        if node == "retrieve":
            n_inj = len(e.get("injection_hits", []))
            flag = f"  injection_hits={n_inj}" if n_inj else ""
            q = e["query"]
            qstr = " + ".join(f"{s!r}" for s in q) if e.get("decomposed") else repr(q[:60])
            tag = " [decomposed]" if e.get("decomposed") else ""
            print(
                f"  {i}. retrieve      round={e['retrieval_round']}{tag}  n={e['n_chunks']}  papers={e.get('papers', [])}"
            )
            print(f"        q={qstr}{flag}")
            for h in e.get("injection_hits", []):
                print(f"        ! {h['source_id']} [{h['pattern']}] {h['snippet']!r}")
        elif node == "grade_context":
            print(
                f"  {i}. grade_context sufficient={e['sufficient']}  -> refined={e['refined_query'][:50]!r}"
            )
        elif node == "generate":
            tag = "revise" if e["is_revision"] else "first"
            print(f"  {i}. generate({tag})  grounded={e['grounded']}  citations={e['n_citations']}")
        elif node == "cite_critic":
            print(
                f"  {i}. cite_critic   supported={e['supported']}  score={e['critic_score']:.2f}  unsupported={e['n_unsupported']}"
            )
        elif node == "output_guard":
            failed = e.get("failed_checks") or []
            tail = f"  failed={failed}" if failed else ""
            print(
                f"  {i}. output_guard  action={e['action']}  reason={e['reason']}  conf={e['confidence']:.2f}{tail}"
            )


def main(argv: list[str] | None = None) -> int:
    _utf8()
    p = argparse.ArgumentParser(
        description="Agentic RAG: retrieve -> grade -> generate -> cite-critic."
    )
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-retrieval-rounds", type=int, default=3)
    p.add_argument("--max-revision-rounds", type=int, default=2)
    p.add_argument("--min-confidence", type=float, default=0.5, help="decline below this (0-1)")
    p.add_argument(
        "--no-scan-injection", action="store_true", help="disable the input injection guardrail"
    )
    p.add_argument(
        "--flag-only-injection",
        action="store_true",
        help="detect+log injection but don't redact (pass chunk text through)",
    )
    trace_grp = p.add_mutually_exclusive_group()
    trace_grp.add_argument(
        "--trace", action="store_true", help="force Langfuse tracing on (overrides env)"
    )
    trace_grp.add_argument(
        "--no-trace", action="store_true", help="force Langfuse tracing off (overrides env)"
    )
    args = p.parse_args(argv)

    import os

    from dotenv import load_dotenv

    from agentic_rag.ingest.config import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env", override=True)
    # CLI flag overrides the env toggle; read lazily by the global tracer below.
    if args.trace:
        os.environ["LANGFUSE_TRACING"] = "true"
    elif args.no_trace:
        os.environ["LANGFUSE_TRACING"] = "false"

    from agentic_rag.agent.config import AgentConfig
    from agentic_rag.agent.graph import build_agent, run_agent
    from agentic_rag.guardrails import Guardrails, GuardrailsConfig
    from agentic_rag.llm.cache import CacheConfig, configure_cache

    # Install the semantic cache if LLM_CACHE_ENABLED (no-op + safe if off/unreachable).
    cache_on = configure_cache(CacheConfig.from_env())

    cfg = AgentConfig(
        k=args.k,
        max_retrieval_rounds=args.max_retrieval_rounds,
        max_revision_rounds=args.max_revision_rounds,
    )
    guards = Guardrails(
        GuardrailsConfig(
            scan_injection=not args.no_scan_injection,
            neutralize_injection=not args.flag_only_injection,
            min_confidence=args.min_confidence,
        )
    )
    print(f"Q: {args.question}\nRunning agent graph ...\n{'-' * 78}")
    app = build_agent(cfg, guards)
    final = run_agent(app, args.question, cfg)

    _print_trace(final["trace"])
    ans = final["answer"]
    decision = final.get("guardrail") or {}
    print("-" * 78)
    # Show what the guardrail decided to surface (the answer, or a decline message).
    print(decision.get("final_answer", ans.answer))
    if decision.get("action") == "answer" and ans.citations:
        print("\nCitations:")
        for c in ans.citations:
            print(f"  {c.citation()}")
    m = final.get("metering") or {}
    if m:
        print(
            f"\n[cost=${m.get('cost_usd', 0.0):.5f}  llm_calls={m.get('n_calls', 0)}  "
            f"cache_hits={m.get('cache_hits', 0)}{' (cache on)' if cache_on else ''}  "
            f"llm_latency={m.get('latency_ms_total', 0.0) / 1000:.2f}s "
            f"(p50={m.get('p50_ms', 0.0):.0f}ms p95={m.get('p95_ms', 0.0):.0f}ms)]"
        )
    print(
        f"\n[guardrail={decision.get('action', '?')} ({decision.get('reason', '?')})  "
        f"confidence={decision.get('confidence', 0.0):.2f}  "
        f"retrieval_rounds={final['retrieval_round']}  revisions={final['revision_round']}  "
        f"grounded={ans.is_grounded}]"
    )

    from agentic_rag.observability import get_tracer

    tracer = get_tracer()
    if tracer.enabled:
        url = tracer.last_trace_url
        print(f"\nLangfuse trace: {url}" if url else "\nLangfuse trace recorded (open the UI).")
    return 0 if decision.get("action") == "answer" else 1


if __name__ == "__main__":
    raise SystemExit(main())
