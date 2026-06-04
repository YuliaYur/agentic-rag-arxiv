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
            print(
                f"  {i}. retrieve      round={e['retrieval_round']}  n={e['n_chunks']}  q={e['query'][:60]!r}"
            )
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


def main(argv: list[str] | None = None) -> int:
    _utf8()
    p = argparse.ArgumentParser(
        description="Agentic RAG: retrieve -> grade -> generate -> cite-critic."
    )
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-retrieval-rounds", type=int, default=3)
    p.add_argument("--max-revision-rounds", type=int, default=2)
    args = p.parse_args(argv)

    from dotenv import load_dotenv

    from agentic_rag.ingest.config import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env", override=True)

    from agentic_rag.agent.config import AgentConfig
    from agentic_rag.agent.graph import build_agent, run_agent

    cfg = AgentConfig(
        k=args.k,
        max_retrieval_rounds=args.max_retrieval_rounds,
        max_revision_rounds=args.max_revision_rounds,
    )
    print(f"Q: {args.question}\nRunning agent graph ...\n{'-' * 78}")
    app = build_agent(cfg)
    final = run_agent(app, args.question, cfg)

    _print_trace(final["trace"])
    ans = final["answer"]
    print("-" * 78)
    print(ans.answer)
    if ans.citations:
        print("\nCitations:")
        for c in ans.citations:
            print(f"  {c.citation()}")
    print(
        f"\n[retrieval_rounds={final['retrieval_round']}  revisions={final['revision_round']}  "
        f"grounded={ans.is_grounded}  insufficient_context={ans.insufficient_context}]"
    )
    return 0 if ans.is_grounded else 1


if __name__ == "__main__":
    raise SystemExit(main())
