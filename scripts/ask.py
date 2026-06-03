"""Ask a question, get a cited answer (single-shot RAG baseline).

    python scripts/ask.py "How does ELECTRA's objective differ from BERT's?"
    python scripts/ask.py "..." --k 6

Requires: Qdrant running + index built (rag-ingest), and OPENAI_API_KEY in .env.
NOTE: this makes a paid LLM call (gpt-4o-mini; a fraction of a cent per query).
"""

from __future__ import annotations

import argparse
import contextlib
import sys


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8()
    p = argparse.ArgumentParser(description="Single-shot RAG baseline: cited answer to a question.")
    p.add_argument("question", help="the question to answer")
    p.add_argument("--k", type=int, default=5, help="number of chunks to retrieve")
    args = p.parse_args(argv)

    from dotenv import load_dotenv

    from agentic_rag.ingest.config import REPO_ROOT

    # override=True so the project's .env wins over any stale OPENAI_API_KEY
    # already present in the OS environment (a common dev-machine gotcha).
    load_dotenv(REPO_ROOT / ".env", override=True)

    from agentic_rag.answer.baseline import build_baseline

    print(f"Q: {args.question}\nBuilding baseline + calling the LLM ...\n{'-' * 78}")
    rag = build_baseline(k=args.k)
    res = rag.answer(args.question)

    print(res.answer)
    if res.citations:
        print("\nCitations:")
        for c in res.citations:
            print(f"  {c.citation()}")
    flags = f"insufficient_context={res.insufficient_context}  grounded={res.is_grounded}"
    print(f"\n[{flags}]")
    if res.violations:
        print("Grounding violations:")
        for v in res.violations:
            print(f"  - {v}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
