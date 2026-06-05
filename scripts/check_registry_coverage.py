"""Memory-free triage: does the corpus name registry recognize the papers each
golden question NAMES? Pure Python (no models, no Qdrant) — safe on a RAM-starved
box. For multi-paper questions, the deterministic decomposition gate can only fire
for papers it detects, so a named-but-undetected expected paper is a registry gap.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _utf8():
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _utf8()
    from agentic_rag.agent.corpus import CORPUS_PAPER_NAMES
    from agentic_rag.retrieve.retriever import detect_named_papers

    rows = [
        json.loads(line)
        for line in (ROOT / "eval" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    gaps = 0
    print(f"{'id':7} {'status':7} {'expected':24} detected  -> verdict")
    print("-" * 78)
    for r in rows:
        expected = set(r["expected_arxiv_ids"])
        detected = detect_named_papers(r["question"], CORPUS_PAPER_NAMES)
        missing = expected - detected  # expected papers the registry did NOT name
        multi = r["type"] in {"comparative", "multi-hop"} or len(expected) > 1
        # The gate only fires for multi-paper questions; a miss there is a real gap.
        verdict = "ok"
        if multi and missing:
            verdict = f"GAP: registry misses {sorted(missing)}"
            gaps += 1
        elif missing:
            verdict = f"(single-hop; not named: {sorted(missing)})"
        print(
            f"{r['id']:7} {r['status']:7} {','.join(sorted(expected)):24} "
            f"{','.join(sorted(detected)) or '-':24} -> {verdict}"
        )
    print("-" * 78)
    print(f"multi-paper registry gaps: {gaps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
