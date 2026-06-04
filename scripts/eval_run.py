"""Run the evaluation suite against the baseline AND the agent; save a report.

    python scripts/eval_run.py                       # full golden set
    python scripts/eval_run.py --status seed          # only curated 'seed' rows
    python scripts/eval_run.py --limit 6              # first 6 questions (cheap smoke)
    python scripts/eval_run.py --no-ragas --no-judge  # retrieval metrics only (cheapest)

Needs Qdrant + the index + OPENAI_API_KEY in .env. Makes MANY paid LLM calls
(each system answers, then ~7 metric/judge calls per question per system), so
start with --limit / --status while curating. Writes JSON + Markdown to
eval/results/ and prints the comparison table.
"""

from __future__ import annotations

# Windows + torch/sentence-transformers can segfault on a duplicate OpenMP runtime
# when other native libs are loaded; these are the standard workarounds. setdefault
# so the caller can still override them from the environment.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import contextlib
import json
import sys
from datetime import datetime


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8()
    p = argparse.ArgumentParser(description="Evaluate baseline vs agent on the golden set.")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    p.add_argument("--status", default=None, help="filter by status (e.g. seed, reviewed, draft)")
    p.add_argument("--k", type=int, default=5, help="retrieval depth + recall@k")
    p.add_argument("--no-ragas", action="store_true", help="skip RAGAS-style metrics")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM-judge")
    p.add_argument(
        "--systems", default="baseline,agent", help="comma list: baseline,agent (default both)"
    )
    args = p.parse_args(argv)

    from dotenv import load_dotenv

    from agentic_rag.ingest.config import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env", override=True)

    from agentic_rag.agent.config import AgentConfig
    from agentic_rag.agent.graph import build_graph
    from agentic_rag.eval.dataset import load_golden_set
    from agentic_rag.eval.report import render_markdown
    from agentic_rag.eval.runner import run_eval
    from agentic_rag.eval.systems import AgentSystem, BaselineSystem
    from agentic_rag.llm.client import LLMClient
    from agentic_rag.retrieve.retriever import build_retriever

    items = load_golden_set()
    if args.status:
        items = [it for it in items if it.status == args.status]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No questions matched the filter.", file=sys.stderr)
        return 2

    wanted = [s.strip() for s in args.systems.split(",") if s.strip()]
    print(f"Loading retriever + index ... (evaluating {len(items)} questions: {wanted})")
    retriever = build_retriever()  # shared by both systems (load models/index once)
    llm = LLMClient()
    cfg = AgentConfig(k=args.k)

    systems = {}
    if "baseline" in wanted:
        systems["baseline"] = BaselineSystem(retriever, llm, k=args.k)
    if "agent" in wanted:
        app = build_graph(retriever, llm, cfg)
        systems["agent"] = AgentSystem(app, cfg)

    metric_llm = LLMClient()  # separate client for metrics + judge

    def progress(i, total, qid):
        print(f"  [{i}/{total}] {qid}", flush=True)

    report = run_eval(
        items,
        systems,
        metric_llm,
        k=args.k,
        use_ragas=not args.no_ragas,
        use_judge=not args.no_judge,
        progress=progress,
    )

    # --- save + print ---
    results_dir = REPO_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report["generated_at"] = stamp
    md = render_markdown(report, title=f"Evaluation results ({stamp})")
    (results_dir / f"{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (results_dir / f"{stamp}.md").write_text(md, encoding="utf-8")
    # Stable 'latest' pointers for easy reference / diffing.
    (results_dir / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (results_dir / "latest.md").write_text(md, encoding="utf-8")

    print("\n" + md)
    print(f"\nSaved: eval/results/{stamp}.json + .md  (and latest.*)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
