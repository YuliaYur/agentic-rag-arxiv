"""The agent, wrapped for the API: build once, run serialized, map to the schema.

``AgentService`` owns the compiled LangGraph agent and turns one ``QueryRequest``
into a ``QueryResponse`` by running ``run_agent`` and translating the final state.
Calls are serialized with a lock because the embedder/reranker are not safe to call
concurrently from the request threadpool — an honest single-instance posture (scale
= more workers/replicas behind the rate limit).
"""

from __future__ import annotations

import dataclasses
import threading

from .schemas import CitationOut, MeteringOut, QueryRequest, QueryResponse, StepOut


def _arxiv_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def _summarize_step(entry: dict) -> StepOut:
    """One-line human summary of a node's trace entry (for the reasoning panel)."""
    node = entry.get("node", "?")
    if node == "retrieve":
        deco = " · decomposed" if entry.get("decomposed") else ""
        papers = entry.get("papers", []) or []
        summary = (
            f"round {entry.get('retrieval_round')}: retrieved {entry.get('n_chunks', 0)} chunks "
            f"from {len(papers)} paper(s){deco}"
        )
    elif node == "grade_context":
        if entry.get("sufficient"):
            summary = "context sufficient → generate"
        else:
            missing = entry.get("missing_papers") or []
            tail = f" (missing {', '.join(missing)})" if missing else ""
            summary = f"context insufficient → re-retrieve{tail}"
    elif node == "generate":
        verb = "revised" if entry.get("is_revision") else "drafted"
        summary = (
            f"{verb} answer · grounded={entry.get('grounded')} · "
            f"{entry.get('n_citations', 0)} citations"
        )
    elif node == "cite_critic":
        summary = (
            f"citation audit · supported={entry.get('supported')} · "
            f"score={entry.get('critic_score', 0.0):.2f}"
        )
    elif node == "output_guard":
        summary = (
            f"guardrail → {entry.get('action')} · confidence={entry.get('confidence', 0.0):.2f}"
        )
    else:
        summary = node
    return StepOut(node=node, summary=summary, detail=entry)


def to_response(question: str, final: dict) -> QueryResponse:
    """Translate the agent's final state into the public response shape."""
    guard = final.get("guardrail") or {}
    answer = final.get("answer")

    citations = [
        CitationOut(
            source_id=c.source_id,
            arxiv_id=c.arxiv_id,
            title=c.title,
            section=c.section,
            page=c.page,
            page_end=c.page_end,
            label=c.citation(),
            url=_arxiv_url(c.arxiv_id),
        )
        for c in (answer.citations if answer is not None else [])
    ]
    steps = [_summarize_step(e) for e in final.get("trace", [])]
    m = final.get("metering") or {}
    metering = MeteringOut(
        cost_usd=round(m.get("cost_usd", 0.0), 6),
        llm_calls=m.get("n_calls", 0),
        cache_hits=m.get("cache_hits", 0),
        latency_ms=round(m.get("latency_ms_total", 0.0), 1),
    )
    return QueryResponse(
        question=question,
        action=guard.get("action", "answer"),
        answer=guard.get("final_answer") or (answer.answer if answer is not None else ""),
        confidence=guard.get("confidence", 0.0),
        grounded=bool(answer.is_grounded) if answer is not None else False,
        insufficient_context=bool(answer.insufficient_context) if answer is not None else False,
        citations=citations,
        steps=steps,
        metering=metering,
        retrieval_rounds=final.get("retrieval_round", 0),
        revision_rounds=final.get("revision_round", 0),
    )


class AgentService:
    def __init__(self, app, run_agent, default_config) -> None:
        self._app = app
        self._run = run_agent
        self._default = default_config
        self._lock = threading.Lock()

    def query(self, req: QueryRequest) -> QueryResponse:
        cfg = self._default
        overrides = {
            field: value
            for field, value in (
                ("k", req.k),
                ("max_retrieval_rounds", req.max_retrieval_rounds),
                ("max_revision_rounds", req.max_revision_rounds),
            )
            if value is not None
        }
        if overrides:
            cfg = dataclasses.replace(self._default, **overrides)
        # Serialize: the embedder/reranker aren't safe for concurrent calls.
        with self._lock:
            final = self._run(self._app, req.question, cfg)
        return to_response(req.question, final)


def build_service() -> AgentService:
    """Wire the real agent (heavy: loads models + index + LLM client). Build once."""
    from agentic_rag.agent.config import AgentConfig
    from agentic_rag.agent.graph import build_agent, run_agent

    return AgentService(build_agent(), run_agent, AgentConfig())
