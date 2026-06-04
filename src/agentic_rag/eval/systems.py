"""Adapters that run each system under test and return a uniform result.

Both the single-shot baseline (ADR-0007) and the agent graph (ADR-0008) are
reduced to the same ``SystemResult`` so the metrics code doesn't care which one
produced an answer. Dependencies (retriever / llm / compiled graph) are injected
so the runner can be exercised with fakes (no API, no Qdrant) in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SystemResult:
    """What every system must produce for evaluation."""

    answer: str
    contexts: list[str]  # retrieved chunk texts the answer was grounded in (for RAGAS)
    retrieved_arxiv_ids: list[str]  # arxiv ids in retrieved order (for recall@k / MRR)
    cited_arxiv_ids: list[str] = field(default_factory=list)
    insufficient_context: bool = False
    extra: dict = field(default_factory=dict)  # system-specific (rounds, action, ...)


def _dedup(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class BaselineSystem:
    """Single-shot RAG: retrieve -> generate -> validate (no loop)."""

    name = "baseline"

    def __init__(self, retriever, llm, k: int = 5) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k

    def run(self, question: str) -> SystemResult:
        # Mirrors SingleShotRAG.answer, but keeps the chunks so we can expose
        # contexts + retrieved ids for the metrics (one retrieval, no waste).
        from agentic_rag.answer.prompt import SYSTEM_PROMPT, build_user_prompt
        from agentic_rag.answer.schemas import CitedAnswer
        from agentic_rag.answer.validate import validate_cited_answer

        chunks = self._retriever.retrieve(question, self._k)
        if not chunks:
            return SystemResult(
                answer="No relevant sources were found in the index.",
                contexts=[],
                retrieved_arxiv_ids=[],
                insufficient_context=True,
            )
        raw = self._llm.structured(SYSTEM_PROMPT, build_user_prompt(question, chunks), CitedAnswer)
        validated = validate_cited_answer(question, raw, chunks)
        return SystemResult(
            answer=validated.answer,
            contexts=[c.text for c in chunks],
            retrieved_arxiv_ids=_dedup([c.arxiv_id for c in chunks]),
            cited_arxiv_ids=_dedup([c.arxiv_id for c in validated.citations]),
            insufficient_context=validated.insufficient_context,
        )


class AgentSystem:
    """Agent graph: retrieve -> grade -> generate -> cite-critic -> guardrail."""

    name = "agent"

    def __init__(self, app, config=None) -> None:
        self._app = app
        self._config = config

    def run(self, question: str) -> SystemResult:
        from agentic_rag.agent.graph import run_agent

        final = run_agent(self._app, question, self._config)
        answer = final.get("answer")
        chunks = final.get("chunks") or []
        decision = final.get("guardrail") or {}
        # Surface the guardrail's user-facing text when it declined.
        text = decision.get("final_answer") or (answer.answer if answer else "")
        return SystemResult(
            answer=text,
            contexts=[c.text for c in chunks],
            retrieved_arxiv_ids=_dedup([c.arxiv_id for c in chunks]),
            cited_arxiv_ids=_dedup([c.arxiv_id for c in (answer.citations if answer else [])]),
            insufficient_context=bool(answer.insufficient_context) if answer else True,
            extra={
                "retrieval_rounds": final.get("retrieval_round"),
                "revision_rounds": final.get("revision_round"),
                "guardrail_action": decision.get("action"),
                "confidence": decision.get("confidence"),
            },
        )
