"""Single-shot RAG baseline: retrieve -> stuff context -> generate -> validate.

No agent, no re-retrieval loop, no grading — deliberately. This is the measurable
baseline the agent will be compared against on the eval set. The retriever and LLM
are injected so the orchestration is testable with fakes (no API calls).
"""

from __future__ import annotations

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .schemas import CitedAnswer
from .validate import ValidatedAnswer, validate_cited_answer


class SingleShotRAG:
    def __init__(self, retriever, llm, k: int = 5) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k

    def answer(self, question: str, k: int | None = None) -> ValidatedAnswer:
        chunks = self._retriever.retrieve(question, k or self._k)
        if not chunks:
            # Nothing retrieved -> abstain without spending an LLM call.
            return ValidatedAnswer(
                question=question,
                answer="No relevant sources were found in the index.",
                insufficient_context=True,
            )
        user = build_user_prompt(question, chunks)
        raw: CitedAnswer = self._llm.structured(SYSTEM_PROMPT, user, CitedAnswer)
        return validate_cited_answer(question, raw, chunks)


def build_baseline(k: int = 5) -> SingleShotRAG:
    """Wire the real retriever + LLM client. Heavy (loads models + index); reuse it."""
    from agentic_rag.llm.client import LLMClient
    from agentic_rag.retrieve.retriever import build_retriever

    return SingleShotRAG(retriever=build_retriever(), llm=LLMClient(), k=k)
