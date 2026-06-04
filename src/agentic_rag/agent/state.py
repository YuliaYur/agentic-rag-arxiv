"""Explicit, typed graph state + the node-output schemas.

The state is a TypedDict (LangGraph's native state type). `trace` uses an
``operator.add`` reducer so each node can append its own structured metadata
entry without clobbering earlier ones — that ordered trace is what Step 6
(Langfuse) will visualize.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from agentic_rag.answer.schemas import CitedAnswer
from agentic_rag.answer.validate import ValidatedAnswer
from agentic_rag.retrieve.models import RetrievedChunk


class GradeResult(BaseModel):
    """grade_context output: is the retrieved context good enough to answer?"""

    sufficient: bool = Field(
        description="True only if the sources contain enough to answer the question well."
    )
    reasoning: str = Field(description="One or two sentences justifying the judgment.")
    refined_query: str = Field(
        description="If not sufficient, a better search query (add specific terms/entities/"
        "sub-questions) to retrieve the missing context. If sufficient, repeat the question."
    )


class CriticResult(BaseModel):
    """cite_critic output: is every claim in the answer supported by a cited source?"""

    supported: bool = Field(
        description="True only if every factual claim is backed by one of the cited sources."
    )
    score: float = Field(description="Fraction of claims supported by cited sources, 0.0-1.0.")
    unsupported_claims: list[str] = Field(
        description="Claims not backed by any cited source; empty if all are supported."
    )
    feedback: str = Field(
        description="How to fix the answer (drop/qualify/recite). Empty string if supported."
    )


class AgentState(TypedDict, total=False):
    """The graph's working memory. `total=False` so nodes return partial updates."""

    # Inputs / query
    original_question: str  # the user's question (what `generate` answers)
    question: str  # current retrieval query (may be reformulated)
    k: int

    # Retrieval
    chunks: list[RetrievedChunk]
    retrieval_round: int

    # Grading
    grade: dict[str, Any]  # GradeResult.model_dump()

    # Generation
    draft: CitedAnswer | None
    validated: ValidatedAnswer | None
    answer: ValidatedAnswer | None  # latest validated answer (the final result)
    revision_round: int

    # Critique
    critic: dict[str, Any] | None  # CriticResult.model_dump()

    # Guardrails (output layer's verdict; input-layer hits live in the trace)
    guardrail: dict[str, Any] | None  # GuardrailDecision.model_dump()

    # Caps (kept in state so routing functions stay pure)
    max_retrieval_rounds: int
    max_revision_rounds: int

    # Ordered, structured per-node metadata for tracing (Step 6).
    trace: Annotated[list[dict[str, Any]], operator.add]
