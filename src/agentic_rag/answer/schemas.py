"""Pydantic schemas for the structured LLM response.

These are the *contract* the LLM must fill (via OpenAI Structured Outputs). No
defaults / all fields required — strict structured-output mode wants every field
present, and it keeps the model honest (it must explicitly decide
`insufficient_context`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One source the model used. `source_id` is the [S#] label from the context."""

    source_id: str = Field(description="The source's bracket label exactly as shown, e.g. 'S1'.")
    arxiv_id: str = Field(description="arXiv id of the cited paper, copied from the source header.")
    section: str = Field(description="Section of the cited source, copied from the source header.")
    page: int = Field(description="Page number of the cited source, copied from the source header.")


class CitedAnswer(BaseModel):
    """Structured single-shot RAG output."""

    answer: str = Field(
        description="Answer using ONLY the provided sources, with an inline [S#] marker after "
        "each factual claim. If insufficient_context is true, a single sentence saying so."
    )
    citations: list[Citation] = Field(
        description="Every source actually used. Empty when insufficient_context is true."
    )
    insufficient_context: bool = Field(
        description="True if the sources do not contain enough information to answer the question."
    )
