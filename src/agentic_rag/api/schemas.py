"""Request/response contract for the API — the clean, typed surface callers see.

Deliberately decoupled from the internal ``ValidatedAnswer`` / ``GuardrailDecision``
/ graph-``trace`` types: those can evolve without breaking the public shape, and the
mapping lives in ``service.py``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="The user's question.")
    # Optional per-request overrides of the agent's defaults.
    k: int | None = Field(default=None, ge=1, le=20, description="Chunks retrieved per round.")
    max_retrieval_rounds: int | None = Field(default=None, ge=1, le=5)
    max_revision_rounds: int | None = Field(default=None, ge=0, le=5)


class CitationOut(BaseModel):
    source_id: str
    arxiv_id: str
    title: str
    section: str
    page: int
    page_end: int
    label: str  # human-readable "[S1] Title (arXiv:..) §.. p.." line
    url: str  # arxiv abstract URL


class StepOut(BaseModel):
    """One agent node's contribution, for the 'how it got here' panel."""

    node: str
    summary: str  # one-line human summary
    detail: dict  # the raw trace entry (already JSON-safe)


class MeteringOut(BaseModel):
    cost_usd: float
    llm_calls: int
    cache_hits: int
    latency_ms: float


class QueryResponse(BaseModel):
    question: str
    action: str  # "answer" | "decline"
    answer: str  # the text to show (the answer, or a decline message)
    confidence: float
    grounded: bool
    insufficient_context: bool
    citations: list[CitationOut]
    steps: list[StepOut]
    metering: MeteringOut
    retrieval_rounds: int
    revision_rounds: int


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
