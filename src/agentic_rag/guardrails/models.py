"""Result types produced by the guardrails layer.

Pydantic models (like ``GradeResult``/``CriticResult``) so they serialize cleanly
into the graph ``trace`` and the final state for logging / Langfuse later.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InjectionHit(BaseModel):
    """One suspected prompt-injection span found in a retrieved chunk."""

    source_id: str = Field(description="The [S#] label of the chunk the hit was found in.")
    pattern: str = Field(description="Name of the rule that matched.")
    snippet: str = Field(description="The matched text (truncated), for the log/trace.")


class CheckResult(BaseModel):
    """One named output check and whether it passed."""

    name: str
    passed: bool
    detail: str = ""


class GuardrailDecision(BaseModel):
    """The output guardrail's verdict on a finished answer."""

    action: Literal["answer", "decline"] = Field(
        description="'answer' to surface the generated answer; 'decline' to refuse."
    )
    reason: str = Field(description="Machine-readable reason key for the decision.")
    confidence: float = Field(description="0.0-1.0 confidence used for the threshold gate.")
    checks: list[CheckResult] = Field(description="Every check that was run, in order.")
    final_answer: str = Field(description="The text to show the user (the answer, or a decline).")
