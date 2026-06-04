"""The guardrails facade the graph holds onto.

One object that carries the config and exposes the two layers as methods, so
``AgentNodes`` depends on a single injected collaborator (and tests can pass a
``Guardrails`` built from any config).
"""

from __future__ import annotations

from agentic_rag.answer.validate import ValidatedAnswer
from agentic_rag.retrieve.models import RetrievedChunk

from .config import GuardrailsConfig
from .injection import sanitize_chunks
from .models import GuardrailDecision, InjectionHit
from .output import check_output


class Guardrails:
    def __init__(self, config: GuardrailsConfig | None = None) -> None:
        self.config = config or GuardrailsConfig()

    # input ------------------------------------------------------------------
    def sanitize_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], list[InjectionHit]]:
        """Run the input (prompt-injection) guardrail over retrieved chunks."""
        if not self.config.scan_injection:
            return chunks, []
        return sanitize_chunks(chunks, neutralize=self.config.neutralize_injection)

    # output -----------------------------------------------------------------
    def check_output(
        self, validated: ValidatedAnswer | None, critic: dict | None
    ) -> GuardrailDecision:
        """Run the output guardrail (structure / abstain / grounding / confidence)."""
        return check_output(validated, critic, self.config)
