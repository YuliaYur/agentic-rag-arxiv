"""Guardrails configuration — one source of truth, overridable by callers.

Every guardrail behaviour is a knob here so it can be tuned (or switched off for
A/B-ing against the eval set) without touching the graph. Defaults are the
production-grade posture: scan AND neutralize injection, and decline rather than
emit a low-confidence or ungrounded answer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailsConfig:
    # --- input: prompt-injection in retrieved chunks -------------------------
    scan_injection: bool = True
    # When True, matched instruction-like spans are redacted from the chunk text
    # before it reaches any prompt. When False we only *flag* (log) and pass the
    # text through unchanged — useful to measure detection without altering inputs.
    neutralize_injection: bool = True

    # --- output: structure + abstain + confidence ----------------------------
    enforce_structure: bool = True
    # Honour the "refuse if context insufficient" rule: if generation set
    # insufficient_context, the layer declines instead of surfacing a non-answer
    # as if it were a result.
    refuse_on_insufficient_context: bool = True
    # Decline if the validated answer carries grounding violations.
    require_grounded: bool = True
    # Confidence below this -> decline. Confidence is the citation critic's
    # fraction-of-claims-supported score, gated to 0 when the answer isn't grounded.
    min_confidence: float = 0.5
