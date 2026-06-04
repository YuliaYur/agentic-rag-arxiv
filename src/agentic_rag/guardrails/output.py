"""Output guardrail: decide whether a finished answer is safe to surface.

Runs the checks the task asks for, in order, and stops at the first failure:

1. **structure** — there is a validated answer with the expected shape.
2. **refuse-if-insufficient** — if generation declared the context insufficient,
   we *decline* (honour the abstain) instead of dressing a non-answer up as a result.
3. **grounded** — the grounding validator found no violations (nothing cited is
   fabricated; the cite-or-abstain contract held).
4. **confidence threshold** — the citation critic's fraction-of-claims-supported
   score must clear ``min_confidence``; below it we decline.

Confidence is gated to 0 when the answer isn't grounded, so an ungrounded answer
can never pass the threshold on critic score alone.
"""

from __future__ import annotations

from agentic_rag.answer.validate import ValidatedAnswer

from .config import GuardrailsConfig
from .models import CheckResult, GuardrailDecision

_DECLINE_TEMPLATE = (
    "I can't give a confident, grounded answer to this from the available sources "
    "(guardrail: {reason}). Try rephrasing, or this may be outside the indexed papers."
)


def _confidence(validated: ValidatedAnswer, critic: dict | None) -> float:
    """Critic's supported-fraction, gated to 0 when the answer isn't grounded."""
    if not validated.is_grounded:
        return 0.0
    if critic is None:
        # No critic ran; fall back to grounded-or-not as a coarse signal.
        return 1.0
    return float(critic.get("score", 0.0))


def check_output(
    validated: ValidatedAnswer | None,
    critic: dict | None,
    config: GuardrailsConfig | None = None,
) -> GuardrailDecision:
    cfg = config or GuardrailsConfig()
    checks: list[CheckResult] = []

    def decline(reason: str, final: str | None = None) -> GuardrailDecision:
        return GuardrailDecision(
            action="decline",
            reason=reason,
            confidence=conf,
            checks=checks,
            final_answer=final if final is not None else _DECLINE_TEMPLATE.format(reason=reason),
        )

    # 1. structure -----------------------------------------------------------
    structure_ok = isinstance(validated, ValidatedAnswer) and isinstance(validated.answer, str)
    if cfg.enforce_structure:
        checks.append(
            CheckResult(
                name="structure",
                passed=structure_ok,
                detail="validated answer present" if structure_ok else "missing/malformed answer",
            )
        )
    if not structure_ok:
        # Can't compute confidence without an answer object.
        return GuardrailDecision(
            action="decline",
            reason="malformed_output",
            confidence=0.0,
            checks=checks,
            final_answer=_DECLINE_TEMPLATE.format(reason="malformed_output"),
        )

    conf = _confidence(validated, critic)

    # 2. refuse if context insufficient --------------------------------------
    if cfg.refuse_on_insufficient_context:
        sufficient = not validated.insufficient_context
        checks.append(
            CheckResult(
                name="context_sufficient",
                passed=sufficient,
                detail="" if sufficient else "generation flagged insufficient context",
            )
        )
        if not sufficient:
            # Surface the model's own honest "not enough info" sentence.
            return decline("context_insufficient", final=validated.answer)

    # 3. grounded ------------------------------------------------------------
    if cfg.require_grounded:
        grounded = validated.is_grounded
        checks.append(
            CheckResult(
                name="grounded",
                passed=grounded,
                detail="" if grounded else "; ".join(validated.violations),
            )
        )
        if not grounded:
            return decline("ungrounded")

    # 4. confidence threshold ------------------------------------------------
    meets = conf >= cfg.min_confidence
    checks.append(
        CheckResult(
            name="confidence_threshold",
            passed=meets,
            detail=f"confidence {conf:.2f} vs min {cfg.min_confidence:.2f}",
        )
    )
    if not meets:
        return decline("low_confidence")

    return GuardrailDecision(
        action="answer",
        reason="ok",
        confidence=conf,
        checks=checks,
        final_answer=validated.answer,
    )
