"""LLM client — all calls route through LiteLLM.

The rest of the app depends only on ``LLMClient.structured(...)``. Routing through
LiteLLM (instead of the OpenAI SDK directly) buys three things with no change to
callers: **per-role model routing** (a cheap model for grade/critic, a stronger one
for synthesis — see ``LLMConfig``), an optional **semantic cache** (``llm/cache.py``),
and **provider-agnostic** calls (swap to Claude etc. by changing a model string).

Each call records cost + latency + cache-hit to the metering scope
(``llm/metering.py``) and emits a traced generation (``observability``), so
cost-per-query and p50/p95 latency are captured wherever the call happens.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel

from .config import LLMConfig

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """LLM call failed or returned nothing usable."""


class LLMRefusal(LLMError):
    """The model refused to answer (structured-output refusal)."""


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def structured(self, system: str, user: str, schema: type[T], role: str | None = None) -> T:
        """Return a validated ``schema`` instance via LiteLLM structured outputs.

        ``role`` ("grade" | "critic" | "synthesis" | None) selects the model through
        ``LLMConfig.resolve`` — that's the routing knob. The provider returns JSON
        conforming to ``schema``; we validate it into a typed object.
        """
        import litellm

        model = self.config.resolve(role)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        # Bracket the call so the traced generation + metering get real latency.
        start = datetime.now(UTC)
        completion = litellm.completion(
            model=model,
            messages=messages,
            response_format=schema,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            caching=litellm.cache is not None,  # honor the global semantic cache if installed
            **kwargs,
        )
        end = datetime.now(UTC)

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMRefusal(message.refusal)
        content = message.content
        if not content:
            raise LLMError("model returned no content")
        try:
            parsed = schema.model_validate_json(content)
        except Exception as exc:  # malformed JSON despite structured outputs
            raise LLMError(f"could not parse structured output: {exc}") from exc

        self._record(schema.__name__, role, model, messages, parsed, completion, start, end)
        return parsed

    def _record(self, name, role, model, messages, parsed, completion, start, end) -> None:
        """Record cost/latency/cache to the metering scope + emit a traced generation."""
        import litellm

        hidden = getattr(completion, "_hidden_params", {}) or {}
        cached = bool(hidden.get("cache_hit"))
        cost = (
            0.0 if cached else float(hidden.get("response_cost") or _safe_cost(litellm, completion))
        )
        latency_ms = (end - start).total_seconds() * 1000.0

        usage = getattr(completion, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)

        from .metering import CallRecord, record_call

        record_call(
            CallRecord(
                role=role,
                model=model,
                cost_usd=cost,
                latency_ms=latency_ms,
                cached=cached,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        )

        # Trace the generation. We pass LiteLLM's *actual* cost as ``totalCost`` so
        # Langfuse shows what we really paid (0 on a cache hit) instead of estimating
        # cost from the token counts — a cached response still carries tokens, so the
        # token-based estimate would wrongly bill a cache hit.
        from agentic_rag.observability import get_tracer

        usage_details = {
            "input": in_tok,
            "output": out_tok,
            "total": in_tok + out_tok,
            "unit": "TOKENS",
            "totalCost": cost,
        }
        get_tracer().generation(
            name=name,
            model=model,
            input=messages,
            output=_repair_mojibake(parsed.model_dump()),
            usage=usage_details,
            metadata={"cost_usd": round(cost, 6), "cached": cached, "role": role},
            start_time=start,
            end_time=end,
        )


def _safe_cost(litellm, completion) -> float:
    try:
        return float(litellm.completion_cost(completion) or 0.0)
    except Exception:  # cost calc is best-effort — never break a call over it
        return 0.0


# Markers of UTF-8 text that was misdecoded as cp1251/latin-1 (e.g. "§" -> "В§").
# gpt-4o-mini occasionally emits this when copying the "§" we put in source headers.
_MOJIBAKE_MARKERS = ("Â", "Ã", "В", "Ð", "â€")


def _repair_mojibake(value):
    """Best-effort repair of cp1251/latin-1 mojibake in traced model output.

    Only touches strings that actually show a mojibake marker (so clean text is
    never altered), and only if the round-trip decodes cleanly. Walks dicts/lists.
    """
    if isinstance(value, str):
        if any(m in value for m in _MOJIBAKE_MARKERS):
            try:
                return value.encode("cp1251").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                return value
        return value
    if isinstance(value, list):
        return [_repair_mojibake(v) for v in value]
    if isinstance(value, dict):
        return {k: _repair_mojibake(v) for k, v in value.items()}
    return value
