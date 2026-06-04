"""Thin wrapper around the LLM provider.

The rest of the app depends only on ``LLMClient.structured(...)``, never on the
OpenAI SDK directly. That isolation is deliberate: swapping to **LiteLLM** later
(for routing/caching/fallbacks across providers) means changing only this file —
LiteLLM mirrors the OpenAI chat-completions API, including `response_format` for
structured outputs.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel

from .config import LLMConfig

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """LLM call failed or returned nothing usable."""


class LLMRefusal(LLMError):
    """The model refused to answer (OpenAI structured-output refusal)."""


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        # Imported lazily so importing the package doesn't require the SDK or a key.
        from openai import OpenAI

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set (put it in .env or the environment).")
        self._client = OpenAI(api_key=api_key)

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        """Return a validated instance of `schema` (OpenAI Structured Outputs).

        Using the schema as `response_format` makes the provider return JSON that
        conforms to the Pydantic model — the parsing/validation happens at the API
        layer, so we get a typed object back, not a string to hand-parse.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # `chat.completions.parse` is the structured-output helper (older SDKs
        # expose it under `.beta`); support both.
        parse = getattr(self._client.chat.completions, "parse", None)
        if parse is None:
            parse = self._client.beta.chat.completions.parse

        # Bracket the call so the traced generation gets the real latency (we record
        # it after the call returns, so without these timestamps it would read ~0ms).
        start = datetime.now(UTC)
        completion = parse(
            model=self.config.model,
            messages=messages,
            response_format=schema,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        end = datetime.now(UTC)
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMRefusal(message.refusal)
        if message.parsed is None:
            raise LLMError("model returned no parsed structured output")

        # Record the call for tracing: token usage lets Langfuse compute cost from
        # its model price list. No-op (and no import cost) when tracing is disabled.
        self._trace_generation(schema.__name__, messages, message.parsed, completion, start, end)
        return message.parsed

    def _trace_generation(
        self, name: str, messages: list, parsed: T, completion, start, end
    ) -> None:
        from agentic_rag.observability import get_tracer

        usage = getattr(completion, "usage", None)
        usage_details = (
            {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
                "unit": "TOKENS",
            }
            if usage is not None
            else None
        )
        get_tracer().generation(
            name=name,
            model=self.config.model,
            input=messages,
            output=_repair_mojibake(parsed.model_dump()),
            usage=usage_details,
            start_time=start,
            end_time=end,
        )


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
