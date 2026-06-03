"""Thin wrapper around the LLM provider.

The rest of the app depends only on ``LLMClient.structured(...)``, never on the
OpenAI SDK directly. That isolation is deliberate: swapping to **LiteLLM** later
(for routing/caching/fallbacks across providers) means changing only this file —
LiteLLM mirrors the OpenAI chat-completions API, including `response_format` for
structured outputs.
"""

from __future__ import annotations

import os
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

        completion = parse(
            model=self.config.model,
            messages=messages,
            response_format=schema,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMRefusal(message.refusal)
        if message.parsed is None:
            raise LLMError("model returned no parsed structured output")
        return message.parsed
