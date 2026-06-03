"""LLM client configuration.

Kept tiny and provider-neutral. `model` uses an OpenAI name today; when we route
through LiteLLM it becomes a LiteLLM model string (e.g. "openai/gpt-4o-mini")
with no other change to callers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-4o-mini"  # cheap, supports structured outputs
    temperature: float = 0.0  # deterministic — we want faithful extraction, not creativity
    max_tokens: int = 800  # cap output to keep cost/latency down
    api_key: str | None = None  # falls back to OPENAI_API_KEY env when None

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(model=os.getenv("LLM_MODEL", cls.model))
