"""LLM client configuration: per-role model routing.

All calls go through LiteLLM, so ``model`` strings are LiteLLM-style
("gpt-4o-mini", "gpt-4o", "anthropic/claude-sonnet-4-6", ...). ``model`` is the
default for any role; ``role_models`` overrides specific roles so the agent can run
a cheap/fast model for grading + the citation critic and a stronger model for the
final synthesis — concentrating spend on the one user-facing artifact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-4o-mini"  # default for any role without an override
    temperature: float = 0.0  # deterministic — we want faithful extraction, not creativity
    max_tokens: int = 800  # cap output to keep cost/latency down
    api_key: str | None = None  # provider key; None -> LiteLLM resolves from env
    # Per-role model overrides as (role, model) pairs (tuple so the config stays
    # frozen/hashable). Roles used by the agent: "grade", "critic", "synthesis".
    role_models: tuple[tuple[str, str], ...] = ()

    def resolve(self, role: str | None) -> str:
        """Model for a role, falling back to the default ``model``."""
        if role:
            for r, m in self.role_models:
                if r == role:
                    return m
        return self.model

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Build from env: LLM_MODEL (default) + LLM_MODEL_{SYNTHESIS,GRADE,CRITIC}."""
        roles = [
            (role, os.environ[env])
            for role, env in (
                ("synthesis", "LLM_MODEL_SYNTHESIS"),
                ("grade", "LLM_MODEL_GRADE"),
                ("critic", "LLM_MODEL_CRITIC"),
            )
            if os.getenv(env)
        ]
        return cls(model=os.getenv("LLM_MODEL", cls.model), role_models=tuple(roles))

    @classmethod
    def routed(cls, synthesis: str = "gpt-4o", cheap: str = "gpt-4o-mini") -> LLMConfig:
        """Preset: strong model for synthesis, cheap model for grade + critic."""
        return cls(model=cheap, role_models=(("synthesis", synthesis),))

    @classmethod
    def uniform(cls, model: str) -> LLMConfig:
        """Preset: one model for every role (the naive baseline)."""
        return cls(model=model)
