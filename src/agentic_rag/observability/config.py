"""Tracing configuration — read from the environment, toggleable.

Tracing is **off unless explicitly enabled** (``LANGFUSE_TRACING`` truthy) *and*
both Langfuse keys are present. That keeps the default path (tests, offline runs)
free of any Langfuse dependency or network call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TracingConfig:
    enabled: bool = False  # the LANGFUSE_TRACING toggle
    host: str = "http://localhost:3000"
    public_key: str | None = None
    secret_key: str | None = None

    @classmethod
    def from_env(cls) -> TracingConfig:
        return cls(
            enabled=os.getenv("LANGFUSE_TRACING", "").strip().lower() in _TRUTHY,
            host=os.getenv("LANGFUSE_HOST") or cls.host,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY") or None,
            secret_key=os.getenv("LANGFUSE_SECRET_KEY") or None,
        )

    @property
    def has_keys(self) -> bool:
        return bool(self.public_key and self.secret_key)
