"""The tracing facade the rest of the app talks to.

Two implementations behind one tiny interface:

- ``NoOpTracer`` — does nothing (the default; tracing disabled). No Langfuse import,
  no network, no overhead.
- ``LangfuseTracer`` — sends a nested trace (one *trace* per agent run, one *span*
  per graph node, one *generation* per LLM call) to a self-hosted Langfuse.

Why a process-global ``get_tracer()`` (set once from env) rather than threading a
tracer through every call: the node spans, the LLM generations, and the root trace
are created in three different places but must share one parent stack to nest
correctly. A single shared instance is the simplest way to get that — and it's how
observability SDKs are normally wired. Tests inject a fake via ``configure_tracer``.

**Tracing must never break the app.** Every Langfuse call is wrapped so a failure
(bad keys, server down, SDK change) degrades to no-tracing instead of raising.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any

from .config import TracingConfig


def _warn(msg: str) -> None:
    print(f"[tracing] {msg}", file=sys.stderr)


class _Handle:
    """Yielded by ``trace``/``span``; collects output+metadata applied on exit."""

    def __init__(self) -> None:
        self.output: Any = None
        self.metadata: dict | None = None

    def update(self, *, output: Any = None, metadata: dict | None = None) -> None:
        if output is not None:
            self.output = output
        if metadata is not None:
            self.metadata = metadata


# --- no-op (tracing disabled) ------------------------------------------------


class NoOpTracer:
    enabled = False
    last_trace_url: str | None = None

    @contextmanager
    def trace(self, name: str, *, input: Any = None, metadata: dict | None = None):
        yield _Handle()

    @contextmanager
    def span(self, name: str, *, input: Any = None):
        yield _Handle()

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        output: Any = None,
        usage: dict | None = None,
        start_time: Any = None,
        end_time: Any = None,
    ) -> None:
        pass

    def flush(self) -> None:
        pass


# --- real Langfuse tracer ----------------------------------------------------


class LangfuseTracer:
    """Manual instrumentation over the Langfuse (v2) SDK.

    Maintains an explicit parent stack: ``trace``/``span`` push their observation
    on enter and pop on exit; ``generation`` attaches to whatever is on top. Because
    the agent graph runs nodes sequentially in one thread, this stack reproduces the
    correct nesting (run > node > llm-call) without relying on context vars.
    """

    enabled = True

    def __init__(self, client: Any) -> None:
        self._client = client
        self._stack: list[Any] = []
        self.last_trace_url: str | None = None

    def _safe(self, fn, default=None):
        try:
            return fn()
        except Exception as exc:  # observability must never break the app
            _warn(f"dropped a span/generation: {exc}")
            return default

    @contextmanager
    def trace(self, name: str, *, input: Any = None, metadata: dict | None = None):
        obj = self._safe(lambda: self._client.trace(name=name, input=input, metadata=metadata))
        if obj is not None:
            self.last_trace_url = self._safe(obj.get_trace_url)
        self._stack.append(obj)
        handle = _Handle()
        try:
            yield handle
        finally:
            self._stack.pop()
            if obj is not None:
                self._safe(lambda: obj.update(output=handle.output, metadata=handle.metadata))
            self._safe(self._client.flush)

    @contextmanager
    def span(self, name: str, *, input: Any = None):
        parent = self._stack[-1] if self._stack else None
        obj = (
            self._safe(lambda: parent.span(name=name, input=input)) if parent is not None else None
        )
        # Keep the stack non-empty so nested generations still find a parent.
        self._stack.append(obj if obj is not None else parent)
        handle = _Handle()
        try:
            yield handle
        finally:
            self._stack.pop()
            if obj is not None:
                self._safe(lambda: obj.end(output=handle.output, metadata=handle.metadata))

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        output: Any = None,
        usage: dict | None = None,
        start_time: Any = None,
        end_time: Any = None,
    ) -> None:
        parent = self._stack[-1] if self._stack else self._client
        if parent is None:
            return

        def make():
            gen = parent.generation(
                name=name, model=model, input=input, usage=usage, start_time=start_time
            )
            gen.end(output=output, end_time=end_time)

        self._safe(make)

    def flush(self) -> None:
        self._safe(self._client.flush)


# --- construction + process-global -------------------------------------------


def build_tracer(config: TracingConfig | None = None):
    """Return a tracer for the given config. Falls back to NoOp on any problem."""
    cfg = config or TracingConfig.from_env()
    if not cfg.enabled:
        return NoOpTracer()
    if not cfg.has_keys:
        _warn("LANGFUSE_TRACING is on but LANGFUSE_PUBLIC_KEY/SECRET_KEY are missing — disabled.")
        return NoOpTracer()
    try:
        from langfuse import Langfuse
    except ImportError:
        _warn("langfuse is not installed (`uv sync`) — tracing disabled.")
        return NoOpTracer()
    try:
        client = Langfuse(public_key=cfg.public_key, secret_key=cfg.secret_key, host=cfg.host)
    except Exception as exc:
        _warn(f"could not initialize Langfuse ({exc}) — tracing disabled.")
        return NoOpTracer()
    return LangfuseTracer(client)


_TRACER: Any = None


def get_tracer():
    """The process-wide tracer, built lazily from the environment on first use."""
    global _TRACER
    if _TRACER is None:
        _TRACER = build_tracer()
    return _TRACER


def configure_tracer(tracer) -> None:
    """Override the global tracer (tests / CLI). Pass None to rebuild from env."""
    global _TRACER
    _TRACER = tracer
