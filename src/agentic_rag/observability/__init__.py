"""Observability: optional Langfuse tracing of the agent graph.

One *trace* per agent run, one *span* per graph node (carrying that node's
structured metadata), one *generation* per LLM call (with token usage so Langfuse
computes cost). Disabled by default; toggle with ``LANGFUSE_TRACING`` + keys.
"""

from .config import TracingConfig
from .tracer import (
    LangfuseTracer,
    NoOpTracer,
    build_tracer,
    configure_tracer,
    get_tracer,
)

__all__ = [
    "TracingConfig",
    "NoOpTracer",
    "LangfuseTracer",
    "build_tracer",
    "get_tracer",
    "configure_tracer",
]
