"""Guardrails layer around the agent graph.

Two layers, defense-in-depth alongside the grounding validator and cite critic:

- **input** — detect/neutralize prompt injection riding in on retrieved document
  text (the RAG-specific data-channel attack); see ``injection.py``.
- **output** — validate structure, honour "refuse if context insufficient", and
  decline below a grounding/confidence threshold; see ``output.py``.

Behaviour is configurable (``GuardrailsConfig``) and every decision is logged into
the graph trace.
"""

from .config import GuardrailsConfig
from .guard import Guardrails
from .injection import neutralize_text, sanitize_chunks, scan_text
from .models import CheckResult, GuardrailDecision, InjectionHit
from .output import check_output

__all__ = [
    "GuardrailsConfig",
    "Guardrails",
    "scan_text",
    "neutralize_text",
    "sanitize_chunks",
    "check_output",
    "CheckResult",
    "GuardrailDecision",
    "InjectionHit",
]
