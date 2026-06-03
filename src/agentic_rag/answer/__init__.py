"""Single-shot RAG baseline: retrieve -> stuff context -> generate cited answer.

Kept intentionally simple and *intact* as the agent layer is added later, so the
two can be compared on the same eval set.
"""

from .baseline import SingleShotRAG, build_baseline
from .schemas import Citation, CitedAnswer
from .validate import SourceRef, ValidatedAnswer, validate_cited_answer

__all__ = [
    "SingleShotRAG",
    "build_baseline",
    "Citation",
    "CitedAnswer",
    "SourceRef",
    "ValidatedAnswer",
    "validate_cited_answer",
]
