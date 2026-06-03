"""Grounding validation — the programmatic enforcement of "cite or abstain".

The prompt and the schema *ask* the model to behave; this is where we *enforce* it.
Three checks, all independent of the model's goodwill:

1. **Citations must be grounded.** Every citation's `source_id` must match a source
   we actually retrieved. We then rebuild the citation's arxiv_id/section/page from
   the real chunk — so the displayed citation is authoritative, not the model's
   (possibly hallucinated) copy.
2. **Inline markers must be grounded.** Every [S#] marker in the answer text must
   refer to a real source; a marker pointing at a non-existent source is a violation.
3. **Cite-or-abstain.** If `insufficient_context` is false, the answer must carry at
   least one grounded citation (no uncited claims). If it's true, there must be no
   citations.

A result with any violation has `is_grounded == False`; callers decide whether to
reject, retry, or surface it. We can't verify *every sentence* has a citation
without claim extraction (that's a job for the upcoming cite-check critic), but we
can guarantee nothing cited is fabricated and that the abstain path is honored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_rag.retrieve.models import RetrievedChunk

from .schemas import CitedAnswer

_MARKER_RE = re.compile(r"\[S(\d+)\]")


@dataclass
class SourceRef:
    """An authoritative, grounded citation (resolved from a retrieved chunk)."""

    source_id: str
    arxiv_id: str
    title: str
    section: str
    page: int
    page_end: int

    def citation(self) -> str:
        pages = f"p.{self.page}" if self.page == self.page_end else f"p.{self.page}-{self.page_end}"
        return f"[{self.source_id}] {self.title} (arXiv:{self.arxiv_id}) §{self.section} {pages}"


@dataclass
class ValidatedAnswer:
    question: str
    answer: str
    insufficient_context: bool
    citations: list[SourceRef] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return not self.violations


def validate_cited_answer(
    question: str, raw: CitedAnswer, chunks: list[RetrievedChunk]
) -> ValidatedAnswer:
    by_sid = {f"S{i}": c for i, c in enumerate(chunks, start=1)}
    violations: list[str] = []
    citations: list[SourceRef] = []
    seen: set[str] = set()

    def add(source_id: str) -> None:
        if source_id in seen:
            return
        chunk = by_sid[source_id]
        seen.add(source_id)
        citations.append(
            SourceRef(
                source_id=source_id,
                arxiv_id=chunk.arxiv_id,
                title=chunk.title,
                section=chunk.section,
                page=chunk.page,
                page_end=chunk.page_end,
            )
        )

    # 1. Ground the citations the model listed.
    for cit in raw.citations:
        if cit.source_id in by_sid:
            add(cit.source_id)
        else:
            violations.append(f"citation references unknown source {cit.source_id!r}")

    # 2. Inline [S#] markers must also be grounded (and get added to the citation list).
    markers = {f"S{n}" for n in _MARKER_RE.findall(raw.answer)}
    for m in sorted(markers):
        if m in by_sid:
            add(m)
        else:
            violations.append(f"answer cites unknown source [{m}]")

    # 3. Cite-or-abstain.
    if raw.insufficient_context:
        if citations:
            violations.append("insufficient_context is true but citations were provided")
    elif not citations:
        violations.append("answer makes claims but cites no grounded sources")

    return ValidatedAnswer(
        question=question,
        answer=raw.answer,
        insufficient_context=raw.insufficient_context,
        citations=citations,
        violations=violations,
    )
