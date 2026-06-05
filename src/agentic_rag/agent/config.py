"""Agent graph configuration (loop caps + retrieval breadth)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import CORPUS_PAPER_NAMES


@dataclass(frozen=True)
class AgentConfig:
    k: int = 5  # chunks retrieved per round
    # On a re-retrieve round the grader may decompose a comparison into per-side
    # sub-queries; cap how many we honor (bounds retrieval fan-out / latency).
    max_sub_queries: int = 3
    # Title-anchor each sub-query (prepend the named paper's distinctive title
    # words) so a foundational paper isn't buried under papers that cite it — see
    # anchor_query_to_title / ADR-0014.
    anchor_sub_queries: bool = True
    # Deterministically force a decomposed re-retrieval when the question NAMES a
    # paper that round-1 retrieval missed — a safety net over the LLM grader,
    # which (at temperature 0, gpt-4o-mini) judges these borderline comparatives
    # "sufficient" inconsistently and so under-triggers the loop. See ADR-0014.
    enforce_named_paper_coverage: bool = True
    # Corpus name registry (name phrase -> arxiv_id) used for BOTH naming-detection
    # and sub-query anchoring. See agent/corpus.py / ADR-0014.
    paper_names: tuple[tuple[str, str], ...] = field(default=CORPUS_PAPER_NAMES)
    max_retrieval_rounds: int = 3  # cap on the grade -> re-retrieve loop
    max_revision_rounds: int = 2  # cap on the cite_critic -> revise loop
    # Stop revising once the critic's supported-claim fraction clears this, even if
    # not every claim passed. Stops the loop churning a "good enough" answer (and
    # combined with keep-best, revisions can only ever improve the final answer).
    accept_score: float = 0.8
