"""Prompts for the grader and the citation critic.

Both reuse the baseline's source formatting (numbered [S#] blocks) so the same
grounding/citation convention runs through the whole graph.
"""

from __future__ import annotations

from agentic_rag.answer.prompt import format_sources
from agentic_rag.answer.validate import ValidatedAnswer
from agentic_rag.retrieve.models import RetrievedChunk

GRADE_SYSTEM = """You judge whether the retrieved SOURCES are sufficient and relevant to answer the QUESTION.

- Set sufficient=true ONLY if the sources together contain enough to answer the question well.
- For multi-hop / comparative questions (needing facts from several papers), be strict:
  if a needed piece is missing, it is NOT sufficient.
- When the question compares NAMED subjects (e.g. "A vs B"), each subject must be grounded in a
  source that is ABOUT that subject — check the source TITLES (arXiv:...). A source counts for a
  subject ONLY if its title is that subject's own paper. One paper merely *describing* another
  side does NOT count: ELECTRA's or RoBERTa's text mentioning BERT does NOT make BERT covered, and
  RoBERTa's title containing the word "BERT" does NOT make it the BERT paper.
- PROCEDURE for a comparison: in `reasoning`, list each named subject and the exact source title
  that IS its paper, e.g. "ELECTRA -> [S1] ELECTRA...; BERT -> NONE". If ANY subject maps to NONE,
  you MUST set sufficient=false and emit one entity-anchored sub_query for each missing subject.
- If not sufficient, set refined_query to a better search query — add specific terms, entities,
  method names, or the missing sub-question — so the next retrieval finds what's missing.
- COMPARATIVE / MULTI-HOP questions: if the question compares or combines several papers/methods
  (e.g. "how does A differ from B?", "X versus Y") and the SOURCES cover only some of them, set
  sub_queries to ONE focused query per distinct subject/side so each is retrieved on its own.
  ANCHOR each sub-query to its subject's OWN identity — name the model/method plus its distinctive
  terms (architecture, authors, the specific mechanism), NOT just the shared topic. A foundational
  paper (e.g. BERT, the original Transformer) is cited by many others, so a generic topical query
  ("masked language modeling objective") retrieves the papers that *cite* it instead of the paper
  itself; the distinctive name/terms are what surface the source. E.g. for "how does ELECTRA differ
  from BERT?": ["ELECTRA replaced-token-detection generator discriminator pre-training",
  "BERT deep bidirectional transformers Devlin masked language modeling"]. Leave sub_queries empty
  for single-subject questions.
- If sufficient, set refined_query to the original question and leave sub_queries empty."""

CRITIC_SYSTEM = """You are a citation auditor. Given the QUESTION, a proposed ANSWER (with inline [S#]
markers), and the numbered SOURCES, check whether the factual claims in the answer are supported by
the cited sources.

- A claim counts as SUPPORTED if it is stated in, or is a reasonable paraphrase or direct
  inference from, a cited source. Do NOT require verbatim wording, and do not penalize correct
  background phrasing that a cited source clearly implies.
- Only flag a claim as unsupported if it is clearly absent from, or contradicted by, the cited
  sources (a genuine hallucination or a miscitation) — not merely reworded.
- supported=true if every factual claim meets the bar above.
- score = fraction of claims that are supported (0.0-1.0).
- List only genuinely unsupported/miscited claims in unsupported_claims.
- feedback: concrete instructions to fix those specific claims. Empty string if fully supported."""


def build_grade_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"QUESTION: {question}\n\nSOURCES:\n{format_sources(chunks)}"


def build_critic_prompt(
    question: str, answer: ValidatedAnswer, chunks: list[RetrievedChunk]
) -> str:
    return f"QUESTION: {question}\n\nANSWER:\n{answer.answer}\n\nSOURCES:\n{format_sources(chunks)}"


def revision_note(critic: dict) -> str:
    """A revision instruction appended to the generate prompt on a re-try."""
    claims = "; ".join(critic.get("unsupported_claims", [])) or "(see feedback)"
    return (
        "\n\nREVISION REQUIRED — a citation audit flagged these unsupported claims: "
        f"{claims}. Feedback: {critic.get('feedback', '')}. "
        "Make the MINIMAL change: only fix the flagged claims — remove them, soften them, or add "
        "the correct [S#] citation. Do NOT introduce any new claims, and do NOT reword sentences "
        "that were already supported; keep the rest of the answer identical. "
        "If the sources truly don't support an answer, set insufficient_context=true."
    )
