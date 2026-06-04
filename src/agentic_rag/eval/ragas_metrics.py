"""RAGAS-style answer/context metrics, implemented natively over our ``LLMClient``.

We don't depend on the ``ragas`` package: it pins an older langchain stack that
conflicts with this project's langgraph / langchain-core 1.x (see ADR-0011). These
functions implement the *same definitions* RAGAS uses, with our own LLM client —
so they're transparent, offline-testable with a fake LLM, and traced in Langfuse
for free. All return a score in [0.0, 1.0]; higher is better.

The four metrics (plain language):

- **faithfulness** — *of what the answer claims, how much is actually backed by the
  retrieved context?* Decompose the answer into atomic claims, check each against
  the context, score = supported / total. Low = hallucination.
- **answer_relevancy** — *does the answer actually address the question?* Generate
  the questions this answer would answer, measure how close they are to the real
  question. Penalizes evasive/off-topic answers. (LLM-judged variant of RAGAS's
  embedding-based metric, to avoid a second embedding model in the loop.)
- **context_precision** — *are the retrieved chunks relevant, and ranked well?*
  Judge each retrieved chunk's relevance, then take a rank-weighted average
  precision so relevant chunks near the top score higher. Low = noisy retrieval.
- **context_recall** — *did retrieval bring back everything the answer needs?*
  Decompose the reference answer into claims, check how many are attributable to
  the retrieved context. Low = missing evidence (the gap the agent's re-retrieve
  loop targets).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- structured-output schemas for the judging steps ------------------------


class _Statements(BaseModel):
    statements: list[str] = Field(description="Atomic, self-contained factual claims.")


class _Verdict(BaseModel):
    statement: str
    supported: bool = Field(description="True if the CONTEXT supports this statement.")


class _Verdicts(BaseModel):
    verdicts: list[_Verdict]


class _Relevances(BaseModel):
    # Aligned, in order, to the numbered contexts in the prompt.
    relevant: list[bool] = Field(description="One true/false per context, in the given order.")


class _GenQuestions(BaseModel):
    questions: list[str] = Field(description="Questions this answer fully answers.")
    noncommittal: bool = Field(description="True if the answer is evasive/'I don't know'.")


class _Similarities(BaseModel):
    scores: list[float] = Field(description="Similarity 0.0-1.0 of each generated question.")


# --- shared primitives ------------------------------------------------------


# A refusal/abstention makes no factual claim about the subject, so it cannot be
# unfaithful. Without this guard the extractor turns "I don't have enough
# information ..." into a phantom statement that then scores as unsupported,
# unfairly punishing honest abstention (an eval bug, not a model failing).
_REFUSAL_MARKERS = (
    "don't have enough",
    "do not have enough",
    "not enough information",
    "insufficient context",
    "no relevant sources",
    "cannot answer",
    "can't answer",
)


def _is_refusal(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _REFUSAL_MARKERS)


def _extract_statements(llm, text: str) -> list[str]:
    system = (
        "Break the TEXT into a list of atomic, self-contained factual statements about the "
        "subject matter. Each should stand alone (resolve pronouns). Ignore filler, citation "
        "markers like [S1], and meta-statements about the answer itself or its uncertainty "
        "(e.g. 'I don't have enough information')."
    )
    return llm.structured(system, f"TEXT:\n{text}", _Statements).statements


def _verify(llm, statements: list[str], context: str) -> float:
    """Fraction of statements the context supports (used by faithfulness + recall)."""
    if not statements:
        return 1.0  # nothing claimed -> nothing unsupported
    system = (
        "For each STATEMENT, decide whether it can be directly inferred from the CONTEXT. "
        "Judge ONLY against the context, not your own knowledge."
    )
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    user = f"CONTEXT:\n{context}\n\nSTATEMENTS:\n{numbered}"
    verdicts = llm.structured(system, user, _Verdicts).verdicts
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.supported) / len(verdicts)


# --- the four metrics -------------------------------------------------------


def faithfulness(llm, answer: str, contexts: list[str]) -> float | None:
    """Returns None when not applicable (no context, or an honest refusal with no
    claims) so the aggregate skips it rather than scoring it 0."""
    if not contexts or _is_refusal(answer):
        return None
    statements = _extract_statements(llm, answer)
    if not statements:
        return None
    return _verify(llm, statements, "\n\n".join(contexts))


def context_recall(llm, reference_answer: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    statements = _extract_statements(llm, reference_answer)
    return _verify(llm, statements, "\n\n".join(contexts))


def context_precision(llm, question: str, reference_answer: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    system = (
        "Decide, for each numbered CONTEXT in order, whether it is useful for answering the "
        "QUESTION given the REFERENCE answer. Return one true/false per context, same order."
    )
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    user = f"QUESTION: {question}\n\nREFERENCE: {reference_answer}\n\nCONTEXTS:\n{numbered}"
    flags = llm.structured(system, user, _Relevances).relevant
    # Align defensively to the number of contexts (pad missing as not-relevant).
    flags = (flags + [False] * len(contexts))[: len(contexts)]
    if not any(flags):
        return 0.0
    # Rank-weighted average precision: relevant chunks higher up score more.
    hits = 0
    precision_sum = 0.0
    for i, rel in enumerate(flags, start=1):
        if rel:
            hits += 1
            precision_sum += hits / i
    return precision_sum / hits


def answer_relevancy(llm, question: str, answer: str) -> float:
    gen = llm.structured(
        "Given the ANSWER, generate 3 questions that this answer fully and specifically answers. "
        "Set noncommittal=true if the answer is evasive or says it doesn't know.",
        f"ANSWER:\n{answer}",
        _GenQuestions,
    )
    if gen.noncommittal or not gen.questions:
        return 0.0
    sims = llm.structured(
        "Rate how semantically similar each CANDIDATE question is to the TARGET question, "
        "from 0.0 (unrelated) to 1.0 (same question). Return one score per candidate, in order.",
        f"TARGET: {question}\n\nCANDIDATES:\n"
        + "\n".join(f"{i}. {q}" for i, q in enumerate(gen.questions, 1)),
        _Similarities,
    ).scores
    if not sims:
        return 0.0
    clamped = [min(1.0, max(0.0, s)) for s in sims]
    return sum(clamped) / len(clamped)


# Metric registry: name -> (fn, needs_reference). The runner calls each with the
# arguments it declares, so adding a metric here is the only change needed.
def compute_ragas_metrics(
    llm, question: str, answer: str, contexts: list[str], reference_answer: str
) -> dict[str, float]:
    return {
        "faithfulness": faithfulness(llm, answer, contexts),
        "answer_relevancy": answer_relevancy(llm, question, answer),
        "context_precision": context_precision(llm, question, reference_answer, contexts),
        "context_recall": context_recall(llm, reference_answer, contexts),
    }
