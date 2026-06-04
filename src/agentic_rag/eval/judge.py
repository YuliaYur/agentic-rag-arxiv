"""LLM-as-judge: a single rubric-scored verdict on overall answer quality.

Where the RAGAS-style metrics each probe one axis, the judge gives a holistic
1-5 score against an explicit rubric, comparing the system answer to the
reference. The rubric is spelled out in the prompt so scores are reproducible and
auditable — not a vibe. Returns the raw 1-5 plus a normalized [0,1] for tables.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

RUBRIC = """Score the ANSWER against the REFERENCE for the QUESTION on a 1-5 scale:

5 - Excellent: fully correct and complete; matches the reference's key facts; well
    grounded in sources; directly answers the question; no errors.
4 - Good: correct and addresses the question; minor omissions or slight imprecision.
3 - Adequate: partially correct but missing an important point, or one notable
    inaccuracy, or only partly answers a multi-part question.
2 - Poor: largely incorrect or mostly off-topic, though touching the subject.
1 - Unacceptable: wrong, irrelevant, or fabricated; or refuses when the reference
    shows the answer was available.

Judge correctness against the REFERENCE. A faithful "insufficient context" refusal
is NOT automatically a 1 — but if the reference shows the answer was attainable,
an unjustified refusal should score low."""


class JudgeVerdict(BaseModel):
    """Rubric-scored judgment of one answer."""

    correctness: int = Field(description="Factual agreement with the reference, 1-5.")
    completeness: int = Field(description="Coverage of the reference's key points, 1-5.")
    relevance: int = Field(description="How directly it answers the question, 1-5.")
    overall: int = Field(description="Holistic overall quality, 1-5 (per the rubric).")
    rationale: str = Field(description="One or two sentences justifying the overall score.")


def judge_answer(llm, question: str, answer: str, reference_answer: str) -> JudgeVerdict:
    system = "You are a meticulous grader of question-answering systems.\n\n" + RUBRIC
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE (ground truth):\n{reference_answer}\n\n"
        f"ANSWER (to grade):\n{answer}"
    )
    return llm.structured(system, user, JudgeVerdict)


def normalized_overall(verdict: JudgeVerdict) -> float:
    """Map the 1-5 overall onto [0,1] so it sits alongside the RAGAS-style scores."""
    return (verdict.overall - 1) / 4.0
