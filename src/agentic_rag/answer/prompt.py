"""Prompt construction for the single-shot baseline.

Sources are numbered [S1]..[Sn]; the model is told to cite those exact labels.
That numbering is the join key the validator uses to ground every citation back
to a real retrieved chunk (see validate.py).
"""

from __future__ import annotations

from agentic_rag.retrieve.models import RetrievedChunk

SYSTEM_PROMPT = """You are a precise research assistant answering questions about machine-learning papers.

Answer using ONLY the numbered SOURCES given in the user message. Rules:
- Use only information found in the SOURCES. Never use outside or prior knowledge.
- After every factual claim, cite the supporting source(s) inline using their bracket
  label, e.g. "The Transformer dispenses with recurrence [S1]." Combine when needed: [S2][S3].
- In `citations`, list every source you used: its source_id exactly as shown (e.g. "S1"),
  its arxiv_id, section, and page (copied from the source header).
- If the SOURCES do not contain enough information to answer, set `insufficient_context`
  to true, set `answer` to one sentence saying you don't have enough information in the
  provided sources, and leave `citations` empty. Do NOT guess or fill gaps from memory.
- Be concise and factual."""


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as labeled sources [S1]..[Sn]."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        pages = f"p.{c.page}" if c.page == c.page_end else f"p.{c.page}-{c.page_end}"
        header = f'[S{i}] "{c.title}" (arXiv:{c.arxiv_id}), §{c.section}, {pages}'
        body = " ".join(c.text.split())  # collapse whitespace from PDF extraction
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"SOURCES:\n{format_sources(chunks)}\n\n"
        "Answer the question using only these sources, with an inline [S#] citation "
        "after each claim."
    )
