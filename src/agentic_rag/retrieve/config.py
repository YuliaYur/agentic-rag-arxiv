"""Configuration for the hybrid retrieval layer.

The retrieve-then-rerank funnel:
    dense top-N  ┐
                 ├─ RRF fuse ─→ top-M candidates ─→ cross-encoder rerank ─→ top-k
    bm25  top-N  ┘
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrieveConfig:
    # bge-v1.5 expects this instruction on the QUERY side only (passages were
    # embedded plain at ingestion time). Omitting it measurably hurts recall.
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # How many candidates each retriever contributes before fusion.
    dense_candidates: int = 50
    bm25_candidates: int = 50

    # RRF constant. 60 is the value from the original RRF paper; it damps the
    # influence of very-high ranks so a single #1 hit can't dominate the fusion.
    rrf_k: int = 60

    # Rerank only this many fused candidates (the expensive step). 30 keeps
    # latency low while comfortably covering the useful tail of the fusion.
    rerank_candidates: int = 30
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # The reranker is BLENDED with fusion (RRF over both rankings) rather than
    # replacing it: a general-purpose cross-encoder can confidently misrank an
    # ambiguous query (e.g. promoting any paper's "training details" section for
    # "the original Transformer's optimizer"), so we keep the fusion signal as a
    # safety net. This constant damps the blend the same way rrf_k damps fusion.
    rerank_rrf_k: int = 60

    # Cap how many chunks from a SINGLE paper may occupy the final top-k. A
    # comparison query ("how does A differ from B?") often lets the dominant
    # paper monopolize every slot, burying the second paper the question needs
    # (see q-0005: 5 Chinchilla chunks crowd out Kaplan's scaling-laws paper).
    # The cap forces room for other papers. Recall@k is unaffected for genuine
    # single-paper questions (one chunk still covers that paper); it trades a
    # little context precision there for multi-hop coverage. None disables it.
    max_per_paper: int | None = 3

    final_k: int = 5  # default number of chunks returned to the caller
