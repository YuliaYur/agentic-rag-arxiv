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

    final_k: int = 5  # default number of chunks returned to the caller
