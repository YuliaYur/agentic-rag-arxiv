"""Hybrid retriever: dense + BM25 -> RRF fusion -> cross-encoder rerank.

The orchestration here is pure Python over injected components (a dense searcher,
a BM25 index, an optional reranker), so it can be unit-tested with fakes without
touching Qdrant or downloading any model. ``build_retriever`` wires the real
components for production use.
"""

from __future__ import annotations

import dataclasses

from .bm25 import BM25Index
from .config import RetrieveConfig
from .fusion import reciprocal_rank_fusion
from .models import RetrievedChunk


class HybridRetriever:
    def __init__(self, chunks, dense_searcher, bm25_index, reranker=None,
                 config: RetrieveConfig | None = None) -> None:
        self._by_id = {c.id: c for c in chunks}
        self._dense = dense_searcher
        self._bm25 = bm25_index
        self._reranker = reranker
        self._cfg = config or RetrieveConfig()

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query, with source metadata + scores."""
        cfg = self._cfg
        k = k or cfg.final_k

        dense = self._dense.search(query, cfg.dense_candidates)   # [(id, cos)]
        bm25 = self._bm25.search(query, cfg.bm25_candidates)      # [(id, bm25)]
        dense_ids = [doc_id for doc_id, _ in dense]
        bm25_ids = [doc_id for doc_id, _ in bm25]

        fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=cfg.rrf_k)  # [(id, rrf)]

        dense_rank = {doc_id: r for r, doc_id in enumerate(dense_ids, start=1)}
        bm25_rank = {doc_id: r for r, doc_id in enumerate(bm25_ids, start=1)}

        # Resolve fused ids -> fresh RetrievedChunk copies carrying their scores.
        candidates: list[RetrievedChunk] = []
        for doc_id, rrf_score in fused:
            base = self._by_id.get(doc_id)
            if base is None:
                continue
            candidates.append(dataclasses.replace(
                base,
                score=rrf_score,
                dense_rank=dense_rank.get(doc_id),
                bm25_rank=bm25_rank.get(doc_id),
            ))

        # Rerank the top fused candidates with the cross-encoder (the costly step).
        if self._reranker is not None and cfg.use_reranker and candidates:
            head = candidates[: cfg.rerank_candidates]
            tail = candidates[cfg.rerank_candidates :]
            scores = self._reranker.score(query, [c.text for c in head])
            for c, s in zip(head, scores):
                c.rerank_score = s
            head.sort(key=lambda c: c.rerank_score, reverse=True)
            candidates = head + tail

        return candidates[:k]


def build_retriever(qdrant_config=None, retrieve_config=None, embed_config=None) -> HybridRetriever:
    """Wire the real retriever: Qdrant dense search + in-memory BM25 + reranker.

    Heavy (loads the embedding model, the reranker, and scrolls the index), so
    build it once and reuse across queries.
    """
    from qdrant_client import QdrantClient

    from ..ingest.config import EmbedConfig, QdrantConfig
    from ..ingest.embed import Embedder
    from .dense import QdrantDenseSearcher, load_chunks
    from .rerank import CrossEncoderReranker

    qcfg = qdrant_config or QdrantConfig()
    rcfg = retrieve_config or RetrieveConfig()
    ecfg = embed_config or EmbedConfig()

    client = QdrantClient(host=qcfg.host, port=qcfg.port, check_compatibility=False)
    embedder = Embedder(ecfg)

    chunks = load_chunks(client, qcfg.collection)
    bm25 = BM25Index([c.id for c in chunks], [c.text for c in chunks])
    dense = QdrantDenseSearcher(embedder, client, qcfg.collection, rcfg)
    reranker = CrossEncoderReranker(rcfg) if rcfg.use_reranker else None

    return HybridRetriever(chunks, dense, bm25, reranker, rcfg)
