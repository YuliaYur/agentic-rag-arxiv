"""Cross-encoder reranking.

The retrieval (dense + BM25) uses *bi-encoders*: query and passage are embedded
separately, so similarity is cheap but coarse. A *cross-encoder* feeds the
(query, passage) pair through the model **together**, so it can judge true
relevance far more accurately — at the cost of one model inference per
candidate (no precomputation possible).

That cost is why reranking is the last, narrow step: we only rerank the top ~30
fused candidates, not the whole corpus. Latency/quality tradeoff is discussed in
DECISIONS.md.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — free, local, ~80MB, fast on CPU,
and the long-standing default for this task.
"""

from __future__ import annotations

from .config import RetrieveConfig


class CrossEncoderReranker:
    def __init__(self, config: RetrieveConfig | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._cfg = config or RetrieveConfig()
        self._model = CrossEncoder(self._cfg.reranker_model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score for each (query, text) pair. Higher = more relevant."""
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self._model.predict(pairs)]
