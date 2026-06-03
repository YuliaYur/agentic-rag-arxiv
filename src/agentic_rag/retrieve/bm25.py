"""In-memory BM25 keyword search over the chunk texts.

BM25 is the classic lexical ranking function: it rewards chunks that contain the
query's *exact* terms, weighted by term rarity and dampened by length. This is
the half of hybrid retrieval that dense embeddings are bad at — exact tokens
like model names ("RoBERTa"), datasets ("SQuAD", "GLUE"), metrics, symbols.

We run it in-process (the corpus is ~1,150 chunks; this is instant) rather than
storing sparse vectors in Qdrant — see DECISIONS.md for that tradeoff.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

# Lowercase alphanumeric tokens. Keeps "bm25", "roberta", "squad" intact; splits
# "O(n^2)" into ["o", "n", "2"] which is fine for keyword matching.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """A BM25 index over (id, text) pairs."""

    def __init__(self, ids: list[str], texts: list[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must be the same length")
        self.ids = ids
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return up to `limit` (id, score) pairs, best first, score > 0 only."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda kv: kv[1], reverse=True)
        return [(doc_id, float(s)) for doc_id, s in ranked[:limit] if s > 0.0]
