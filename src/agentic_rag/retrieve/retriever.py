"""Hybrid retriever: dense + BM25 -> RRF fusion -> cross-encoder rerank.

The orchestration here is pure Python over injected components (a dense searcher,
a BM25 index, an optional reranker), so it can be unit-tested with fakes without
touching Qdrant or downloading any model. ``build_retriever`` wires the real
components for production use.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from itertools import zip_longest

from .bm25 import BM25Index, tokenize
from .config import RetrieveConfig
from .fusion import reciprocal_rank_fusion
from .models import RetrievedChunk

# Generic words that shouldn't drive title matching or pad an anchored query.
# (Distinctive matching also relies on document-frequency, below; this just trims
# obvious noise like articles and ubiquitous paper-title filler.)
_TITLE_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "for",
        "and",
        "or",
        "to",
        "in",
        "on",
        "as",
        "is",
        "are",
        "be",
        "with",
        "by",
        "this",
        "that",
        "its",
        "it",
        "you",
        "all",
        "we",
        "using",
        "via",
        "from",
        "at",
        "into",
        "our",
        "approach",
        "towards",
        "toward",
    }
)


class HybridRetriever:
    def __init__(
        self,
        chunks,
        dense_searcher,
        bm25_index,
        reranker=None,
        config: RetrieveConfig | None = None,
    ) -> None:
        self._by_id = {c.id: c for c in chunks}
        # arxiv_id -> title, for title-anchoring decomposed sub-queries (ADR-0014).
        self._titles = {c.arxiv_id: c.title for c in chunks}
        self._dense = dense_searcher
        self._bm25 = bm25_index
        self._reranker = reranker
        self._cfg = config or RetrieveConfig()

    @property
    def titles(self) -> dict[str, str]:
        """arxiv_id -> paper title for the indexed corpus (read-only view)."""
        return self._titles

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query, with source metadata + scores."""
        cfg = self._cfg
        k = k or cfg.final_k

        dense = self._dense.search(query, cfg.dense_candidates)  # [(id, cos)]
        bm25 = self._bm25.search(query, cfg.bm25_candidates)  # [(id, bm25)]
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
            candidates.append(
                dataclasses.replace(
                    base,
                    score=rrf_score,
                    dense_rank=dense_rank.get(doc_id),
                    bm25_rank=bm25_rank.get(doc_id),
                )
            )

        # Rerank the top fused candidates with the cross-encoder (the costly step),
        # then BLEND the rerank order with the fusion order via RRF. Pure rerank-sort
        # discards the fusion signal and lets the cross-encoder bury a fusion-strong
        # result on ambiguous queries; blending keeps a candidate that's strong in
        # *both* on top while still letting the reranker promote within the head.
        if self._reranker is not None and cfg.use_reranker and candidates:
            head = candidates[: cfg.rerank_candidates]
            tail = candidates[cfg.rerank_candidates :]
            scores = self._reranker.score(query, [c.text for c in head])
            for c, s in zip(head, scores, strict=True):
                c.rerank_score = s
            fusion_order = [c.id for c in head]  # head is already in fusion order
            rerank_order = [c.id for c in sorted(head, key=lambda c: c.rerank_score, reverse=True)]
            blended = reciprocal_rank_fusion([fusion_order, rerank_order], k=cfg.rerank_rrf_k)
            position = {doc_id: i for i, (doc_id, _score) in enumerate(blended)}
            head.sort(key=lambda c: position[c.id])
            candidates = head + tail

        return _select_top_k(candidates, k, cfg.max_per_paper)


def _select_top_k(candidates, k, max_per_paper):
    """Take the top-k candidates, capping how many may come from one paper.

    Greedy over the ranked candidates: admit a chunk unless its paper already
    holds ``max_per_paper`` slots. If the cap leaves us short of k (fewer papers
    than k/cap demands), backfill from the skipped chunks in their original rank
    order — so the cap never *shrinks* the result, it only diversifies it.
    """
    if not max_per_paper or max_per_paper <= 0:
        return candidates[:k]

    selected: list[RetrievedChunk] = []
    overflow: list[RetrievedChunk] = []
    per_paper: dict[str, int] = {}
    for c in candidates:
        if len(selected) >= k:
            break
        if per_paper.get(c.arxiv_id, 0) < max_per_paper:
            selected.append(c)
            per_paper[c.arxiv_id] = per_paper.get(c.arxiv_id, 0) + 1
        else:
            overflow.append(c)
    if len(selected) < k:
        selected.extend(overflow[: k - len(selected)])
    return selected[:k]


def round_robin_merge(result_lists, k):
    """Interleave several ranked chunk lists, deduping by id, to k chunks.

    Used to merge per-sub-question retrievals for a decomposed comparison query:
    taking every list's rank-1 before any list's rank-2 GUARANTEES each side of
    the comparison is represented, instead of letting the dominant side's higher
    scores fill every slot. Each input list keeps its own order.
    """
    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for tier in zip_longest(*result_lists):
        for chunk in tier:
            if chunk is None or chunk.id in seen:
                continue
            seen.add(chunk.id)
            merged.append(chunk)
            if len(merged) >= k:
                return merged
    return merged


def anchor_query_to_title(query, titles, aliases=None, max_df=2):
    """Prepend a paper's distinctive title terms to a sub-query that names it.

    A foundational paper (BERT, the original Transformer) is cited by many others,
    so a *topical* sub-query like "BERT masked language modeling" retrieves the
    papers that DESCRIBE it ahead of the paper itself. If the sub-query clearly
    points at one corpus paper, we prepend that paper's distinctive title words so
    retrieval locks onto the source. Two ways a query "points at" a paper:

      1. an alias phrase (all its tokens present in the query) — for papers whose
         common name isn't their title (e.g. "original transformer" ->
         "Attention Is All You Need"); aliases maps {phrase: arxiv_id}.
      2. otherwise, the paper whose title shares the most DISTINCTIVE tokens with
         the query — distinctive = appearing in at most ``max_df`` titles, so a
         model name ("electra", "roberta") counts but generic words
         ("transformer", "language", "models") do not.

    Pure and corpus-data-driven; a no-op (returns the query unchanged) when nothing
    matches strongly, so it can never *hurt* a query it doesn't understand.
    """
    if not titles:
        return query
    q_tokens = {t for t in tokenize(query) if t not in _TITLE_STOP}
    if not q_tokens:
        return query

    # 1. alias phrases (curated name != title cases)
    for phrase, arxiv_id in (aliases or {}).items():
        ptoks = [t for t in tokenize(phrase) if t not in _TITLE_STOP]
        if ptoks and all(t in q_tokens for t in ptoks) and arxiv_id in titles:
            return _prepend_title(titles[arxiv_id], query, q_tokens)

    # 2. distinctive title-token overlap
    title_tokens = {
        aid: {t for t in tokenize(title) if t not in _TITLE_STOP} for aid, title in titles.items()
    }
    df = Counter(t for toks in title_tokens.values() for t in toks)
    best_key, best_aid = (0, 0), None
    for aid, toks in title_tokens.items():
        shared = q_tokens & toks
        distinctive = sum(1 for t in shared if df[t] <= max_df)
        key = (distinctive, len(shared))
        if distinctive >= 1 and key > best_key:
            best_key, best_aid = key, aid
    if best_aid is not None:
        return _prepend_title(titles[best_aid], query, q_tokens)
    return query


def _prepend_title(title, query, q_tokens):
    """Prefix the title's distinctive words (those not already in the query)."""
    extra = [t for t in tokenize(title) if t not in _TITLE_STOP and t not in q_tokens]
    return f"{' '.join(extra)} {query}".strip() if extra else query


def title_anchored_query(title, query):
    """Public: ``query`` prefixed with ``title``'s distinctive words. Used to build
    a decomposed sub-query that is BOTH topical (keeps the question) and anchored to
    one paper (its title terms), so retrieval gets that paper's *relevant* passage,
    not just its abstract."""
    q_tokens = {t for t in tokenize(query) if t not in _TITLE_STOP}
    return _prepend_title(title, query, q_tokens)


def detect_named_papers(question, names):
    """arxiv_ids of corpus papers the QUESTION explicitly names, via the corpus
    name registry ``names`` (name-phrase -> arxiv_id).

    Deterministic counterpart to the LLM grader for one high-value judgment: did
    the question name a paper that retrieval failed to surface? A name matches when
    every (non-stopword) token of the name phrase is present in the question — so
    "bert" flags BERT but not RoBERTa, and "masked autoencoder" (not the lone word
    "masked") is what flags MAE. Registry-curated to keep this precise.
    """
    q_tokens = {t for t in tokenize(question) if t not in _TITLE_STOP}
    named: set[str] = set()
    for phrase, arxiv_id in dict(names).items():
        ptoks = [t for t in tokenize(phrase) if t not in _TITLE_STOP]
        if ptoks and all(t in q_tokens for t in ptoks):
            named.add(arxiv_id)
    return named


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
