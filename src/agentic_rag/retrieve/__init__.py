"""Hybrid retrieval: dense + BM25, fused with RRF, then cross-encoder reranked."""

from .models import RetrievedChunk
from .retriever import HybridRetriever, build_retriever

__all__ = ["RetrievedChunk", "HybridRetriever", "build_retriever"]
