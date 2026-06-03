"""Central configuration for the ingestion pipeline.

Every tunable lives here so the CLI, pipeline, and tests share one source of
truth. Chunking parameters are explained in DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo root = three levels up from this file (src/agentic_rag/ingest/config.py).
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ChunkConfig:
    """Structure-aware chunking parameters.

    Sizes are measured in *tokens*, using whatever token counter the pipeline
    injects (the real run uses the embedding model's own tokenizer so we never
    silently exceed its context window). See DECISIONS.md for the rationale
    behind these defaults and how to tune them.
    """

    target_tokens: int = 384      # aim per chunk; comfortably under the 512-token model window
    overlap_tokens: int = 64      # ~17% carry-over so facts split across a boundary stay retrievable
    min_tokens: int = 48          # merge/drop fragments smaller than this
    max_tokens: int = 480         # hard ceiling; oversize units are split to respect the model window
    respect_sections: bool = True  # never let a chunk span two sections
    drop_references: bool = True   # skip the bibliography (low-value, noisy for QA)
    prepend_context: bool = True   # embed "<title> > <section>\n<text>" for better retrieval


@dataclass(frozen=True)
class EmbedConfig:
    """Local sentence-transformers embedding settings."""

    model_name: str = "BAAI/bge-small-en-v1.5"  # 384-dim, 512-token ctx, strong MTEB retrieval, CPU-friendly
    dim: int = 384
    batch_size: int = 64
    normalize: bool = True  # unit vectors -> cosine == dot product


@dataclass(frozen=True)
class QdrantConfig:
    """Connection + collection settings for the local Dockerized Qdrant."""

    host: str = "localhost"
    port: int = 6333
    collection: str = "arxiv_papers"
    upsert_batch: int = 256


@dataclass(frozen=True)
class IngestConfig:
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    manifest_path: Path = REPO_ROOT / "data" / "raw" / "manifest.json"
    sources_md: Path = REPO_ROOT / "SOURCES.md"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
