"""Orchestrate the ingestion: parse -> chunk -> embed -> index.

Idempotent and re-runnable:
* Papers already fully indexed (same chunk count) are skipped unless ``--force``.
* Re-indexed papers upsert by deterministic id and prune any stale tail, so a
  re-run never duplicates and never leaves orphans.

Dry-run mode parses + chunks only (no model, no Qdrant) so chunking can be
inspected offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunk import Chunk, chunk_blocks, count_tokens
from .config import IngestConfig
from .corpus import Paper, load_corpus
from .parse import parse_pdf


@dataclass
class IngestResult:
    papers_processed: int = 0
    papers_skipped: int = 0
    papers_missing: int = 0
    total_chunks: int = 0
    indexed_chunks: int = 0
    examples: list[Chunk] = field(default_factory=list)


def _select(papers: list[Paper], wanted: list[str] | None, limit: int | None) -> list[Paper]:
    if wanted:
        wl = {w.lower() for w in wanted}
        papers = [p for p in papers if p.arxiv_id in wl or p.slug.lower() in wl]
    if limit is not None:
        papers = papers[:limit]
    return papers


def run_ingest(
    config: IngestConfig,
    *,
    papers: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
    recreate: bool = False,
    do_index: bool = True,
    n_examples: int = 3,
    log=print,
) -> IngestResult:
    corpus = _select(load_corpus(config.manifest_path, config.sources_md), papers, limit)
    result = IngestResult()

    # Lazily stand up the heavy bits (model + Qdrant) only when indexing.
    embedder = None
    index = None
    token_counter = count_tokens
    if do_index:
        from .embed import Embedder
        from .index import VectorIndex

        log(f"Loading embedding model: {config.embed.model_name} ...")
        embedder = Embedder(config.embed)
        token_counter = embedder.token_counter()
        index = VectorIndex(config.qdrant)
        index.ensure_collection(dim=embedder.dim, recreate=recreate)
        log(f"Qdrant collection '{config.qdrant.collection}' ready (dim={embedder.dim}, cosine).")
    else:
        log("DRY RUN: parsing + chunking only (no embedding, no Qdrant).")

    try:
        from tqdm import tqdm

        iterator = tqdm(corpus, desc="Ingesting", unit="paper")
    except Exception:  # tqdm optional
        iterator = corpus

    for paper in iterator:
        pdf = paper.pdf_path(config.raw_dir)
        if not pdf.exists():
            log(f"  ! missing PDF, skipping: {pdf.name} (run scripts/fetch_corpus.py)")
            result.papers_missing += 1
            continue

        blocks = parse_pdf(pdf, drop_references=config.chunk.drop_references)
        chunks = chunk_blocks(
            blocks,
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            slug=paper.slug,
            config=config.chunk,
            token_counter=token_counter,
        )
        result.total_chunks += len(chunks)
        # Keep a spread of early examples for the end-of-run display.
        for c in chunks:
            if len(result.examples) < n_examples and c.section not in {"Frontmatter", ""}:
                result.examples.append(c)

        if not do_index:
            log(f"  {paper.arxiv_id}  {paper.title[:48]:48s}  {len(chunks):4d} chunks")
            result.papers_processed += 1
            continue

        existing = index.count_for_paper(paper.arxiv_id)
        if existing == len(chunks) and existing > 0 and not force:
            log(f"  {paper.arxiv_id}  up-to-date ({existing} chunks), skip")
            result.papers_skipped += 1
            continue

        vectors = embedder.encode([c.embed_text for c in chunks])
        index.upsert_chunks(chunks, vectors)
        index.prune_stale(paper.arxiv_id, keep_count=len(chunks))
        result.indexed_chunks += len(chunks)
        result.papers_processed += 1
        log(f"  {paper.arxiv_id}  {paper.title[:40]:40s}  indexed {len(chunks):4d} chunks")

    return result
