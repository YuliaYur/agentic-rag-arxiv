# agentic-rag-arxiv

Agentic RAG over a curated corpus of 20 transformer-lineage arXiv papers. This
repo currently implements the **ingestion pipeline**: PDFs → structure-aware
chunks → local embeddings → a Qdrant vector index that downstream retrieval can
cite.

See [`SOURCES.md`](SOURCES.md) for the corpus and [`DECISIONS.md`](DECISIONS.md)
for the design rationale (parser choice, chunking parameters, tuning guide).

## Pipeline

```
data/raw/*.pdf  →  parse (PyMuPDF)  →  chunk (section-aware)  →  embed (bge-small-en-v1.5)  →  Qdrant
```

## Quickstart

```bash
# 0. (once) create + activate a venv, then install
pip install -e ".[dev]"          # or: pip install -r requirements.txt

# 1. fetch the corpus into data/raw/ (skips files already present)
python scripts/fetch_corpus.py

# 2. start the local vector DB
docker compose up -d             # Qdrant on localhost:6333 (dashboard at /dashboard)

# 3. ingest: parse → chunk → embed → index (idempotent, re-runnable)
rag-ingest                       # or: python -m agentic_rag.ingest.cli

# inspect the index + run a sample query
python scripts/inspect_index.py --query "how does RoBERTa differ from BERT?"
```

### CLI

`rag-ingest` (entry point) / `python -m agentic_rag.ingest.cli`:

| Flag | Effect |
|---|---|
| *(none)* | parse + chunk + embed + index everything; skips papers already up-to-date |
| `--dry-run` | parse + chunk only — no model, no Qdrant (offline inspection) |
| `--papers 1706.03762 bert` | limit to specific papers (by arxiv_id or slug) |
| `--force` | re-embed + re-index even if up-to-date |
| `--recreate` | drop and rebuild the Qdrant collection first |
| `--target-tokens N` / `--overlap-tokens N` | override chunking on the fly |
| `--examples N` | print N example chunks with metadata at the end |

Re-runs are **idempotent**: deterministic point ids mean re-running never
duplicates, and a stale-tail prune keeps the index consistent if chunking
parameters change.

## Tests

```bash
pytest            # chunking + metadata + corpus/parse logic (no network, no model)
```

## Layout

```
src/agentic_rag/ingest/
  config.py     # all tunables (chunk sizes, model, Qdrant)
  corpus.py     # load arxiv_id/slug/title from manifest.json + SOURCES.md
  parse.py      # PyMuPDF: reading order + heading/section detection
  chunk.py      # structure-aware chunking + chunk metadata
  embed.py      # local sentence-transformers embeddings
  index.py      # idempotent Qdrant upsert
  pipeline.py   # orchestration
  cli.py        # rag-ingest entry point
scripts/
  fetch_corpus.py    # reproducible corpus download
  inspect_index.py   # index stats + sample query (retrieval smoke test)
tests/               # offline unit tests
```
