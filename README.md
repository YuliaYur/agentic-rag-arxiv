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
# 0. (once) install deps + enable hooks
uv sync                          # or fallback: pip install -e ".[dev]"
pre-commit install               # ruff lint+format on commit

# 1. fetch the corpus into data/raw/ (skips files already present)
python scripts/fetch_corpus.py

# 2. start the local vector DB
docker compose up -d             # Qdrant on localhost:6333 (dashboard at /dashboard)

# 3. ingest: parse → chunk → embed → index (idempotent, re-runnable)
rag-ingest                       # or: python -m agentic_rag.ingest.cli

# inspect the index + run a sample query
python scripts/inspect_index.py --query "how does RoBERTa differ from BERT?"

# 4. hybrid retrieval (dense + BM25 + rerank)
python scripts/search.py "BLEU score for machine translation" --k 5 --compare

# 5. single-shot RAG baseline: cited answer (needs OPENAI_API_KEY in .env; paid call)
python scripts/ask.py "How does ELECTRA's objective differ from BERT's?"

# 6. agentic answer graph: grade + re-retrieve + cite-critic loops (paid; a few calls)
python scripts/agent_ask.py "How does ELECTRA's objective differ from BERT and RoBERTa?"
```

## Answering (single-shot baseline)

`retrieve → stuff context → generate` with a structured, **grounded** response:
the LLM returns an answer plus citations, and a validator enforces that every
citation and inline `[S#]` marker maps to a retrieved source (else it's flagged),
or the model must declare the context insufficient. Rationale in
[`DECISIONS.md`](DECISIONS.md) (ADR-0007).

```python
from agentic_rag.answer import build_baseline

rag = build_baseline(k=5)                 # wires retriever + LLM client
res = rag.answer("How does RoBERTa change BERT's pre-training?")
print(res.answer, res.is_grounded)
for c in res.citations:
    print(c.citation())
```

The LLM is reached through a thin client (`agentic_rag.llm`) designed to route
through LiteLLM later. This baseline is kept intact for eval comparison against
the agent.

## Agentic answer graph

A LangGraph state machine that adds two capped loops the baseline lacks —
**re-retrieve** when context is weak, and **revise** when claims aren't supported:

```
START → retrieve → grade_context ─(ok | cap)→ generate → cite_critic ─(supported | cap)→ END
              ↑           └─(weak, reformulate query)┘          ↑          └─(unsupported, revise)┘
```

`grade_context` reformulates the query and loops to `retrieve` (≤3 rounds);
`cite_critic` audits claim support and loops to `generate` (≤2 revisions). Each
node appends structured metadata to a `trace`. Rationale + why it beats the
baseline on multi-hop questions: [`DECISIONS.md`](DECISIONS.md) (ADR-0008).

```python
from agentic_rag.agent import build_agent, run_agent

app = build_agent()
final = run_agent(app, "How does ELECTRA's objective differ from BERT and RoBERTa?")
print(final["answer"].answer)        # cited answer
for e in final["trace"]:             # per-node control-flow metadata
    print(e["node"], e)
```

## Retrieval

Hybrid retrieval combines dense vector search (bge embeddings in Qdrant) with
in-memory BM25 keyword search, fuses them with **Reciprocal Rank Fusion**, and
reranks the top candidates with a local cross-encoder. Full rationale + a worked
"where dense fails" example in [`DECISIONS.md`](DECISIONS.md).

```python
from agentic_rag.retrieve import build_retriever

retriever = build_retriever()                 # build once (loads models + index)
hits = retriever.retrieve("GLUE benchmark", k=5)
for h in hits:
    print(h.score, h.citation(), h.text[:80])  # metadata intact for citation
```

CLI:

| Command | Effect |
|---|---|
| `python scripts/search.py "<query>"` | ranked results with scores + sources |
| `... --k 8` | return 8 results |
| `... --no-rerank` | fusion only (skip the cross-encoder) |
| `... --compare` | show dense-only vs hybrid side by side |

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

## Tests & quality

```bash
pytest                       # offline unit tests (no network, no model downloads)
ruff check . && ruff format --check .   # lint + format (also run by pre-commit)
```

Design decisions are recorded in [`DECISIONS.md`](DECISIONS.md); the evaluation
golden set lives in [`eval/`](eval/). Service config and API keys go in a local
`.env` (git-ignored).

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
src/agentic_rag/retrieve/
  config.py     # retrieval tunables (candidates, RRF k, reranker)
  models.py     # RetrievedChunk (result + scores + metadata)
  dense.py      # Qdrant dense search + chunk loader
  bm25.py       # in-memory BM25 keyword search
  fusion.py     # Reciprocal Rank Fusion
  rerank.py     # cross-encoder reranker
  retriever.py  # HybridRetriever.retrieve(query, k); build_retriever()
src/agentic_rag/llm/      # thin LLM client (LiteLLM-routable)
src/agentic_rag/answer/   # single-shot RAG baseline (schemas, prompt, validate)
src/agentic_rag/agent/    # LangGraph agent (state, nodes, routing, graph)
scripts/
  fetch_corpus.py    # reproducible corpus download
  inspect_index.py   # index stats + sample query
  search.py          # hybrid retrieval from the CLI
  ask.py             # single-shot RAG baseline (cited answer)
  agent_ask.py       # agentic answer graph (with control-flow trace)
tests/               # offline unit tests (chunking, metadata, fusion, bm25, retriever, answer, agent)
```
