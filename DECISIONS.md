# Architecture Decision Log

Architecture Decision Records (ADRs) for `agentic-rag-arxiv`. Each entry is dated
and follows **Status · Context · Decision · Consequences**. Newest decisions
extend, and may supersede, older ones. Project overview and current status live
in [`CLAUDE.md`](CLAUDE.md).

| ADR | Title | Status |
|---|---|---|
| 0001 | Repository structure, dependency & quality tooling | Accepted |
| 0002 | PDF parser — PyMuPDF | Accepted |
| 0003 | Structure-aware chunking | Accepted |
| 0004 | Embedding model — bge-small-en-v1.5 | Accepted |
| 0005 | Vector store — local Qdrant, idempotent indexing | Accepted |
| 0006 | Hybrid retrieval — dense + BM25 + RRF + cross-encoder rerank | Accepted |

---

## ADR-0001 — Repository structure, dependency & quality tooling

**Status:** Accepted (2026-06-03)

**Context.** This is a portfolio project targeting production-grade (4/5)
maturity, built incrementally across many sessions, with the owner learning from
the choices. It needs a structure and toolchain that stay maintainable as the
agent graph, eval suite, and services land — not a single notebook.

**Decision.**
- **`src/` layout** (`src/agentic_rag/…`) with `tests/`, `eval/`, `scripts/`,
  `data/`. The src layout prevents accidental imports of the un-installed
  package and forces tests to run against the installed code.
- **Dependency management: `uv`.** Fast, standards-based (`pyproject.toml` +
  `uv.lock`), one tool for venv + resolution + lock. Chosen over **poetry**
  (slower; historically fought packaging standards). `pip install -e ".[dev]"`
  still works as a fallback so no one is forced onto uv.
- **Lint + format: `ruff`** (lint *and* `ruff format`, replacing black/isort/
  flake8 with one fast tool), line length 100, configured in `pyproject.toml`.
- **`pre-commit`** runs ruff (lint+format) plus hygiene hooks (trailing
  whitespace, EOF, large-file guard) on every commit.
- **Tests offline by design:** pure logic is unit-tested with no network or model
  downloads; heavy components are injected and faked. Network/model paths are
  exercised by scripts, not the test suite.
- **`DECISIONS.md` is this ADR log;** `CLAUDE.md` is the living status/convention
  doc.

**Consequences.** Contributors get one-command setup (`uv sync`), consistent
formatting enforced automatically, and a fast, deterministic test suite. Slight
overhead: the `src/` layout requires an editable install to import the package.
Commits in this repo are authored by YuliaYur only (no co-author trailer).

---

## ADR-0002 — PDF parser: PyMuPDF

**Status:** Accepted (2026-06-03)

**Context.** The corpus PDFs are messy: **two-column layouts, inline/display
equations, tables, figure captions, dense reference sections.** The parser must
recover correct *reading order* (not interleave columns) and ideally expose
*structure* (headings) so chunks can respect sections.

**Decision.** Use **PyMuPDF (fitz)**.

| Parser | Layout / reading order | Local & free | Deps & speed | Verdict |
|---|---|---|---|---|
| **PyMuPDF** | Exposes block bounding boxes + per-span font sizes — enough to rebuild column order and detect headings ourselves | ✅ no key | ✅ one wheel, no system deps, very fast | **Chosen** |
| Unstructured (`hi_res`) | Best out-of-the-box (detectron2 layout model) | ✅ | ❌ heavy: detectron2/onnx + poppler + tesseract; slow; awkward on Windows | Rejected — operational cost ≫ marginal gain here |
| LlamaParse | Excellent, tables→markdown | ❌ cloud API, paid at volume, sends PDFs to a third party | n/a | Rejected — breaks "local + free"; licensing concern |

We layer our own transparent, testable logic on the two raw signals PyMuPDF
gives us: bounding boxes → **reading order** (`order_blocks`: full-width blocks
act as band separators; between them, left column then right column), and font
size + a numbered-heading regex → **section detection** (`heading_kind`; a
font-only "heading" before the first real section is treated as the paper title,
not a section).

**Consequences.** Fast, dependency-light, fully local parsing with structure we
control. **Tradeoffs:** no semantic table/equation understanding (acceptable for
prose QA); heuristics occasionally mis-tag frontmatter (e.g. an affiliation line
as a heading) — affects a chunk's `section` label, not retrievability. Upgrade
path if faithful tables are ever needed: swap an Unstructured `hi_res` backend
behind the same `Block` interface; the rest of the pipeline is parser-agnostic.

---

## ADR-0003 — Structure-aware chunking

**Status:** Accepted (2026-06-03)

**Context.** Chunk size and boundaries directly drive retrieval quality. Whole
papers are too big; single sentences too fragmented. Chunks also must carry
citation metadata.

**Decision.** Section-respecting, sentence-boundary chunking (`ingest/chunk.py`).
Defaults (`ChunkConfig`):

| Param | Value | Why |
|---|---|---|
| `target_tokens` | 384 | Comfortably under the 512-token model window, with headroom for the context header. |
| `overlap_tokens` | 64 (~17%) | Facts straddling a boundary stay retrievable from either side. |
| `max_tokens` | 480 | Hard ceiling enforced during packing — **0/1150 chunks exceed 512**. Oversize "sentences" (linearized equations) are word-split. |
| `min_tokens` | 48 | Tiny trailing fragments merge into the previous chunk. |
| `respect_sections` | True | A chunk is always about one section → coherent embedding + exact citation. |
| `drop_references` | True | The bibliography is noisy for QA and dilutes retrieval. |
| `prepend_context` | True | Embed `"<title> > <section>\n<text>"`; store clean text. Cheap, well-established retrieval win. |

Token counting is **injected**: the live pipeline passes the embedding model's
own tokenizer (so packing targets the true window); tests/offline use a fast
regex counter (deterministic, no download). A noise floor drops sub-5-token
fragments (stray page numbers, figure labels).

Every chunk carries: `text, arxiv_id, title, slug, section, page, page_end,
chunk_index, n_tokens, content_hash`. Title/arxiv_id come from local files
(`manifest.json` + `SOURCES.md`), more reliable than PDF metadata.

**Consequences.** ~1,150 coherent, citable chunks (median ≈ 346 tokens, none over
the window). Sentence splitting is regex-based (imperfect on abbreviations) —
acceptable since boundaries only need to be reasonable.

**Tuning (if retrieval is poor):** answers miss detail / wrong span → *decrease*
`target_tokens` (384→256); answers fragmented → *increase* (→512); boundary facts
missed → *raise* overlap; citation-style questions → set `drop_references=False`.
Always evaluate against `eval/` and re-run `rag-ingest --force` (idempotent).

---

## ADR-0004 — Embedding model: BAAI/bge-small-en-v1.5

**Status:** Accepted (2026-06-03)

**Context.** Embeddings must be local and free (constraint), run on CPU, and be
good enough for paper QA retrieval.

**Decision.** **`BAAI/bge-small-en-v1.5`** — 384-dim, 512-token context, ~33M
params, normalized embeddings (cosine = dot product). Strong MTEB retrieval for
its size; embeds the whole corpus in ~2 min on CPU.

**Consequences.** Fast, free, local. **Query-side caveat:** bge-v1.5 wants the
instruction prefix *"Represent this sentence for searching relevant passages: "*
on the **query** only — the retrieval layer adds it. Alternative if speed ever
dominates: `all-MiniLM-L6-v2` (same dim, faster, weaker); for more recall:
`bge-base-en-v1.5` (768-dim, requires re-indexing with `--recreate`).

---

## ADR-0005 — Vector store: local Qdrant, idempotent indexing

**Status:** Accepted (2026-06-03)

**Context.** Need a free, self-hostable vector store that keeps vectors *and*
citation metadata together, with re-runnable ingestion.

**Decision.** **Qdrant** via `docker-compose.yml` on `localhost:6333`, storage
persisted to `./data/qdrant` (git-ignored). Collection `arxiv_papers`: 384-dim,
**cosine**, with a keyword payload index on `arxiv_id`. Chosen over FAISS (a
library — you manage persistence + metadata yourself), Chroma (lighter, less
robust), and Pinecone/Weaviate Cloud (paid/hosted — breaks constraints).

**Idempotency:** deterministic point ids (`uuid5("arxiv_id:chunk_index")`) so
re-runs upsert, never duplicate; a paper already indexed with the same chunk
count is skipped unless `--force`; a stale-tail prune deletes orphaned points if
re-chunking yields fewer chunks. CLI `rag-ingest` with tqdm progress; `--dry-run`
parses+chunks offline.

**Consequences.** One-command local vector DB, citations available from the
payload without a second lookup, safe re-runs. The Dockerized server is the
production-shaped choice; a no-Docker `QdrantClient(path=…)` local mode exists
for quick experiments.

---

## ADR-0006 — Hybrid retrieval: dense + BM25 → RRF → cross-encoder rerank

**Status:** Accepted (2026-06-03)

**Context.** Dense (semantic) search alone misses queries that hinge on exact
terms — model names, datasets, metrics, symbols. We want both semantic and
lexical matching, fused and reranked, behind a clean interface.

**Decision.** `HybridRetriever.retrieve(query, k) -> list[RetrievedChunk]`
(`src/agentic_rag/retrieve/`):

```
dense (bge cosine, Qdrant) top-50 ┐
                                  ├─ RRF fuse → top-30 → cross-encoder rerank → top-k
BM25 (in-memory) top-50           ┘
```

- **Dense + BM25.** Dense reuses the bge index; **BM25 runs in-memory**
  (`rank_bm25` over chunk texts loaded once from Qdrant) rather than as Qdrant
  sparse vectors — at ~1,150 chunks it's instant, keeps the dense index
  untouched, and is pure/testable offline. *Scale-up path:* Qdrant native sparse
  + server-side fusion, behind the same interface.
- **Fusion: Reciprocal Rank Fusion** (`rrf(d)=Σ 1/(k+rank), k=60`). Chosen over
  score-based fusion because dense cosine (~0–1) and BM25 (unbounded) are on
  incomparable scales; RRF fuses on **rank**, so it's parameter-light and robust,
  and a chunk ranked well by both retrievers rises.
- **Rerank: cross-encoder** `cross-encoder/ms-marco-MiniLM-L-6-v2` (free, local,
  ~80MB). Bi-encoders score query/passage independently (cheap, coarse); a
  cross-encoder scores the pair *together* (accurate) but costs one inference
  **per candidate** — so it runs only on the **top-30** fused candidates. Classic
  retrieve-broad-then-rerank-narrow: near-cross-encoder quality at
  near-bi-encoder latency.

**Worked example — where dense alone fails, hybrid wins.** Query *"BLEU score for
machine translation"*: dense treated *BLEU* as generic "translation" topic and
buried the Transformer **Abstract** (which states the SOTA BLEU result) at rank
**#14**, filling top slots with GPT-3's raw numeric BLEU *tables*. **BM25** ranked
that Abstract chunk **#1** on the exact token *BLEU*; **RRF + rerank** promoted it
into the top-3 and surfaced Transformer §5.4 (dense had it at #10). Lesson: for
term-hinged queries, dense blurs the exact token; BM25 anchors it; fusion+rerank
gets the best of both.

**Consequences.** Robust across semantic and lexical queries; each result carries
full citation metadata + score provenance (dense/BM25 rank, rerank score). Cost:
the reranker adds a few hundred ms on CPU per query; `build_retriever()` loads
models + scrolls the index once (reuse it).

**Tuning:** recall too low → raise `dense_candidates`/`bm25_candidates`; top
result *almost* right → raise `rerank_candidates` or upgrade the reranker
(`bge-reranker-base`); too slow → lower `rerank_candidates` or
`use_reranker=False` (compare with `search.py --compare`).
