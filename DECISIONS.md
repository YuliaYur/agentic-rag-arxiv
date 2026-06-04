# Architecture Decision Log

Architecture Decision Records (ADRs) for `agentic-rag-arxiv`. Each entry is dated
and follows **Status · Context · Decision · Consequences**. Newest decisions
extend, and may supersede, older ones.

| ADR | Title | Status |
|---|---|---|
| 0001 | Repository structure, dependency & quality tooling | Accepted |
| 0002 | PDF parser — PyMuPDF | Accepted |
| 0003 | Structure-aware chunking | Accepted |
| 0004 | Embedding model — bge-small-en-v1.5 | Accepted |
| 0005 | Vector store — local Qdrant, idempotent indexing | Accepted |
| 0006 | Hybrid retrieval — dense + BM25 + RRF + cross-encoder rerank | Accepted |
| 0007 | Single-shot RAG baseline — structured, grounded, cited answers | Accepted |
| 0008 | Agentic answer graph (LangGraph) — grade + cite-critic loops | Accepted |

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
- **`DECISIONS.md` is this ADR log** for the project's design decisions.

**Consequences.** Contributors get one-command setup (`uv sync`), consistent
formatting enforced automatically, and a fast, deterministic test suite. Slight
overhead: the `src/` layout requires an editable install to import the package.

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

---

## ADR-0007 — Single-shot RAG baseline: structured, grounded, cited answers

**Status:** Accepted (2026-06-03)

**Context.** Before adding an agent (retrieve → grade → generate → cite-check, with
a re-retrieve loop), we need a **simple, measurable baseline**: retrieve →
stuff context → generate. Three needs: a swappable LLM client, machine-checkable
citations, and an honest "I don't know" path.

**Decision.** `src/agentic_rag/answer/` (baseline) + `src/agentic_rag/llm/` (client):

- **Thin LLM client** (`llm/client.py`) exposing only `structured(system, user,
  schema) -> PydanticModel`. The rest of the app never imports the OpenAI SDK
  directly, so routing through **LiteLLM** later (caching/fallbacks/multi-provider)
  touches only this file. Model: **`gpt-4o-mini`**, temperature 0 (faithful
  extraction, not creativity), `max_tokens` capped for cost.
- **Structured output** via OpenAI Structured Outputs bound to a Pydantic schema
  (`CitedAnswer`: `answer`, `citations[]`, `insufficient_context`). The provider
  returns schema-conforming JSON — validation happens at the API layer, so we get
  a typed object, not a string to parse.
- **Three-layer enforcement of "cite every claim, or abstain":**
  1. *Schema* — the model must return citations and an explicit
     `insufficient_context` flag (all fields required).
  2. *Prompt* — use only the numbered sources; put an inline `[S#]` after each
     claim; if the sources don't answer it, set `insufficient_context=true`,
     say so in one sentence, cite nothing, and never guess.
  3. *Programmatic validator* (`validate.py`, the real teeth) — sources are
     labeled `[S1]..[Sn]`, and that label is the join key: every citation **and**
     every inline `[S#]` marker must resolve to a retrieved chunk, or it's a
     **violation**; grounded citations are rebuilt from the chunk's authoritative
     metadata (so a model-fabricated arxiv_id/section/page is overwritten with the
     truth); `insufficient_context=false` with no grounded citation is a violation
     (no uncited claims), and `=true` with citations is a violation.

  We **cannot** verify every sentence has a citation without claim extraction
  (that's the upcoming cite-check critic's job), but we *can* guarantee nothing
  cited is fabricated and that the abstain path is honored.
- **CLI** `scripts/ask.py`; the retriever and LLM are injected into `SingleShotRAG`
  so the orchestration and all parsing/validation are unit-tested with fakes (no
  API calls, no spend). When retrieval returns nothing, it abstains **without**
  calling the LLM.

**Why a measurable baseline before agentic complexity.** An agent adds latency,
cost, and failure modes (loops, extra LLM calls). Without a baseline you can't
tell whether that complexity actually *helps* — you'd be guessing. The baseline
gives a fixed reference point so that, on the same golden set (`eval/`), we can
quantify the agent's marginal benefit (faithfulness, citation accuracy,
answer quality) against its marginal cost. If the agent doesn't beat this, it
isn't worth shipping. Hence the baseline is kept **intact** alongside the agent.

**Consequences.** Cheap, fast, honest single-shot answers with grounded citations
and a real abstain path; a clean seam for LiteLLM. Limitation: single-shot can't
recover from poor retrieval (no re-retrieve) and doesn't self-check claim coverage
— exactly the gaps the agent layer will target and be measured on.

---

## ADR-0008 — Agentic answer graph (LangGraph) with grade + cite-critic loops

**Status:** Accepted (2026-06-03)

**Context.** Single-shot RAG (ADR-0007) has two structural gaps: it can't recover
from weak retrieval, and it doesn't self-check that its claims are actually
supported. Multi-hop/comparative questions (needing facts from several papers)
expose both. We want an agent that can *re-retrieve* and *revise* — bounded so it
can't loop forever — built with **LangGraph** and reusing every existing piece.

**Decision.** `src/agentic_rag/agent/` — a LangGraph `StateGraph` over a typed
`AgentState` (TypedDict) with four nodes and two capped loops:

```
START → retrieve → grade_context ─(sufficient | round cap)→ generate → cite_critic ─(supported | rev cap)→ END
              ↑           │                                      ↑              │
              └─(weak & rounds left, reformulated query)─────────┘  └─(unsupported & revisions left)─┘
```

- **retrieve** — hybrid retrieval (ADR-0006), reused unchanged.
- **grade_context** — an LLM judges relevance/sufficiency (`GradeResult`). If weak,
  it emits a **reformulated query** and we loop back to retrieve (cap
  `max_retrieval_rounds=3`). The reformulated query drives *retrieval only*;
  `generate` always answers the **original** question.
- **generate** — the baseline's structured cited answer (ADR-0007), reused: same
  prompt, `CitedAnswer` schema, and grounding validator. On a revision it appends
  the critic's feedback to the prompt.
- **cite_critic** — an LLM auditor (`CriticResult`: supported? + score +
  unsupported_claims + feedback) checks every claim is backed by a cited source.
  If not, loop back to generate (cap `max_revision_rounds=2`).

**Design choices:**
- **State is explicit + typed**; caps live *in the state* so the routing functions
  (`route_after_grade`, `route_after_critic`) are **pure** (state → next-node) and
  unit-testable with no LLM.
- **Per-node structured metadata** is appended to `state["trace"]` via an
  `operator.add` reducer (retrieval_round, grade, sufficient, critic_score, …) —
  ready for Step 6 (Langfuse) and already surfaced by the CLI.
- **Two enforcement layers compose:** the programmatic grounding validator (from
  ADR-0007, in `generate`) guarantees citations aren't fabricated; the LLM
  `cite_critic` judges *claim coverage* — the part code can't check alone.
- Nodes are methods on `AgentNodes` (holding retriever + LLM) so LangGraph calls
  them with just state; dependencies are injected, so tests use fakes.

**Why this should beat the single-shot baseline (esp. multi-hop).** A comparative
question like "how does ELECTRA's objective differ from BERT *and* RoBERTa?" needs
chunks from ≥3 papers. One retrieval often misses a paper; single-shot then
answers from partial context. The agent's grader detects the gap and
**re-retrieves with a sharper query** until the context covers all hops, and the
critic catches **unsupported claims** and forces a revision. We'll *measure* this
on `eval/` (the multi-hop golden questions) against the baseline.

**Consequences.** Higher answer quality/faithfulness on hard questions, with a
visible reasoning trace — at the cost of **more LLM calls** (grade + generate +
critic, × loops) and higher latency. Both loops are hard-capped, and LangGraph's
`recursion_limit` is a final backstop. The single-shot baseline stays intact for
the eval comparison.
