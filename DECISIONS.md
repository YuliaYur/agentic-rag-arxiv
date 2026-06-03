# Ingestion decisions

Design decisions for the ingestion pipeline (`src/agentic_rag/ingest/`), with the
reasoning and tradeoffs behind each. The pipeline turns the arXiv PDFs in
`data/raw/` into a citable Qdrant vector index:

```
PDF  →  parse (PyMuPDF)  →  structure-aware chunk  →  embed (bge-small)  →  Qdrant
```

Run it with `rag-ingest` (see README.md). Everything tunable lives in
`ingest/config.py`.

---

## 1. PDF parser: **PyMuPDF**, over Unstructured and LlamaParse

Our PDFs are messy in a specific way: **two-column layout, inline + display
equations, tables, figure captions, and dense reference sections**. The parser
has to recover *reading order* (so the right column doesn't interleave with the
left) and ideally expose *structure* (headings) so chunks can respect sections.

| Parser | Layout / reading order | Local & free | Deps & speed | Verdict |
|---|---|---|---|---|
| **PyMuPDF (fitz)** | Good — exposes block bounding boxes + per-span font sizes, enough to rebuild column order and detect headings ourselves | ✅ fully local, no key | ✅ single wheel, no system deps, very fast (~2–10 papers/s to parse) | **Chosen** |
| Unstructured (`hi_res`) | Best out-of-the-box — detectron2 layout model handles columns/tables natively | ✅ local | ❌ heavy: pulls detectron2/onnx, needs poppler + tesseract system installs; slow; awkward on Windows | Rejected — operational cost too high for the marginal layout gain here |
| LlamaParse | Excellent, including tables→markdown | ❌ **cloud API, needs a key, not free at volume**; sends PDFs to a third party | n/a | Rejected — violates the "runs locally for free" requirement and adds an external dependency + licensing concern (see SOURCES.md) |

**Why PyMuPDF wins for *this* corpus:** it gives us the two raw signals we
actually need — block bounding boxes and font sizes — with zero system
dependencies and high speed, and we layer our own (testable, transparent)
reading-order and heading logic on top. We control the structure extraction
instead of trusting an opaque model, and there's no cloud round-trip.

**Tradeoffs we accept (documented honestly):**
- PyMuPDF does **not** understand tables or math semantically. Equations come
  through as best-effort linearized text; tables come through as text blocks,
  not structured cells. For a QA-over-papers RAG that's acceptable — we retrieve
  prose; we don't need to reconstruct a table's grid.
- Our reading-order and heading heuristics (below) are *heuristics*. They handle
  the common arXiv layout well but occasionally mis-tag frontmatter (an author
  affiliation line can look like a heading). This affects the `section` label on
  a few frontmatter chunks, not retrievability of the text.
- If we later need faithful table extraction, the clean upgrade path is to swap
  `parse.py` for an Unstructured `hi_res` backend behind the same `Block`
  interface — the rest of the pipeline is parser-agnostic.

**How reading order is reconstructed** (`parse.order_blocks`): walk blocks
top-to-bottom; a block wider than 55% of the page is treated as *full-width*
(title, abstract, wide table/figure) and emitted in place as a band separator.
Between separators, the buffered blocks are split into left/right columns by
horizontal center and emitted **left column fully, then right column**, each
top-to-bottom. This is the correct reading order for standard two-column papers.

**How sections are detected** (`parse.heading_kind`): primarily a
numbered-heading regex (`"3 Pre-training"`, `"3.1 ..."`) which is extremely
reliable on arXiv, plus a set of known unnumbered headings (`Abstract`,
`Introduction`, `References`, …), with font-size as a corroborating signal. A
font-only "heading" seen before the first real section is treated as the paper
*title* (frontmatter), not a section, so the title doesn't masquerade as a
section.

---

## 2. Structure-aware chunking

Implemented in `ingest/chunk.py`. Three principles:

1. **Never cross a section boundary.** A chunk is always about one section. This
   keeps each embedding semantically coherent and lets us cite an exact section.
2. **Split at sentence boundaries.** Packing works on sentences, so chunks don't
   end mid-thought and overlap is a clean sentence carry-over.
3. **Carry full citation metadata** on every chunk (next section).

### Chosen parameters and *why*

Defaults in `ChunkConfig`:

| Param | Value | Why |
|---|---|---|
| `target_tokens` | **384** | Comfortably under the embedding model's 512-token window, leaving headroom for the context header (below) without truncation. Big enough to hold a full idea/paragraph, small enough that a hit is specific. |
| `overlap_tokens` | **64** (~17%) | A fact that straddles a chunk boundary (definition in one chunk, its use in the next) stays retrievable from either side. ~15–20% is the common sweet spot; more wastes index space, less risks losing boundary facts. |
| `max_tokens` | **480** | Hard ceiling enforced during packing so **no chunk ever exceeds the model window** (verified: 0/1153 chunks over 512 tokens). Oversize single "sentences" (e.g. a linearized equation) are word-split to respect it. |
| `min_tokens` | **48** | A trailing fragment smaller than this is merged into the previous chunk so we don't index thin, low-signal stubs. |
| `respect_sections` | **True** | Principle 1 above. |
| `drop_references` | **True** | The bibliography is low-value and noisy for QA ("[35, 2, 5]", author lists) and would dilute retrieval. Dropped by default. |
| `prepend_context` | **True** | What gets *embedded* is `"<title> > <section>\n<chunk text>"`; what gets *stored/returned* is the clean text. The header injects document/section context so a chunk that says "we improve over the baseline by 2.1 points" is embedded knowing *which paper and section* it's from — a cheap, well-established retrieval win. |

**Token counting is real, not guessed.** Chunking takes an injected
`token_counter`. The live pipeline passes the **embedding model's own
tokenizer**, so packing targets the true context window. Tests and offline
dry-runs use a fast regex word/punctuation counter (deterministic, no model
download). This is why the chunking logic is fully unit-testable without the
network.

### Resulting corpus shape

20 papers → **~1,150 chunks**; chunk size median ≈ 346 tokens, mean ≈ 289, max
512, **none over the model window**. Larger papers with big appendices (T5,
GPT-3, CLIP) produce proportionally more chunks, as expected.

---

## 3. Chunk metadata (for citation)

Every chunk carries (stored as the Qdrant point payload):

`text`, `arxiv_id`, `title`, `slug`, `section`, `page`, `page_end`,
`chunk_index`, `n_tokens`, `content_hash`.

- **title / arxiv_id** come from local files only (`manifest.json` +
  `SOURCES.md`), joined on `arxiv_id` — no network, and more reliable than the
  PDF's own (often missing) title metadata.
- **section / page / page_end** come from the parser, so a retrieved chunk can
  be cited as *"Longformer (2004.05150), §3, p.3"*. A chunk that spans a page
  break reports both `page` and `page_end`.
- **content_hash** (sha1 of the text) supports change detection / dedup.

---

## 4. Embeddings: **BAAI/bge-small-en-v1.5**

Local, free, runs on CPU.

- **384-dim**, 512-token context, ~33M params.
- Strong MTEB retrieval scores for its size — punches well above all-MiniLM for
  retrieval quality while staying small and fast.
- **Speed:** loads in seconds; embeds the whole 20-paper corpus (~1,150 chunks)
  in a couple of minutes on CPU. GPU not required.
- **Normalized** embeddings → cosine similarity is a dot product.

**Query-side note:** bge-v1.5 expects the instruction prefix *"Represent this
sentence for searching relevant passages: "* on the **query** only (passages are
embedded plain). The retrieval layer must add it — see `scripts/inspect_index.py`
for the reference usage.

**Alternative considered:** `sentence-transformers/all-MiniLM-L6-v2` — also
384-dim, even faster and lighter, but weaker retrieval quality. Good fallback if
embedding speed ever dominates; swap `EmbedConfig.model_name` (same dimension, no
other changes needed).

---

## 5. Index: local Qdrant via Docker

- `docker-compose.yml` runs Qdrant on `localhost:6333`, storage persisted to
  `./data/qdrant` (git-ignored).
- Collection `arxiv_papers`: 384-dim vectors, **cosine** distance. A keyword
  payload index on `arxiv_id` makes per-paper count/prune/filter fast.
- Both the vector **and** the full metadata payload are stored, so retrieval can
  cite sources without a second lookup.

### Idempotency & re-runnability

- **Deterministic point ids:** `uuid5("arxiv_id:chunk_index")`. Re-running
  upserts the same ids — never duplicates.
- **Skip-if-unchanged:** a paper already indexed with the same chunk count is
  skipped unless `--force`.
- **Stale-tail prune:** after re-indexing a paper we delete any points whose
  `chunk_index` is beyond the new count, so re-chunking with different parameters
  (fewer chunks) leaves no orphans.
- **CLI + progress:** `rag-ingest` with a tqdm progress bar and per-paper
  summary lines; `--dry-run` parses+chunks offline (no model, no Qdrant) for
  quick inspection.

---

## 6. If retrieval quality is poor later — how to tune

In rough order of impact:

1. **Chunk size (`--target-tokens`).** Symptoms drive the direction:
   - *Answers miss detail / retrieve the right paper but the wrong span* →
     **decrease** (e.g. 384 → 256). Smaller chunks = more precise hits.
   - *Answers feel fragmented / lack surrounding context* → **increase** (e.g.
     384 → 512, the model max). Bigger chunks = more context per hit.
2. **Overlap (`--overlap-tokens`).** If facts that span a boundary get missed,
   **raise** overlap (64 → 96/128). If the index is bloated with near-duplicate
   neighbors, **lower** it.
3. **Context header (`prepend_context`).** On by default. If retrieval confuses
   similar claims across papers, the title/section header helps; if you suspect
   it's biasing matches toward titles, A/B it off.
4. **Keep vs. drop references / appendices (`drop_references`).** If users ask
   "what does paper X cite about Y", turn dropping **off**. For pure conceptual
   QA, keep it **on**.
5. **Embedding model (`--model`).** If small-model recall is the ceiling, step up
   to `bge-base-en-v1.5` (768-dim) — better recall at higher index size and
   slower embedding. (Changing dimension requires `--recreate`.)
6. **Hybrid / re-ranking (next layer, not ingestion).** Biggest wins usually
   come *after* ingestion: add BM25/keyword hybrid search for exact-term queries
   (model names, metric names), and a cross-encoder re-ranker over the top-k.
   The chunk granularity chosen here is deliberately re-ranker-friendly.
7. **Evaluate, don't guess.** Tune against the golden eval set (`eval/`, per
   SOURCES.md). Change one parameter, re-run `rag-ingest --force`, re-measure
   retrieval recall@k. The pipeline is idempotent precisely so this loop is cheap.

---

# Retrieval decisions

The retrieval layer (`src/agentic_rag/retrieve/`) sits on top of the index. The
funnel:

```
              dense (bge cosine, Qdrant) top-50 ┐
                                                ├─ RRF fuse ─→ top-30 ─→ cross-encoder rerank ─→ top-k
              BM25 (in-memory) top-50           ┘
```

Public interface: `HybridRetriever.retrieve(query, k) -> list[RetrievedChunk]`,
where each `RetrievedChunk` carries the full source metadata (title, arxiv_id,
section, page/page_end) plus scores. Build once with `build_retriever()`; query
many times. CLI: `python scripts/search.py "<query>"`.

## 8. Dense + BM25, and why **both**

- **Dense** (our bge embeddings) matches on *meaning* — great for paraphrase and
  concept queries ("how is attention made cheaper?" finds "efficient
  self-attention" even with no shared words).
- **BM25** matches on *exact terms*, weighted by rarity — great for the things
  embeddings blur: model names (RoBERTa vs BERT), datasets (GLUE, SQuAD),
  metrics (BLEU), symbols, acronyms.

They fail in opposite directions, so combining them covers both query types.

**BM25 runs in-memory** (`rank_bm25` over the chunk texts loaded from Qdrant),
not as Qdrant sparse vectors. Rationale: the corpus is ~1,150 chunks, so
in-process BM25 is instant; it keeps the existing dense index untouched (no
re-ingest), and the logic is pure and unit-testable offline. **Tradeoff:** it
rebuilds the BM25 index in memory at `build_retriever()` time (sub-second here)
and won't scale to millions of chunks. The clean upgrade at scale is Qdrant's
native sparse vectors + server-side Query-API fusion — same `retrieve()`
interface, swap the implementation behind it.

## 9. Fusion: **Reciprocal Rank Fusion (RRF)**

`rrf_score(d) = Σ 1 / (k + rank_r(d))` over each retriever r; default k=60.

Chosen over score-based fusion (e.g. weighted sum of normalized scores) because
dense cosine (~0–1) and BM25 (unbounded, corpus-dependent) live on
incomparable scales — normalizing them is brittle and needs per-corpus tuning.
RRF ignores raw scores and fuses on **rank position**, so it's parameter-light,
robust, and a strong default. A chunk ranked well by *both* retrievers
accumulates from both lists and rises — exactly the hybrid behavior we want.

## 10. Reranking: cross-encoder, and the latency/quality tradeoff

- Dense and BM25 are **bi-encoders / bag-of-words**: query and passage are scored
  independently, so retrieval is cheap but the relevance judgment is coarse.
- A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, free/local/~80MB)
  feeds *(query, passage) together* through a transformer, so it judges true
  relevance far more accurately.

**The tradeoff:** a cross-encoder cannot precompute anything — it runs one model
inference *per candidate*. Scoring all 1,150 chunks per query would be slow. So
reranking is the **last, narrow stage**: fetch broadly and cheaply (dense+BM25),
fuse, then rerank only the **top ~30** fused candidates. This is the standard
"retrieve-then-rerank" pattern — near-cross-encoder quality at near-bi-encoder
latency. On CPU, reranking 30 short passages adds roughly a few hundred ms;
`rerank_candidates` and `use_reranker` trade that latency against quality. (For
higher quality at more cost: `BAAI/bge-reranker-base`.)

## 11. Worked example — where dense alone fails and hybrid wins

Query: **"BLEU score for machine translation"** (BLEU = an exact metric token).

- **Dense-only top-5** anchored on the *concept* "translation evaluation": it put
  the Transformer's **Abstract** — which literally states the new SOTA *BLEU*
  result — all the way down at **rank #14** (off the list), and instead filled
  slots with GPT-3's raw numeric BLEU *tables* (walls of digits, useless as an
  answer) and an off-topic T5 "inter-run variance" chunk.
- **BM25** ranked that Transformer Abstract chunk **#1** — because it contains the
  literal token *BLEU* — and also surfaced Transformer §5.4 ("Table 2: …better
  BLEU scores than previous state-of-the-art", which dense had at #10).
- **RRF + rerank** merged the two: the actual result-bearing chunks rose into the
  top-3, displacing the number-dumps.

The general lesson: for queries that hinge on a precise term, dense embeddings
treat the term as generic topic signal and can bury the exact-match chunk; BM25
anchors on the literal token, and fusion+rerank gets the best of both.

## 12. Tuning the retrieval layer

- **Recall too low (right chunk never retrieved):** raise `dense_candidates` /
  `bm25_candidates` (50 → 100) so fusion sees a wider net.
- **Exact-term queries underperform:** the BM25 half is the lever — verify
  tokenization keeps your terms intact.
- **Top result often *almost* right:** raise `rerank_candidates` (30 → 50) so the
  reranker can reach deeper, or upgrade the reranker model.
- **Too slow:** lower `rerank_candidates`, or set `use_reranker=False` to fall
  back to fusion-only (faster, lower quality — compare with `search.py --compare`).
