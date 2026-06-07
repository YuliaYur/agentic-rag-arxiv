<h1 align="center">agentic-rag-arxiv</h1>

<p align="center">
  <b>Ask a question about 20 landmark AI research papers — get a short, cited, fact-checked answer.</b>
</p>

<p align="center">
  <a href="https://github.com/YuliaYur/agentic-rag-arxiv/actions/workflows/eval-gate.yml"><img src="https://img.shields.io/github/actions/workflow/status/YuliaYur/agentic-rag-arxiv/eval-gate.yml?branch=main&label=eval%20gate&style=for-the-badge&logo=githubactions&logoColor=white" alt="eval gate"></a>
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python 3.11+">
  <img src="https://img.shields.io/badge/run-docker%20compose%20up-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="docker compose up">
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/ab645b42-6774-429e-8f71-6d39b812bf44" controls muted width="820"></video>
</p>

---

## What this is

Large language models (like ChatGPT) are confident even when they're wrong — they
make facts up. The standard fix is **RAG** (*Retrieval-Augmented Generation*): before
the model answers, you **search a library of trusted documents** and hand it the
relevant passages, so the answer is built from real sources instead of memory.

This project is RAG over a small, carefully chosen library: **20 of the most
important AI research papers** — the family of ideas behind modern language models
(*"Attention Is All You Need"*, BERT, GPT, and their descendants). You type a
question; the system finds the right passages and writes a short answer **with
clickable citations back to the exact paper and section**.

The twist is the **"agentic"** part. A plain RAG system searches once and answers
once. This one behaves more like a careful researcher:

- it **reads its own draft** and checks every claim is actually backed by a source —
  if a sentence isn't grounded, it rewrites it;
- if a question spans **two papers** (*"how is ELECTRA different from BERT?"*), it
  notices that it only found one and **goes back to search for the other**;
- if it genuinely doesn't have enough to answer, it **says so instead of bluffing**.

So every answer is either **grounded and cited, or politely declined** — never a
confident hallucination. That self-checking loop is the difference between a demo and
something you'd actually trust.

---

## How it's built (and the tech behind it)

Here's the toolbox, with a one-line explanation of each piece — so even outside the
AI world you can map it to whatever your team already uses.

| Term                             | What it is | Used here for |
|----------------------------------|---|---|
| **RAG**                          | *Retrieval-Augmented Generation* — search documents first, then let the LLM answer from them | the core pattern of the whole system |
| **Qdrant**                       | a **vector database** (stores text as numbers so you can search by *meaning*, not just keywords; alternatives: Pinecone, Weaviate, pgvector) | holding the searchable index of all 20 papers |
| **Hybrid search + reranking**    | combines meaning-based search with classic keyword search (**BM25**), merges the two rankings, then a second model re-sorts the top hits for precision | finding the *right* passage, not just a related one |
| **LangGraph**                    | a framework for building **agents** as a flow chart of steps with loops (search → check → rewrite → re-search) | the self-checking "researcher" loop |
| **LiteLLM + Redis**              | one unified gateway in front of many LLM providers, with **routing** (cheap model for easy steps, strong model for the final answer) and a **semantic cache** (reuse answers to similar questions) | cutting cost and latency |
| **Guardrails**                   | safety checks on the way in and out — blocking prompt-injection hidden in documents, and refusing to answer when confidence is low | trust and safety |
| **Langfuse**                     | self-hosted **observability** — a dashboard that traces every step, token, and dollar of a request (like Datadog, but for LLM apps) | seeing *why* the agent did what it did |
| **RAGAS-style eval + LLM-judge** | automated **quality scoring** for AI answers (more on this below) | proving the answers are actually good |
| **FastAPI + Streamlit**          | a Python web **API** and a quick **web UI** | the service and the demo you saw above |
| **Docker Compose**               | runs the whole multi-service stack with one command | one-command setup, no manual install |
| **CI/CD (GitHub Actions)**       | automated pipeline that runs on every change — here it re-runs the evaluation and **blocks the merge if answer quality drops** | treating answer quality like a test |

**The flow, end to end:**

![Architecture](docs/architecture.svg)

> A question comes in → the agent searches Qdrant (meaning + keyword, merged and
> reranked) → grades whether it found enough, re-searching if not → drafts an answer →
> critiques its own citations and rewrites weak claims → applies output guardrails →
> returns a structured, cited answer. Every step is traced in Langfuse and routed
> through LiteLLM for cost control.

---

## Run it yourself — one command

Everything runs locally in containers. You don't need to download the papers, build
an index, or install Python machine-learning libraries by hand — the search index
ships *inside the repo* and loads on startup.

```bash
git clone https://github.com/YuliaYur/agentic-rag-arxiv && cd agentic-rag-arxiv
cp .env.example .env          # paste your OPENAI_API_KEY into the file
docker compose up --build     # first build downloads the models, then it's instant
```

Then open:

- **the demo UI** → http://localhost:8501
- **the API docs** (interactive, try-it-yourself) → http://localhost:8000/docs

That's it. To stop: `docker compose down`.

**Why Docker matters here.** [Docker](https://www.docker.com/) packages an app *and*
all its dependencies into a portable container, so "works on my machine" becomes
"works on every machine." This project needs **several services running together** —
the API, the web UI, the vector database, and a cache — and `docker compose`
**orchestrates all of them at once**, in the right startup order, wired together
automatically. A few details I'm proud of:

- **One image, several containers.** The API, the UI, and the one-shot index-loader
  are all the *same* lightweight image — built once, reused three times. Smaller
  download, faster builds, nothing duplicated.
- **Self-loading index.** A throwaway `index-init` container loads the bundled index
  into Qdrant on first boot, then exits — so search works immediately with **zero
  setup**.
- **Correct startup order**, enforced by health-checks: database → load index → API →
  UI. No race conditions, no "connection refused" on cold start.
- **Optional observability** is one flag away:
  `docker compose --profile observability up` adds the Langfuse tracing dashboard at
  http://localhost:3000.

---

## Does it actually work? (results & how they're measured)

A working demo isn't proof of quality — *measured* quality is. A lot of engineers
ship LLM features with no idea whether they're any good. This project scores itself
on a hand-built **"golden set"** of questions with known-correct answers, and
compares the smart agent against a plain single-shot RAG **baseline**.

**Quick glossary of the metrics** (these are industry-standard for RAG; **RAGAS** is
the popular open-source framework that defines them):

- **Recall@5** — of the passages the answer *should* have found, how many showed up in
  the top 5? *(Did we retrieve the right evidence?)*
- **Faithfulness** — is every statement in the answer actually supported by the
  retrieved passages? *(The anti-hallucination score.)*
- **Context recall** — did retrieval gather all the facts needed to fully answer?
- **LLM-judge** — a separate strong model grades each answer 1–5 on a rubric, like a
  human reviewer would.
- **Fabricated citations** — made-up references. The target is, and stays, **zero**.

**The smart agent vs. the plain baseline** (6 vetted questions, mix of single-paper
and cross-paper):

| Metric | Plain baseline | **Smart agent** |
|---|---|---|
| Recall@5 | 0.75 | **1.00** |
| Faithfulness | 0.706 | **0.762** |
| Context recall | 0.610 | **0.656** |
| LLM-judge (normalized) | 0.958 | 0.958 |
| Fabricated citations | 0 | **0** *(enforced)* |

The headline: on **multi-paper comparison** questions, the agent reaches **perfect
recall (1.00)** — it always retrieves *both* papers — where the plain version keeps
leaving one side out. (Confirmed on a larger 24-question set too: 1.000 vs 0.958.)

**Cost & speed** — the LiteLLM routing + caching layer, before and after:

| Setup | $ per query | Median latency | Cache hits |
|---|---|---|---|
| One strong model, no cache | $0.0246 | 22.8s | 0% |
| **Routed** (cheap model for checks, strong for the answer) | **$0.0071 — 71% cheaper** | 24.3s | 0% |
| **Routed + warm cache** | **$0.0000 — free on repeats** | 16.7s | 100% |

**And it can't silently get worse.** Every pull request re-runs this evaluation in CI
(GitHub Actions) and **fails the build if quality drops** below committed thresholds.
Answer quality is treated like any other test — a versioned signal, not a vibe.

---

## Going deeper (for developers)

Want to run pieces individually, tweak the agent, or develop without Docker? Here's
the toolkit.

**Local setup without Docker:**

```bash
uv sync                          # install everything (or: pip install -e ".[dev,serve,ui]")
pre-commit install               # auto lint + format on commit (ruff)
python scripts/fetch_corpus.py   # download the 20 PDFs into data/raw/
docker compose up -d qdrant      # just the vector database
rag-ingest                       # parse → chunk → embed → index (safe to re-run)
```

**Command-line tools** — each is a standalone script for poking at one layer:

| Command | What it does | Handy flags |
|---|---|---|
| `python scripts/search.py "<query>"` | raw hybrid retrieval — see which chunks come back | `--k 8`, `--no-rerank`, `--compare` |
| `python scripts/ask.py "<question>"` | the **plain** single-shot RAG baseline | `--k 5` |
| `python scripts/agent_ask.py "<question>"` | the full **smart agent** in your terminal | `--k`, `--min-confidence 0.5`, `--max-retrieval-rounds 3`, `--max-revision-rounds 2`, `--trace` |
| `python scripts/eval_run.py` | score baseline vs agent on the golden set | `--status seed`, `--limit N`, `--no-ragas`, `--no-judge` |
| `python scripts/bench_routing_cache.py` | reproduce the cost/latency table above | `--strong gpt-4o`, `--cheap gpt-4o-mini`, `-n` |
| `python scripts/trace_coverage.py q-0001` | debug multi-hop: trace a question through each retrieval stage | `--expect`, `--k` |
| `python scripts/inspect_index.py` | sanity-check what's stored in Qdrant | `--query`, `--top` |

**Use the agent from Python:**

```python
from agentic_rag.agent import build_agent, run_agent

final = run_agent(build_agent(), "How does ELECTRA's objective differ from BERT and RoBERTa?")
print(final["guardrail"]["final_answer"])   # the cited answer, or a safe decline
```

**The serving API** — `POST /query` returns a structured `QueryResponse`: the `answer`,
a `confidence` score, a `grounded` flag, the `citations[]`, the agent's `steps[]` (the
full reasoning trace the UI panel shows), and `metering` (cost / latency / cache).
It's rate-limited and returns clean JSON errors — never a stack trace. The agent is
built once at startup and requests are serialized (the ML models aren't
thread-safe) — an honest single-instance posture.

**Tests & quality:**

```bash
pytest                                   # 143 offline unit tests — no network, no model downloads
ruff check . && ruff format --check .    # lint + format
```

**Repo layout:**

```
src/agentic_rag/
  ingest/        parse → chunk → embed → index (CLI: rag-ingest)
  retrieve/      hybrid retrieval (dense + BM25 + RRF + cross-encoder rerank)
  llm/           LiteLLM client — per-role routing, cost/latency metering, caching
  answer/        single-shot RAG baseline (kept as the comparison point)
  agent/         the LangGraph agent (state, nodes, routing, graph, corpus registry)
  guardrails/    prompt-injection defense (in) + abstain/confidence gate (out)
  observability/ optional Langfuse tracing (no-op when disabled)
  eval/          evaluation harness + CI-gate logic + index-fixture loader
  api/           FastAPI service (schemas, service, app)
scripts/         search · ask · agent_ask · eval_run · bench_routing_cache · trace_coverage · ...
ui/              streamlit_app.py — the demo front-end
eval/            golden_set · thresholds · fixtures · results   (committed)
```

---

### Read more

- **[`DECISIONS.md`](DECISIONS.md)** — the architecture decision log (ADRs): *why*
  each choice was made, with the trade-offs.
- **[`SOURCES.md`](SOURCES.md)** — the 20-paper corpus and where it came from.
- **[`eval/README.md`](eval/README.md)** — the evaluation methodology in detail.

<sub>Built by <a href="https://github.com/YuliaYur">YuliaYur</a> as a portfolio project — a production-grade agentic RAG system, not a notebook.</sub>
