# Evaluation

The eval story has two halves: a **golden set** (hand-curated questions with
reference answers) and an **automated metric suite** (retrieval metrics +
RAGAS-style answer/context metrics + an LLM-judge) run over **both** systems —
the single-shot baseline (ADR-0007) and the agent graph (ADR-0008) — to produce a
comparison table. This directory is **committed** (unlike `data/`): the golden set
and results are the heart of the evaluation story, so they're versioned.

> ⚠️ **The reference answers in `golden_set.jsonl` are DRAFTS.** The 24 questions
> marked `"status": "draft"` were machine-generated and **need expert review**
> (YuliaYur). Only the 6 `"status": "seed"` questions are vetted. Curate the
> drafts — fix any wrong reference answer or `expected_arxiv_ids`, flip `status`
> to `"reviewed"` — before trusting metrics computed over them.

## Files

```
golden_set.jsonl     # the questions (one JSON object per line)
results/             # saved runs: <timestamp>.json + .md, plus latest.*
README.md            # this file
```

## Golden-set schema (one line per question)

```json
{
  "id": "q-0001",
  "question": "How does ELECTRA's pre-training objective differ from BERT's MLM?",
  "type": "comparative",
  "expected_arxiv_ids": ["2003.10555", "1810.04805"],
  "reference_answer": "ELECTRA replaces MLM with replaced-token detection ...",
  "notes": "ELECTRA = replaced-token detection; BERT = masked subset.",
  "status": "seed"
}
```

- `type` ∈ `factual` (single-hop, one paper) | `comparative` | `multi-hop`
  (cross-paper — the questions that justify the agent loop).
- `expected_arxiv_ids` — the paper(s) retrieval should surface (drives recall/MRR).
- `reference_answer` — ground truth; feeds context-recall and the judge.
- `status` — `seed` (vetted) | `draft` (needs curation) | `reviewed` (you curated it).

## Running

```bash
python scripts/eval_run.py --status seed     # the 6 curated questions (cheap)
python scripts/eval_run.py --limit 6         # first 6 of whatever's in the file
python scripts/eval_run.py                   # full set (after you curate it)
python scripts/eval_run.py --no-ragas --no-judge   # retrieval metrics only (free-ish)
```

Needs Qdrant + the index + `OPENAI_API_KEY`. It makes **many** paid LLM calls
(each system answers, then ~7 metric/judge calls per question per system), so use
`--status`/`--limit` while iterating. Results are written to `eval/results/`.

## The metrics, in plain language

**Retrieval** (no LLM — scores the retriever against `expected_arxiv_ids`):

- **Recall@k** — *of the papers this question needs, how many came back in the top
  k?* 1.0 = all of them. A multi-hop question expecting 2 papers that retrieves
  only 1 scores 0.5. This is the ceiling on everything else: if the right paper
  never arrives, no amount of clever generation can be correct.
- **MRR** (mean reciprocal rank) — *how high was the first relevant paper?* Rank 1
  → 1.0, rank 2 → 0.5. Rewards putting good sources at the top.

**Answer & context quality** (RAGAS-style, LLM-scored — implemented natively, see
[ADR-0011](../DECISIONS.md); each is a fraction in [0,1]):

- **Faithfulness** — *of what the answer claims, how much is actually supported by
  the retrieved context?* Decompose the answer into atomic claims, check each
  against the context. **Low = hallucination** (claims not in the sources).
- **Answer relevancy** — *does the answer actually address the question?* Generate
  the questions the answer would answer and measure closeness to the real one.
  **Low = evasive or off-topic.**
- **Context precision** — *are the retrieved chunks relevant, and ranked well?*
  Rank-weighted, so relevant chunks near the top score higher. **Low = noisy
  retrieval** (junk chunks crowding the context).
- **Context recall** — *did retrieval bring back everything the reference answer
  needs?* Fraction of the reference's claims that are supported by the context.
  **Low = missing evidence** — exactly the gap the agent's re-retrieve loop targets.

**Overall** (LLM-judge):

- **Judge** — a holistic 1–5 score against an explicit [rubric](../src/agentic_rag/eval/judge.py)
  comparing the answer to the reference (correctness, completeness, relevance),
  normalized to [0,1]. This is the closest single number to "is this a good
  answer?".

## How to read the scores — what "good" looks like

Treat these as **relative** signals (baseline vs agent, run over run) more than
absolute truth — the LLM-scored metrics have noise. Rough bands:

| Metric | weak | ok | good |
|---|---|---|---|
| Recall@k | < 0.6 | 0.6–0.8 | > 0.8 |
| MRR | < 0.5 | 0.5–0.7 | > 0.7 |
| Faithfulness | < 0.7 | 0.7–0.9 | > 0.9 |
| Answer relevancy | < 0.6 | 0.6–0.8 | > 0.8 |
| Context precision | < 0.5 | 0.5–0.75 | > 0.75 |
| Context recall | < 0.6 | 0.6–0.8 | > 0.8 |
| Judge (norm) | < 0.5 | 0.5–0.75 | > 0.75 |

**Diagnosing with them together:**
- **Low recall/context-recall but high faithfulness** → the retriever is the
  bottleneck; the generator is honestly working with too little. Fix retrieval
  (or lean on the agent's re-retrieve loop).
- **High recall but low faithfulness** → the right context was there but the
  answer drifted/hallucinated. A generation/prompt problem.
- **Low context precision** → too much junk in the context; tighten `k` or rerank.
- **Faithfulness ≫ judge** → answer is well-grounded but incomplete or off-target.

A subtle one this suite already surfaced (see `results/`): on a question where
**retrieval missed the source paper**, the baseline *hallucinated a confident wrong
answer* (judge 2/5) while the agent *correctly abstained* (judge 1/5, because the
rubric penalizes refusing when the reference shows the answer was attainable). The
"safer" behavior scored worse — a reminder to read per-question results, not just
the aggregate, and a candidate rubric refinement (should faithful abstention on a
retrieval miss really score 1?).

## Why the agent doesn't obviously win (yet)

The agent's edge is supposed to show on **multi-hop** questions where one retrieval
misses a paper. If recall is tied (both systems miss the same second paper), the
agent's grade/re-retrieve loop can't help, and its extra steps only add latency,
cost, and chances for the critic to force an over-cautious abstention. The eval is
doing its job by making this visible and measurable rather than assumed — curate
the multi-hop questions and expand the set to test the hypothesis properly.

## Planned next

- **CI gate:** fail the build if key metrics regress below thresholds (run on the
  `reviewed` subset to keep CI cheap and deterministic-ish).
- Expand to the full curated ~30+ set once the drafts are reviewed.
