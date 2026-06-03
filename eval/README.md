# Evaluation

The eval story has two halves (per the project plan): a **golden set** authored
by hand, and **automated metrics** (RAGAS + an LLM-judge) run over it and gated
in CI.

> This directory is **committed** (unlike `data/`): the golden set is small JSON
> and is the heart of the evaluation story, so it's versioned.

## Files

- `golden_set.jsonl` — hand-authored questions with the source paper(s) that
  should be retrieved/cited. One JSON object per line. Deliberately includes the
  **comparative / multi-hop** questions that single-shot RAG struggles with —
  these justify the agent's grade/re-retrieve loop.

## Schema (one line per question)

```json
{
  "id": "q-0001",
  "question": "How does ELECTRA's pre-training objective differ from BERT's MLM?",
  "type": "comparative",
  "expected_arxiv_ids": ["2003.10555", "1810.04805"],
  "notes": "Should contrast replaced-token detection vs masked language modeling."
}
```

`type` ∈ `factual` | `comparative` | `multi-hop`.

## Planned metrics (not yet wired)

- **Retrieval:** recall@k / MRR of `expected_arxiv_ids` against the hybrid
  retriever's hits.
- **Answer quality (RAGAS):** faithfulness, answer relevancy, context precision/recall.
- **LLM-judge:** a rubric-scored judgment of the final cited answer.
- **CI gate:** fail the build if metrics drop below thresholds.

## Current status

Seed golden set only. Metric runners + CI gating come with the agent/answer
layer. See the project [`README`](../README.md) for overall status.
