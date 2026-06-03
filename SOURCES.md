# Corpus sources

Curated corpus for the agentic RAG project: **20 papers tracing the transformer lineage**, from the original architecture through language-model scaling, efficient attention, retrieval, and the vision-transformer branch. Chosen as a *coherent lineage* so that evaluation questions can be comparative and multi-hop (e.g. "how does the pretraining objective of ELECTRA differ from BERT and RoBERTa?", "which papers address the quadratic cost of attention, and how?", "how does patch tokenization in ViT relate to token embeddings in the original Transformer?"). Single-shot RAG struggles with these — they're what justify the agent loop.

> **The PDFs are not committed to this repo.** Run `python scripts/fetch_corpus.py` to download them into `data/raw/` (which is git-ignored). This keeps the repo small, diffable, and clear of redistribution concerns. See **Licensing** below.

---

## Foundations

| # | Year | Paper | arXiv | PDF |
|---|------|-------|-------|-----|
| 1 | 2017 | Attention Is All You Need (Transformer) — Vaswani et al. | [1706.03762](https://arxiv.org/abs/1706.03762) | [pdf](https://arxiv.org/pdf/1706.03762) |
| 2 | 2018 | Deep Contextualized Word Representations (ELMo) — Peters et al. | [1802.05365](https://arxiv.org/abs/1802.05365) | [pdf](https://arxiv.org/pdf/1802.05365) |

## Encoder pretraining

| # | Year | Paper | arXiv | PDF |
|---|------|-------|-------|-----|
| 3 | 2018 | BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al. | [1810.04805](https://arxiv.org/abs/1810.04805) | [pdf](https://arxiv.org/pdf/1810.04805) |
| 4 | 2019 | RoBERTa: A Robustly Optimized BERT Pretraining Approach — Liu et al. | [1907.11692](https://arxiv.org/abs/1907.11692) | [pdf](https://arxiv.org/pdf/1907.11692) |
| 5 | 2019 | Exploring the Limits of Transfer Learning (T5) — Raffel et al. | [1910.10683](https://arxiv.org/abs/1910.10683) | [pdf](https://arxiv.org/pdf/1910.10683) |
| 6 | 2020 | ELECTRA: Pre-training Text Encoders as Discriminators — Clark et al. | [2003.10555](https://arxiv.org/abs/2003.10555) | [pdf](https://arxiv.org/pdf/2003.10555) |

## Decoder models & scaling

| # | Year | Paper | arXiv | PDF |
|---|------|-------|-------|-----|
| 7 | 2020 | Scaling Laws for Neural Language Models — Kaplan et al. | [2001.08361](https://arxiv.org/abs/2001.08361) | [pdf](https://arxiv.org/pdf/2001.08361) |
| 8 | 2020 | Language Models are Few-Shot Learners (GPT-3) — Brown et al. | [2005.14165](https://arxiv.org/abs/2005.14165) | [pdf](https://arxiv.org/pdf/2005.14165) |
| 9 | 2022 | Training Compute-Optimal LLMs (Chinchilla) — Hoffmann et al. | [2203.15556](https://arxiv.org/abs/2203.15556) | [pdf](https://arxiv.org/pdf/2203.15556) |

## Efficient & long-context attention

| # | Year | Paper | arXiv | PDF |
|---|------|-------|-------|-----|
| 10 | 2020 | Longformer: The Long-Document Transformer — Beltagy et al. | [2004.05150](https://arxiv.org/abs/2004.05150) | [pdf](https://arxiv.org/pdf/2004.05150) |
| 11 | 2022 | FlashAttention: Fast and Memory-Efficient Exact Attention — Dao et al. | [2205.14135](https://arxiv.org/abs/2205.14135) | [pdf](https://arxiv.org/pdf/2205.14135) |

## Adaptation

| # | Year | Paper | arXiv | PDF |
|---|------|-------|-------|-----|
| 12 | 2021 | LoRA: Low-Rank Adaptation of Large Language Models — Hu et al. | [2106.09685](https://arxiv.org/abs/2106.09685) | [pdf](https://arxiv.org/pdf/2106.09685) |

## Embeddings & retrieval (the bridge to RAG)

| # | Year | Paper | arXiv | PDF |
|---|------|-------|-------|-----|
| 13 | 2019 | Sentence-BERT: Sentence Embeddings using Siamese Networks — Reimers & Gurevych | [1908.10084](https://arxiv.org/abs/1908.10084) | [pdf](https://arxiv.org/pdf/1908.10084) |
| 14 | 2020 | Dense Passage Retrieval for Open-Domain QA (DPR) — Karpukhin et al. | [2004.04906](https://arxiv.org/abs/2004.04906) | [pdf](https://arxiv.org/pdf/2004.04906) |
| 15 | 2020 | Retrieval-Augmented Generation for Knowledge-Intensive NLP (RAG) — Lewis et al. | [2005.11401](https://arxiv.org/abs/2005.11401) | [pdf](https://arxiv.org/pdf/2005.11401) |

## Vision transformers

| # | Year | Paper | arXiv | PDF |
|---|------|-------|-------|-----|
| 16 | 2020 | An Image is Worth 16x16 Words (ViT) — Dosovitskiy et al. | [2010.11929](https://arxiv.org/abs/2010.11929) | [pdf](https://arxiv.org/pdf/2010.11929) |
| 17 | 2020 | Training Data-Efficient Image Transformers (DeiT) — Touvron et al. | [2012.12877](https://arxiv.org/abs/2012.12877) | [pdf](https://arxiv.org/pdf/2012.12877) |
| 18 | 2021 | Swin Transformer: Hierarchical ViT using Shifted Windows — Liu et al. | [2103.14030](https://arxiv.org/abs/2103.14030) | [pdf](https://arxiv.org/pdf/2103.14030) |
| 19 | 2021 | Learning Transferable Visual Models From Natural Language Supervision (CLIP) — Radford et al. | [2103.00020](https://arxiv.org/abs/2103.00020) | [pdf](https://arxiv.org/pdf/2103.00020) |
| 20 | 2021 | Masked Autoencoders Are Scalable Vision Learners (MAE) — He et al. | [2111.06377](https://arxiv.org/abs/2111.06377) | [pdf](https://arxiv.org/pdf/2111.06377) |

---

## Why these documents are good for RAG

- **Genuinely messy PDFs.** Two-column layouts, inline equations, tables, figure captions, and dense reference sections — exactly the structure that exercises a real parsing layer (LlamaParse / Unstructured / Marker) rather than a clean text file.
- **Multi-hop questions.** The lineage supports comparative queries that a single retrieval can't answer, giving the agent's grade/re-retrieve loop something real to do.
- **Domain you can judge.** As an ML engineer you can author a credible golden eval set and assess answer quality with authority — half the evaluation story.

## Licensing

arXiv papers carry a **per-paper license shown on each abstract page** (look for the license line near the submission history). Most of the canonical papers here use arXiv's **default non-exclusive distribution license**, which permits downloading and personal/research use but does **not** grant you redistribution rights. A minority may be CC-BY. Because licenses vary and several are not redistributable:

1. **We do not commit the PDFs.** `data/raw/` is git-ignored; `fetch_corpus.py` downloads each paper from its canonical arXiv URL at build time. Downloading for your own processing is fine; redistributing the files in a public repo is the part to avoid.
2. **Verify before redistributing anything derived.** If you ever want to ship extracted text, chunks, or quotes publicly, check the individual paper's license on its abstract page first.
3. **Be a polite client.** arXiv asks automated clients to throttle requests and identify themselves. The fetch script sleeps between downloads and sets a descriptive `User-Agent`. For bulk needs, arXiv also offers the [Kaggle arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) and an [S3 bulk-access program](https://info.arxiv.org/help/bulk_data_s3.html).

## What *is* committed

- This `SOURCES.md` (provenance + licensing).
- `scripts/fetch_corpus.py` (reproducible ingestion entry point).
- The golden evaluation dataset under `eval/` (small JSON — versioned, since it's the heart of the eval story).
- Optionally, 1–2 small unambiguously-open sample docs under `data/sample/` for an offline smoke test.

## Provenance

All 20 arXiv IDs and links in this file were verified against arxiv.org. If a `fetch` fails, check whether the paper has a newer version suffix (e.g. `1706.03762v7`) — the un-suffixed ID always resolves to the latest version.
