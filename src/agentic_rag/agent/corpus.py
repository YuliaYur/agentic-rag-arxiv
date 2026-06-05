"""Corpus paper-name registry: common name(s) -> arXiv id.

The agent uses this for two deterministic, corpus-aware steps (ADR-0014):

  * detection — which papers does a question explicitly NAME? (so we can tell when
    a comparison's named paper is missing from retrieval and force a decomposed
    re-retrieval, instead of trusting the LLM grader, which judges these
    borderline multi-hop cases "sufficient" inconsistently).
  * anchoring — when a sub-query names a paper whose model name isn't in its title
    (the original Transformer is "Attention Is All You Need"; ViT is "An Image is
    Worth 16x16 Words"), the alias lets retrieval lock onto the right paper.

Names are matched as token phrases: every (non-stopword) token of the name must
appear in the text. Multi-word names ("masked autoencoder", "dense passage") are
deliberately specific so a topic word alone (e.g. "masked" in "masked language
modeling") does NOT false-match the model (MAE). Provenance: SOURCES.md.
"""

from __future__ import annotations

# (name phrase, arxiv_id). Multiple names may map to the same paper.
CORPUS_PAPER_NAMES: tuple[tuple[str, str], ...] = (
    ("original transformer", "1706.03762"),
    ("attention is all you need", "1706.03762"),
    ("vaswani", "1706.03762"),
    ("bert", "1810.04805"),
    ("roberta", "1907.11692"),
    ("electra", "2003.10555"),
    ("elmo", "1802.05365"),
    ("t5", "1910.10683"),
    ("text-to-text", "1910.10683"),
    ("gpt-3", "2005.14165"),
    ("gpt3", "2005.14165"),
    ("kaplan", "2001.08361"),
    ("scaling laws", "2001.08361"),
    ("chinchilla", "2203.15556"),
    ("longformer", "2004.05150"),
    ("flashattention", "2205.14135"),
    ("flash attention", "2205.14135"),
    ("lora", "2106.09685"),
    ("sentence-bert", "1908.10084"),
    ("sbert", "1908.10084"),
    ("dpr", "2004.04906"),
    ("dense passage", "2004.04906"),
    ("rag", "2005.11401"),
    ("vit", "2010.11929"),
    ("deit", "2012.12877"),
    ("swin", "2103.14030"),
    ("clip", "2103.00020"),
    ("mae", "2111.06377"),
    ("masked autoencoder", "2111.06377"),
)
