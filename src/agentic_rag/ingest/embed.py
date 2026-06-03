"""Local, free embeddings via sentence-transformers.

Model: BAAI/bge-small-en-v1.5 (see DECISIONS.md).
* 384-dim, 512-token context, ~33M params -> runs fast on CPU.
* Strong MTEB retrieval scores for its size; a good default for paper QA.

The model's own tokenizer is exposed as ``token_counter`` so chunking packs to
the real context window rather than a guess.
"""

from __future__ import annotations

from .config import EmbedConfig


class Embedder:
    def __init__(self, config: EmbedConfig | None = None) -> None:
        # Imported lazily so that importing the package (e.g. for tests) does not
        # require torch / sentence-transformers to be installed.
        from sentence_transformers import SentenceTransformer

        self.config = config or EmbedConfig()
        self.model = SentenceTransformer(self.config.model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def token_counter(self):
        """A token counter backed by the model tokenizer (no special tokens)."""
        tok = self.model.tokenizer
        return lambda text: len(tok.encode(text, add_special_tokens=False))

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()
