"""Semantic LLM cache (LiteLLM over Redis).

**Off by default.** When enabled, ``configure_cache`` installs a process-global
LiteLLM ``redis-semantic`` cache: each call's messages are embedded and matched to
prior calls by cosine similarity; a hit above the threshold returns the stored
completion with **no provider call** (so ~0 cost and ~0 latency). Backed by the
``redis-stack-server`` (RediSearch) service in docker-compose.

Trade-offs (why it's opt-in + conservative) live in ADR-0015: a cached answer can
go **stale** if the corpus/index changes, and too low a similarity threshold can
return a **wrong** answer for a merely-similar question. Mitigations: a strict
default threshold, a TTL, and bypassing the cache when freshness matters.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    redis_url: str | None = None  # overrides host/port; e.g. redis://:pass@host:6379
    similarity_threshold: float = 0.95  # cosine; higher = stricter (fewer false hits)
    ttl: int = 3600  # seconds a cached answer stays valid
    embedding_model: str = "text-embedding-3-small"

    def url(self) -> str:
        return self.redis_url or f"redis://{self.host}:{self.port}"

    @classmethod
    def from_env(cls) -> CacheConfig:
        return cls(
            enabled=_truthy(os.getenv("LLM_CACHE_ENABLED", "false")),
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            redis_url=os.getenv("REDIS_URL") or None,
            similarity_threshold=float(os.getenv("LLM_CACHE_SIMILARITY_THRESHOLD", "0.95")),
            ttl=int(os.getenv("LLM_CACHE_TTL", "3600")),
            embedding_model=os.getenv("LLM_CACHE_EMBEDDING_MODEL", "text-embedding-3-small"),
        )


def configure_cache(config: CacheConfig | None = None) -> bool:
    """Install (or clear) the global LiteLLM semantic cache. Returns whether it's on.

    Fail-safe: any setup error (Redis down, missing dep) disables caching rather
    than breaking the app — calls just go to the provider as usual.
    """
    cfg = config or CacheConfig.from_env()
    import litellm

    if not cfg.enabled:
        litellm.cache = None
        return False
    try:
        litellm.cache = litellm.Cache(
            type="redis-semantic",
            redis_url=cfg.url(),
            ttl=cfg.ttl,
            similarity_threshold=cfg.similarity_threshold,
            redis_semantic_cache_embedding_model=cfg.embedding_model,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — caching must never break the app
        print(f"[cache] semantic cache disabled ({exc})", file=sys.stderr)
        litellm.cache = None
        return False


def cache_enabled() -> bool:
    """True if a global LiteLLM cache is currently installed."""
    import litellm

    return getattr(litellm, "cache", None) is not None
