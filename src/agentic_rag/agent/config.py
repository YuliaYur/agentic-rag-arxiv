"""Agent graph configuration (loop caps + retrieval breadth)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    k: int = 5  # chunks retrieved per round
    max_retrieval_rounds: int = 3  # cap on the grade -> re-retrieve loop
    max_revision_rounds: int = 2  # cap on the cite_critic -> revise loop
