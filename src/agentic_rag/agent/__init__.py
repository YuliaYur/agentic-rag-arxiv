"""Agentic answer graph (LangGraph): retrieve -> grade -> generate -> cite-critic.

Two capped loops give it what single-shot RAG lacks: it can *re-retrieve* with a
reformulated query when context is weak, and *revise* when the answer isn't fully
supported by its citations.
"""

from .graph import build_agent, build_graph, run_agent
from .state import AgentState, CriticResult, GradeResult

__all__ = ["build_agent", "build_graph", "run_agent", "AgentState", "GradeResult", "CriticResult"]
