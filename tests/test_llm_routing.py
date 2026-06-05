"""Tests for per-role model routing (config resolution + node wiring, no API)."""

from __future__ import annotations

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.graph import build_graph, run_agent
from agentic_rag.agent.state import CriticResult, GradeResult
from agentic_rag.answer.schemas import Citation, CitedAnswer
from agentic_rag.llm.config import LLMConfig
from agentic_rag.retrieve.models import RetrievedChunk

# --- config resolution -------------------------------------------------------


def test_resolve_falls_back_to_default():
    c = LLMConfig()  # no overrides
    assert c.resolve(None) == "gpt-4o-mini"
    assert c.resolve("synthesis") == "gpt-4o-mini"


def test_routed_preset_sends_synthesis_to_strong_model():
    c = LLMConfig.routed(synthesis="gpt-4o", cheap="gpt-4o-mini")
    assert c.resolve("synthesis") == "gpt-4o"
    assert c.resolve("grade") == "gpt-4o-mini"
    assert c.resolve("critic") == "gpt-4o-mini"
    assert c.resolve(None) == "gpt-4o-mini"


def test_uniform_preset_is_one_model_for_all():
    c = LLMConfig.uniform("gpt-4o")
    assert c.resolve("grade") == "gpt-4o"
    assert c.resolve("synthesis") == "gpt-4o"


def test_from_env_reads_per_role_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_MODEL_SYNTHESIS", "gpt-4o")
    monkeypatch.delenv("LLM_MODEL_GRADE", raising=False)
    c = LLMConfig.from_env()
    assert c.resolve("synthesis") == "gpt-4o"
    assert c.resolve("grade") == "gpt-4o-mini"


# --- the agent nodes pass the right role to the client -----------------------


class _RoleCapturingLLM:
    """Records the role each schema is requested with; returns canned answers."""

    def __init__(self):
        self.roles: dict[str, str | None] = {}

    def structured(self, system, user, schema, role=None):
        self.roles[schema.__name__] = role
        name = schema.__name__
        if name == "GradeResult":
            return GradeResult(sufficient=True, reasoning="r", refined_query="q")
        if name == "CitedAnswer":
            return CitedAnswer(
                answer="ELECTRA uses replaced-token detection [S1].",
                citations=[Citation(source_id="S1", arxiv_id="2003.10555", section="3", page=4)],
                insufficient_context=False,
            )
        return CriticResult(supported=True, score=1.0, unsupported_claims=[], feedback="")


class _FakeRetriever:
    titles: dict = {}  # empty -> the deterministic named-paper gate stays off

    def retrieve(self, query, k):
        return [
            RetrievedChunk(
                id="a",
                text="ELECTRA replaces tokens.",
                arxiv_id="2003.10555",
                title="ELECTRA",
                slug="electra",
                section="3",
                page=4,
                page_end=4,
                chunk_index=0,
            )
        ]


def test_nodes_route_by_role():
    llm = _RoleCapturingLLM()
    app = build_graph(_FakeRetriever(), llm, AgentConfig())
    run_agent(app, "How does ELECTRA differ from BERT?", AgentConfig())
    assert llm.roles["GradeResult"] == "grade"
    assert llm.roles["CitedAnswer"] == "synthesis"
    assert llm.roles["CriticResult"] == "critic"
