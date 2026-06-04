"""Observability tests: config toggling, NoOp safety, and that a graph run is
traced per node (via an injected fake tracer). All offline — no Langfuse, no network."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.graph import build_graph, run_agent
from agentic_rag.agent.state import CriticResult, GradeResult
from agentic_rag.answer.schemas import Citation, CitedAnswer
from agentic_rag.observability import NoOpTracer, build_tracer, configure_tracer
from agentic_rag.observability.config import TracingConfig
from agentic_rag.retrieve.models import RetrievedChunk

# --- config ------------------------------------------------------------------


def test_config_disabled_by_default(monkeypatch):
    for var in ("LANGFUSE_TRACING", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = TracingConfig.from_env()
    assert cfg.enabled is False
    assert cfg.has_keys is False


def test_config_enabled_with_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    cfg = TracingConfig.from_env()
    assert cfg.enabled and cfg.has_keys


def test_build_tracer_noop_when_disabled():
    assert isinstance(build_tracer(TracingConfig(enabled=False)), NoOpTracer)


def test_build_tracer_noop_when_enabled_but_no_keys():
    # enabled but missing keys -> safely disabled (no crash)
    assert isinstance(build_tracer(TracingConfig(enabled=True)), NoOpTracer)


def test_noop_tracer_is_safe():
    t = NoOpTracer()
    with t.trace("run", input="q") as root:
        root.update(output={"a": 1}, metadata={"m": 2})
        with t.span("node", input={}) as span:
            span.update(output={}, metadata={})
            t.generation(name="llm", model="gpt-4o-mini", usage={"input": 1, "output": 2})
    assert t.last_trace_url is None
    t.flush()


# --- fake tracer that records the nesting ------------------------------------


class _RecordingSpan:
    def __init__(self, name):
        self.name = name
        self.output = None
        self.metadata = None

    def update(self, *, output=None, metadata=None):
        if output is not None:
            self.output = output
        if metadata is not None:
            self.metadata = metadata


class FakeTracer:
    enabled = True
    last_trace_url = "http://localhost:3000/trace/fake"

    def __init__(self):
        self.root = None
        self.spans = []
        self.generations = []

    @contextmanager
    def trace(self, name, *, input=None, metadata=None):
        self.root = _RecordingSpan(name)
        yield self.root

    @contextmanager
    def span(self, name, *, input=None):
        s = _RecordingSpan(name)
        self.spans.append(s)
        yield s

    def generation(self, name, *, model, input=None, output=None, usage=None):
        self.generations.append({"name": name, "model": model, "usage": usage})

    def flush(self):
        pass


# --- a graph run, fully traced ----------------------------------------------


class FakeRetriever:
    def retrieve(self, query, k):
        return [
            RetrievedChunk(
                id="a",
                text="ELECTRA detects replaced tokens.",
                arxiv_id="2003.10555",
                title="ELECTRA",
                slug="electra",
                section="3 Method",
                page=4,
                page_end=4,
                chunk_index=0,
            )
        ]


class FakeLLM:
    def structured(self, system, user, schema):
        name = schema.__name__
        if name == "GradeResult":
            return GradeResult(sufficient=True, reasoning="r", refined_query="q")
        if name == "CitedAnswer":
            return CitedAnswer(
                answer="ELECTRA detects replaced tokens [S1].",
                citations=[
                    Citation(source_id="S1", arxiv_id="2003.10555", section="3 Method", page=4)
                ],
                insufficient_context=False,
            )
        return CriticResult(supported=True, score=1.0, unsupported_claims=[], feedback="")


def test_graph_run_emits_one_span_per_node():
    fake = FakeTracer()
    configure_tracer(fake)
    try:
        app = build_graph(FakeRetriever(), FakeLLM(), AgentConfig())
        run_agent(app, "How does ELECTRA work?", AgentConfig())
    finally:
        configure_tracer(None)  # reset so other tests get the env-based (disabled) tracer

    assert fake.root is not None
    assert [s.name for s in fake.spans] == [
        "retrieve",
        "grade_context",
        "generate",
        "cite_critic",
        "output_guard",
    ]
    # each span carries that node's structured metadata
    retrieve_span = fake.spans[0]
    assert retrieve_span.metadata["node"] == "retrieve"
    guard_span = fake.spans[-1]
    assert guard_span.metadata["action"] == "answer"
    # the root trace got the final outcome
    assert fake.root.output["action"] == "answer"


def test_tracing_disabled_run_is_identical():
    # With the default (disabled) tracer the graph still runs and produces an answer.
    configure_tracer(NoOpTracer())
    try:
        app = build_graph(FakeRetriever(), FakeLLM(), AgentConfig())
        final = run_agent(app, "How does ELECTRA work?", AgentConfig())
    finally:
        configure_tracer(None)
    assert final["guardrail"]["action"] == "answer"


@pytest.fixture(autouse=True)
def _reset_tracer():
    yield
    configure_tracer(None)
