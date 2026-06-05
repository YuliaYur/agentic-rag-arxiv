"""Endpoint tests for the FastAPI service (TestClient + a fake agent — no models)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from agentic_rag.api.app import create_app, limiter  # noqa: E402
from agentic_rag.api.schemas import (  # noqa: E402
    CitationOut,
    MeteringOut,
    QueryResponse,
    StepOut,
)
from agentic_rag.llm.client import LLMError  # noqa: E402


def _canned(question: str) -> QueryResponse:
    return QueryResponse(
        question=question,
        action="answer",
        answer="ELECTRA uses replaced-token detection [S1].",
        confidence=0.9,
        grounded=True,
        insufficient_context=False,
        citations=[
            CitationOut(
                source_id="S1",
                arxiv_id="2003.10555",
                title="ELECTRA",
                section="3",
                page=4,
                page_end=4,
                label="[S1] ELECTRA",
                url="https://arxiv.org/abs/2003.10555",
            )
        ],
        steps=[
            StepOut(node="retrieve", summary="round 1: 5 chunks", detail={"node": "retrieve"}),
            StepOut(
                node="output_guard", summary="guardrail → answer", detail={"node": "output_guard"}
            ),
        ],
        metering=MeteringOut(cost_usd=0.001, llm_calls=4, cache_hits=0, latency_ms=1200.0),
        retrieval_rounds=1,
        revision_rounds=0,
    )


class FakeService:
    """Stands in for AgentService — no agent, no models, no API."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc
        self.calls: list[str] = []

    def query(self, req):
        self.calls.append(req.question)
        if self._exc is not None:
            raise self._exc
        return _canned(req.question)


@pytest.fixture(autouse=True)
def _limiter_off():
    # The limiter is a module-global; keep it off for most tests (the rate-limit test
    # toggles it itself).
    limiter.enabled = False
    yield
    limiter.enabled = False


def test_query_happy_path():
    with TestClient(create_app(FakeService())) as c:
        r = c.post("/query", json={"question": "How does ELECTRA differ from BERT?"})
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "answer"
    assert body["answer"].startswith("ELECTRA")
    assert body["citations"][0]["arxiv_id"] == "2003.10555"
    assert any(s["node"] == "output_guard" for s in body["steps"])
    assert body["metering"]["llm_calls"] == 4


def test_empty_question_is_422():
    with TestClient(create_app(FakeService())) as c:
        r = c.post("/query", json={"question": ""})
    assert r.status_code == 422


def test_health_ready_with_service():
    with TestClient(create_app(FakeService())) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "agent_ready": True}


def test_not_ready_returns_503(monkeypatch):
    # No injected service + startup build fails -> agent not ready -> 503 on /query.
    def _boom():
        raise RuntimeError("no qdrant")

    monkeypatch.setattr("agentic_rag.api.app.build_service", _boom)
    with TestClient(create_app(None)) as c:
        assert c.get("/health").json()["agent_ready"] is False
        assert c.post("/query", json={"question": "x"}).status_code == 503


def test_agent_failure_returns_clean_502():
    with TestClient(create_app(FakeService(exc=LLMError("boom")))) as c:
        r = c.post("/query", json={"question": "x"})
    assert r.status_code == 502
    assert r.json()["error"] == "agent_failure"


def test_unexpected_error_is_clean_json_not_stack_trace():
    # e.g. an upstream provider quota error (not our LLMError) -> still clean JSON.
    client = TestClient(
        create_app(FakeService(exc=ValueError("quota exceeded"))), raise_server_exceptions=False
    )
    with client as c:
        r = c.post("/query", json={"question": "x"})
    assert r.status_code == 502
    assert r.json()["error"] == "agent_failure"
    assert "quota exceeded" in r.json()["detail"]


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setenv("API_RATE_LIMIT", "2/minute")
    # Disabled: every call passes (the limiter never blocks).
    limiter.enabled = False
    with TestClient(create_app(FakeService())) as c:
        off = [c.post("/query", json={"question": q}).status_code for q in "abcd"]
    assert off == [200, 200, 200, 200]
    # Enabled with a 2/min budget: it starts returning 429.
    limiter.enabled = True
    with TestClient(create_app(FakeService())) as c:
        on = [c.post("/query", json={"question": q}).status_code for q in "abcd"]
    assert 429 in on
