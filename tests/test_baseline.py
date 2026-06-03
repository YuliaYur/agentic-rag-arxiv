"""Tests for the single-shot baseline orchestration (fakes; no API calls)."""

from __future__ import annotations

from agentic_rag.answer.baseline import SingleShotRAG
from agentic_rag.answer.schemas import Citation, CitedAnswer
from agentic_rag.retrieve.models import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def retrieve(self, query, k):
        self.calls.append((query, k))
        return self._chunks


class FakeLLM:
    def __init__(self, response):
        self._response = response
        self.called = False

    def structured(self, system, user, schema):
        self.called = True
        self.system, self.user = system, user
        return self._response


def _chunks():
    return [
        RetrievedChunk(
            id="a",
            text="ELECTRA replaces tokens.",
            arxiv_id="2003.10555",
            title="ELECTRA",
            slug="electra",
            section="3 Method",
            page=4,
            page_end=4,
            chunk_index=0,
        ),
    ]


def test_baseline_returns_grounded_answer_and_calls_llm():
    llm = FakeLLM(
        CitedAnswer(
            answer="ELECTRA uses replaced-token detection [S1].",
            citations=[Citation(source_id="S1", arxiv_id="2003.10555", section="3 Method", page=4)],
            insufficient_context=False,
        )
    )
    retriever = FakeRetriever(_chunks())
    rag = SingleShotRAG(retriever, llm, k=3)

    res = rag.answer("How does ELECTRA train?")

    assert llm.called
    assert retriever.calls == [("How does ELECTRA train?", 3)]
    assert res.is_grounded
    assert res.citations[0].arxiv_id == "2003.10555"
    # the retrieved chunk's text reached the prompt
    assert "ELECTRA replaces tokens." in llm.user


def test_baseline_abstains_without_calling_llm_when_no_chunks():
    llm = FakeLLM(None)
    rag = SingleShotRAG(FakeRetriever([]), llm)

    res = rag.answer("anything")

    assert res.insufficient_context
    assert not res.citations
    assert not llm.called  # crucial: no wasted API spend when nothing was retrieved
