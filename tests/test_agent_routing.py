"""Tests for the agent graph routing + looping (mocked LLM + retriever, no API)."""

from __future__ import annotations

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.graph import build_graph, run_agent
from agentic_rag.agent.nodes import route_after_critic, route_after_grade
from agentic_rag.agent.state import CriticResult, GradeResult
from agentic_rag.answer.schemas import Citation, CitedAnswer
from agentic_rag.retrieve.models import RetrievedChunk

# --- fakes -------------------------------------------------------------------


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def retrieve(self, query, k):
        self.queries.append(query)
        return self.chunks


class FakeLLM:
    """Returns scripted responses per schema; repeats the last when exhausted."""

    def __init__(self, grades, answers, critics):
        self.q = {
            "GradeResult": list(grades),
            "CitedAnswer": list(answers),
            "CriticResult": list(critics),
        }
        self.calls: list[str] = []

    def structured(self, system, user, schema):
        name = schema.__name__
        self.calls.append(name)
        items = self.q[name]
        return items.pop(0) if len(items) > 1 else items[0]


def _chunks():
    return [
        RetrievedChunk(
            id="a",
            text="ELECTRA replaces tokens; trained over all tokens.",
            arxiv_id="2003.10555",
            title="ELECTRA",
            slug="electra",
            section="3 Method",
            page=4,
            page_end=4,
            chunk_index=0,
        ),
    ]


def _answer():
    return CitedAnswer(
        answer="ELECTRA uses replaced-token detection over all tokens [S1].",
        citations=[Citation(source_id="S1", arxiv_id="2003.10555", section="3 Method", page=4)],
        insufficient_context=False,
    )


def _grade(sufficient, refined="better query"):
    return GradeResult(sufficient=sufficient, reasoning="r", refined_query=refined)


def _critic(supported, score=1.0):
    return CriticResult(
        supported=supported,
        score=score,
        unsupported_claims=[] if supported else ["some claim"],
        feedback="" if supported else "fix it",
    )


def _run(grades, answers, critics, cfg):
    retriever = FakeRetriever(_chunks())
    llm = FakeLLM(grades, answers, critics)
    app = build_graph(retriever, llm, cfg)
    final = run_agent(app, "How does ELECTRA differ from BERT?", cfg)
    return final, retriever, llm


# --- pure routing ------------------------------------------------------------


def test_route_after_grade():
    assert (
        route_after_grade(
            {"grade": {"sufficient": True}, "retrieval_round": 1, "max_retrieval_rounds": 3}
        )
        == "generate"
    )
    assert (
        route_after_grade(
            {"grade": {"sufficient": False}, "retrieval_round": 1, "max_retrieval_rounds": 3}
        )
        == "retrieve"
    )
    # cap reached -> stop looping, go generate even though insufficient
    assert (
        route_after_grade(
            {"grade": {"sufficient": False}, "retrieval_round": 3, "max_retrieval_rounds": 3}
        )
        == "generate"
    )


def test_route_after_critic():
    assert (
        route_after_critic(
            {"critic": {"supported": True}, "revision_round": 0, "max_revision_rounds": 2}
        )
        == "end"
    )
    assert (
        route_after_critic(
            {"critic": {"supported": False}, "revision_round": 0, "max_revision_rounds": 2}
        )
        == "generate"
    )
    assert (
        route_after_critic(
            {"critic": {"supported": False}, "revision_round": 2, "max_revision_rounds": 2}
        )
        == "end"
    )


# --- full graph --------------------------------------------------------------


def test_happy_path_no_loops():
    final, retriever, llm = _run([_grade(True)], [_answer()], [_critic(True)], AgentConfig())
    assert final["retrieval_round"] == 1
    assert final["revision_round"] == 0
    assert [e["node"] for e in final["trace"]] == [
        "retrieve",
        "grade_context",
        "generate",
        "cite_critic",
    ]
    assert final["answer"].is_grounded
    assert len(retriever.queries) == 1


def test_retrieval_loop_then_sufficient():
    final, retriever, _ = _run(
        [_grade(False, refined="ELECTRA RTD vs BERT MLM"), _grade(True)],
        [_answer()],
        [_critic(True)],
        AgentConfig(),
    )
    assert final["retrieval_round"] == 2
    # second retrieval used the reformulated query
    assert retriever.queries == ["How does ELECTRA differ from BERT?", "ELECTRA RTD vs BERT MLM"]


def test_retrieval_round_cap():
    cfg = AgentConfig(max_retrieval_rounds=2)
    final, retriever, _ = _run([_grade(False)], [_answer()], [_critic(True)], cfg)
    assert final["retrieval_round"] == 2  # capped, not infinite
    assert len(retriever.queries) == 2
    assert final["answer"] is not None  # still produced an answer


def test_revision_loop_then_supported():
    final, _, llm = _run(
        [_grade(True)], [_answer(), _answer()], [_critic(False), _critic(True)], AgentConfig()
    )
    assert final["revision_round"] == 1
    assert llm.calls.count("CitedAnswer") == 2  # one revision happened
    assert final["critic"]["supported"] is True


def test_revision_round_cap():
    cfg = AgentConfig(max_revision_rounds=1)
    final, _, llm = _run([_grade(True)], [_answer()], [_critic(False)], cfg)
    assert final["revision_round"] == 1  # capped
    # initial generate + 1 revision = 2 generate calls, then stop
    assert llm.calls.count("CitedAnswer") == 2
