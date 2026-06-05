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

    def structured(self, system, user, schema, role=None):
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


def _grade(sufficient, refined="better query", sub_queries=None):
    return GradeResult(
        sufficient=sufficient,
        reasoning="r",
        refined_query=refined,
        sub_queries=sub_queries or [],
    )


def _critic(supported, score=None):
    # Unsupported drafts default to a low score (below accept_score) so they trigger
    # a revision; supported drafts score 1.0.
    if score is None:
        score = 1.0 if supported else 0.4
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
        "output_guard",
    ]
    assert final["answer"].is_grounded
    assert final["guardrail"]["action"] == "answer"  # clean answer passes the output guard
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


def test_decomposed_reretrieval_covers_both_sides():
    # Round 1 (single query) surfaces only ELECTRA; the grader judges it
    # insufficient and emits per-side sub-queries. Round 2 must fan out over those
    # sub-queries and the merged context must now cover BOTH papers.
    electra = RetrievedChunk(
        id="e",
        text="ELECTRA uses replaced-token detection.",
        arxiv_id="2003.10555",
        title="ELECTRA",
        slug="electra",
        section="3",
        page=4,
        page_end=4,
        chunk_index=0,
    )
    bert = RetrievedChunk(
        id="b",
        text="BERT uses masked language modeling.",
        arxiv_id="1810.04805",
        title="BERT",
        slug="bert",
        section="3",
        page=4,
        page_end=4,
        chunk_index=0,
    )

    class SidedRetriever:
        """Returns ELECTRA chunks unless the query mentions BERT (per-side fake)."""

        def __init__(self):
            self.queries = []

        def retrieve(self, query, k):
            self.queries.append(query)
            return [bert] if "BERT" in query else [electra]

    retriever = SidedRetriever()
    llm = FakeLLM(
        [_grade(False, sub_queries=["ELECTRA objective", "BERT objective"]), _grade(True)],
        [_answer()],
        [_critic(True)],
    )
    app = build_graph(retriever, llm, AgentConfig())
    final = run_agent(app, "How does ELECTRA differ from BERT?", AgentConfig())

    # Round 1 used the single original question; round 2 fanned out per side.
    assert retriever.queries[0] == "How does ELECTRA differ from BERT?"
    assert retriever.queries[1:] == ["ELECTRA objective", "BERT objective"]
    # The merged round-2 context covers both papers (the multi-hop coverage fix).
    papers = {c.arxiv_id for c in final["chunks"]}
    assert papers == {"2003.10555", "1810.04805"}
    # And the trace records the decomposition.
    retrieve_entries = [e for e in final["trace"] if e["node"] == "retrieve"]
    assert retrieve_entries[-1]["decomposed"] is True


def test_deterministic_gate_forces_decomposition_when_llm_says_sufficient():
    # The crux of ADR-0014: even when the LLM grader (wrongly) calls round-1
    # context sufficient, the deterministic named-paper check must notice BERT is
    # missing, force a decomposed re-retrieval, and lock coverage once both the
    # named papers are present.
    electra = RetrievedChunk(
        id="e",
        text="ELECTRA replaced-token detection.",
        arxiv_id="2003.10555",
        title="ELECTRA",
        slug="electra",
        section="3",
        page=4,
        page_end=4,
        chunk_index=0,
    )
    bert = RetrievedChunk(
        id="b",
        text="BERT masked language modeling.",
        arxiv_id="1810.04805",
        title="BERT",
        slug="bert",
        section="3",
        page=4,
        page_end=4,
        chunk_index=0,
    )

    class TitledRetriever:
        """Has a corpus title map; returns BERT only for the BERT-title sub-query."""

        _titles = {
            "2003.10555": "ELECTRA: Pre-training Text Encoders as Discriminators",
            "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers",
        }

        def __init__(self):
            self.queries = []

        @property
        def titles(self):
            return self._titles

        def retrieve(self, query, k):
            self.queries.append(query)
            return [bert] if "bidirectional" in query.lower() else [electra]

    retriever = TitledRetriever()
    # LLM always says "sufficient" — the deterministic gate must override it.
    llm = FakeLLM([_grade(True)], [_answer()], [_critic(True)])
    app = build_graph(retriever, llm, AgentConfig())
    final = run_agent(app, "How does ELECTRA differ from BERT?", AgentConfig())

    assert final["retrieval_round"] == 2  # gate forced a second round despite "sufficient"
    assert any(e["node"] == "retrieve" and e["decomposed"] for e in final["trace"])
    assert {c.arxiv_id for c in final["chunks"]} == {"2003.10555", "1810.04805"}


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


def test_accept_score_stops_revising_early():
    # critic not "supported" but score clears accept_score (0.8) -> stop, no revision.
    final, _, llm = _run([_grade(True)], [_answer()], [_critic(False, score=0.85)], AgentConfig())
    assert final["revision_round"] == 0
    assert llm.calls.count("CitedAnswer") == 1  # no churn
    assert final["guardrail"]["action"] == "answer"


def test_keep_best_returns_strongest_draft_not_last():
    cite = [Citation(source_id="S1", arxiv_id="2003.10555", section="3 Method", page=4)]
    strong = CitedAnswer(
        answer="Strong first answer [S1].", citations=cite, insufficient_context=False
    )
    weak = CitedAnswer(
        answer="Weaker revised answer [S1].", citations=cite, insufficient_context=False
    )
    # draft0 scores 0.6 (revise), revision draft1 scores worse (0.3) -> keep draft0.
    final, _, _ = _run(
        [_grade(True)], [strong, weak], [_critic(False, 0.6), _critic(False, 0.3)], AgentConfig()
    )
    assert final["answer"].answer == "Strong first answer [S1]."  # best kept, not the last
    assert final["best_quality"] == 0.6
