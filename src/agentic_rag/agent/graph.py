"""Assemble the LangGraph StateGraph and helpers to run it.

START -> retrieve -> grade_context -.
            ^                        |-(sufficient | cap)-> generate -> cite_critic -.
            '----(weak & rounds left)'                          ^                    |
                                                                '-(unsupported &     |
                                                                   revisions left)---'
                            (supported | cap) -> output_guard -> END

The retrieve node runs the input guardrail (injection scan) on its chunks; the
terminal output_guard node runs the output guardrail (structure/abstain/confidence).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_rag.guardrails import Guardrails
from agentic_rag.observability import get_tracer

from .config import AgentConfig
from .nodes import AgentNodes, route_after_critic, route_after_grade
from .state import AgentState


def _span_input(name: str, state: dict) -> dict:
    """Small, JSON-safe snapshot of what a node is working on (no raw chunks)."""
    if name == "retrieve":
        return {"query": state.get("question"), "k": state.get("k")}
    if name == "output_guard":
        return {}
    return {
        "question": state.get("original_question"),
        "revision_round": state.get("revision_round", 0),
    }


def _traced(name: str, fn):
    """Wrap a node so each call is one Langfuse span carrying that node's metadata.

    The node's own ``trace`` entry (retrieval_round / grade / critic_score / …) is
    reused verbatim as the span's output+metadata — instrumentation lives here, so
    ``nodes.py`` stays free of any tracing code. LLM generations created inside the
    node attach under this span via the tracer's parent stack.
    """

    def wrapped(state: dict) -> dict:
        with get_tracer().span(name, input=_span_input(name, state)) as span:
            result = fn(state)
            entries = result.get("trace") or []
            entry = entries[-1] if entries else {}
            span.update(output=entry, metadata=entry)
            return result

    return wrapped


def build_graph(
    retriever, llm, config: AgentConfig | None = None, guardrails: Guardrails | None = None
):
    cfg = config or AgentConfig()
    nodes = AgentNodes(retriever, llm, cfg, guardrails)

    g = StateGraph(AgentState)
    g.add_node("retrieve", _traced("retrieve", nodes.retrieve))
    g.add_node("grade_context", _traced("grade_context", nodes.grade_context))
    g.add_node("generate", _traced("generate", nodes.generate))
    g.add_node("cite_critic", _traced("cite_critic", nodes.cite_critic))
    g.add_node("output_guard", _traced("output_guard", nodes.output_guard))

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_context")
    g.add_conditional_edges(
        "grade_context", route_after_grade, {"retrieve": "retrieve", "generate": "generate"}
    )
    g.add_edge("generate", "cite_critic")
    g.add_conditional_edges(
        "cite_critic", route_after_critic, {"generate": "generate", "end": "output_guard"}
    )
    g.add_edge("output_guard", END)
    return g.compile()


def initial_state(question: str, config: AgentConfig | None = None) -> dict:
    cfg = config or AgentConfig()
    return {
        "original_question": question,
        "question": question,
        "k": cfg.k,
        "chunks": [],
        "retrieval_round": 0,
        "grade": {},
        "draft": None,
        "validated": None,
        "answer": None,
        "revision_round": 0,
        "critic": None,
        "best_validated": None,
        "best_critic": None,
        "best_quality": -1.0,
        "guardrail": None,
        "max_retrieval_rounds": cfg.max_retrieval_rounds,
        "max_revision_rounds": cfg.max_revision_rounds,
        "accept_score": cfg.accept_score,
        "trace": [],
    }


def run_agent(app, question: str, config: AgentConfig | None = None) -> dict:
    """Run the compiled graph on a question; returns the final state.

    Wrapped in one root trace so the whole run (every node span + LLM generation)
    groups under a single entry in Langfuse. No-op when tracing is disabled.
    """
    # recursion_limit caps total super-steps as a backstop; our own loop caps
    # stop things well before this.
    with get_tracer().trace("agent-run", input=question) as root:
        final = app.invoke(initial_state(question, config), {"recursion_limit": 50})
        decision = final.get("guardrail") or {}
        answer = final.get("answer")
        citations = [c.citation() for c in answer.citations] if answer else []
        root.update(
            output={
                "action": decision.get("action"),
                "answer": decision.get("final_answer"),
                "confidence": decision.get("confidence"),
                "citations": citations,
            },
            metadata={
                "retrieval_rounds": final.get("retrieval_round"),
                "revision_rounds": final.get("revision_round"),
            },
        )
        return final


def build_agent(config: AgentConfig | None = None, guardrails: Guardrails | None = None):
    """Wire the real retriever + LLM client and compile the graph. Heavy; reuse it."""
    from agentic_rag.llm.client import LLMClient
    from agentic_rag.retrieve.retriever import build_retriever

    return build_graph(build_retriever(), LLMClient(), config or AgentConfig(), guardrails)
