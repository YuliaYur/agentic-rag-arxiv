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

from .config import AgentConfig
from .nodes import AgentNodes, route_after_critic, route_after_grade
from .state import AgentState


def build_graph(
    retriever, llm, config: AgentConfig | None = None, guardrails: Guardrails | None = None
):
    cfg = config or AgentConfig()
    nodes = AgentNodes(retriever, llm, cfg, guardrails)

    g = StateGraph(AgentState)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_context", nodes.grade_context)
    g.add_node("generate", nodes.generate)
    g.add_node("cite_critic", nodes.cite_critic)
    g.add_node("output_guard", nodes.output_guard)

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
        "guardrail": None,
        "max_retrieval_rounds": cfg.max_retrieval_rounds,
        "max_revision_rounds": cfg.max_revision_rounds,
        "trace": [],
    }


def run_agent(app, question: str, config: AgentConfig | None = None) -> dict:
    """Run the compiled graph on a question; returns the final state."""
    # recursion_limit caps total super-steps as a backstop; our own loop caps
    # stop things well before this.
    return app.invoke(initial_state(question, config), {"recursion_limit": 50})


def build_agent(config: AgentConfig | None = None, guardrails: Guardrails | None = None):
    """Wire the real retriever + LLM client and compile the graph. Heavy; reuse it."""
    from agentic_rag.llm.client import LLMClient
    from agentic_rag.retrieve.retriever import build_retriever

    return build_graph(build_retriever(), LLMClient(), config or AgentConfig(), guardrails)
