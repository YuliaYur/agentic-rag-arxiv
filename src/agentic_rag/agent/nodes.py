"""Graph nodes + routing functions.

Nodes are methods on ``AgentNodes`` so they can hold their dependencies (the
retriever and the LLM client) while LangGraph calls them with just the state.
Each returns a *partial* state update (LangGraph merges it), always including one
``trace`` entry of structured metadata.

Routing functions are module-level and **pure** (state -> next-node name), so the
loop/cap logic is trivially unit-testable without any LLM. The caps live in the
state, so routing needs nothing else.
"""

from __future__ import annotations

from agentic_rag.answer.prompt import SYSTEM_PROMPT, build_user_prompt
from agentic_rag.answer.schemas import CitedAnswer
from agentic_rag.answer.validate import validate_cited_answer
from agentic_rag.guardrails import Guardrails
from agentic_rag.retrieve.retriever import (
    anchor_query_to_title,
    detect_named_papers,
    round_robin_merge,
    title_anchored_query,
)

from .config import AgentConfig
from .prompts import (
    CRITIC_SYSTEM,
    GRADE_SYSTEM,
    build_critic_prompt,
    build_grade_prompt,
    revision_note,
)
from .state import CriticResult, GradeResult


class AgentNodes:
    def __init__(
        self,
        retriever,
        llm,
        config: AgentConfig | None = None,
        guardrails: Guardrails | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._cfg = config or AgentConfig()
        self._guards = guardrails or Guardrails()

    # 1. retrieve -------------------------------------------------------------
    def retrieve(self, state: dict) -> dict:
        rnd = state.get("retrieval_round", 0) + 1
        k = state.get("k", self._cfg.k)
        # Decomposed re-retrieval: if the grader split a comparison into per-side
        # sub-queries, retrieve each side and round-robin merge so every side gets
        # slots (a single embedding of "A vs B" is dominated by A; see ADR-0014).
        sub_queries = (state.get("sub_queries") or [])[: self._cfg.max_sub_queries]
        if sub_queries:
            # Title-anchor each sub-query so a foundational paper (BERT, the
            # original Transformer) isn't buried under the papers that cite it.
            # Skip when already anchored (the deterministic gate pre-anchors its
            # sub-queries; re-anchoring a query that names several papers would
            # cross-contaminate it via the alias of whichever name sorts first).
            if self._cfg.anchor_sub_queries and not state.get("sub_queries_anchored"):
                titles = getattr(self._retriever, "titles", {})
                aliases = dict(self._cfg.paper_names)
                queries = [anchor_query_to_title(q, titles, aliases) for q in sub_queries]
            else:
                queries = list(sub_queries)
            # When the gate targets a specific paper per side, KEEP only that paper's
            # chunks from a deep pool: this guarantees the right paper *and* picks its
            # passage most relevant to the (topical) sub-query. If the topical query
            # didn't surface the target at all (a ViT-heavy "A vs B" can drown the B
            # anchor), fall back to the bare title, which reliably retrieves the paper.
            targets = state.get("sub_query_targets") or []
            titles = getattr(self._retriever, "titles", {})
            per_side = []
            for i, q in enumerate(queries):
                tgt = targets[i] if i < len(targets) else None
                if not tgt:
                    per_side.append(self._retriever.retrieve(q, k))
                    continue
                side = [c for c in self._retriever.retrieve(q, max(k, 50)) if c.arxiv_id == tgt]
                if not side and tgt in titles:
                    side = [
                        c
                        for c in self._retriever.retrieve(titles[tgt], max(k, 50))
                        if c.arxiv_id == tgt
                    ]
                per_side.append(side[:k])
            chunks = round_robin_merge(per_side, k)
            query_used: str | list[str] = queries
        else:
            query_used = state["question"]
            chunks = self._retriever.retrieve(query_used, k)
        # Input guardrail: defang prompt injection in the retrieved text *before*
        # it reaches any prompt. Sanitized chunks flow downstream to all nodes.
        chunks, hits = self._guards.sanitize_chunks(chunks)
        entry = {
            "node": "retrieve",
            "retrieval_round": rnd,
            "query": query_used,
            "decomposed": bool(sub_queries),
            "n_chunks": len(chunks),
            "papers": sorted({c.arxiv_id for c in chunks}),
            "top_sources": [c.citation() for c in chunks[:3]],
            "injection_hits": [h.model_dump() for h in hits],
        }
        return {"chunks": chunks, "retrieval_round": rnd, "trace": [entry]}

    # 2. grade_context --------------------------------------------------------
    def grade_context(self, state: dict) -> dict:
        grade: GradeResult = self._llm.structured(
            GRADE_SYSTEM,
            build_grade_prompt(state["original_question"], state["chunks"]),
            GradeResult,
        )
        sufficient = grade.sufficient
        sub_queries = grade.sub_queries

        # Deterministic override for named-paper coverage — a safety net over the
        # LLM grader, which judges these borderline multi-hop cases inconsistently
        # (see ADR-0014). For a question that names >=2 corpus papers:
        #   * a named paper is MISSING -> force insufficient and decompose into one
        #     title sub-query per named paper (every side, not just the missing one,
        #     so the merge keeps the covered side instead of ping-ponging);
        #   * all named papers PRESENT -> force sufficient, locking the coverage in
        #     so a later LLM-driven re-retrieve can't wander off it and lose a side.
        named, missing = ([], [])
        anchored = False
        targets: list[str] = []
        if self._cfg.enforce_named_paper_coverage:
            named, missing = self._named_coverage(state)
        if named and missing:
            sufficient = False
            # One sub-query per named paper, each = that paper's title terms
            # prepended to the ORIGINAL question. Anchored (so the foundational
            # paper surfaces) AND topical (so we fetch its *relevant* passage, not
            # its abstract — a bare title tanks faithfulness/context metrics). The
            # retrieve node filters each side to its target paper (below).
            titles = self._retriever.titles
            sub_queries = [
                title_anchored_query(titles[aid], state["original_question"]) for aid in named
            ]
            targets = list(named)
            anchored = True
        elif named:  # >=2 named and none missing -> coverage complete
            sufficient = True

        entry = {
            "node": "grade_context",
            "retrieval_round": state["retrieval_round"],
            "sufficient": sufficient,
            "refined_query": grade.refined_query,
            "sub_queries": sub_queries,
            "named_papers": named,
            "missing_papers": missing,
            "reasoning": grade.reasoning,
        }
        updates: dict = {
            "grade": {**grade.model_dump(), "sufficient": sufficient},
            "trace": [entry],
        }
        # Reformulate the retrieval query only if we might loop back. Carry the
        # per-side sub-queries too: the retrieve node fans out over them when
        # present (decomposed re-retrieval), else falls back to the single query.
        if not sufficient:
            updates["question"] = grade.refined_query or state["original_question"]
            updates["sub_queries"] = sub_queries
            updates["sub_queries_anchored"] = anchored
            updates["sub_query_targets"] = targets
        return updates

    def _named_coverage(self, state: dict) -> tuple[list[str], list[str]]:
        """(named, missing) arxiv_ids: papers the question names, and which aren't
        retrieved. Only multi-paper (>=2 named) questions can have a coverage gap."""
        titles = getattr(self._retriever, "titles", {})
        if not titles:
            return [], []
        named = detect_named_papers(state["original_question"], self._cfg.paper_names)
        named &= titles.keys()  # only papers actually in the index
        if len(named) < 2:
            return [], []
        retrieved = {c.arxiv_id for c in state["chunks"]}
        return sorted(named), sorted(named - retrieved)

    # 3. generate -------------------------------------------------------------
    def generate(self, state: dict) -> dict:
        critic = state.get("critic")
        is_revision = critic is not None and not critic.get("supported", True)
        revision_round = state.get("revision_round", 0) + (1 if is_revision else 0)

        # Always answer the ORIGINAL question (retrieval query may be reformulated).
        user = build_user_prompt(state["original_question"], state["chunks"])
        if is_revision:
            user += revision_note(critic)

        draft: CitedAnswer = self._llm.structured(SYSTEM_PROMPT, user, CitedAnswer)
        validated = validate_cited_answer(state["original_question"], draft, state["chunks"])
        entry = {
            "node": "generate",
            "revision_round": revision_round,
            "is_revision": is_revision,
            "grounded": validated.is_grounded,
            "n_citations": len(validated.citations),
            # Resolved, human-readable citations (title + section + page), rebuilt
            "citations": [c.citation() for c in validated.citations],
            "insufficient_context": validated.insufficient_context,
        }
        return {
            "draft": draft,
            "validated": validated,
            "answer": validated,
            "revision_round": revision_round,
            "trace": [entry],
        }

    # 4. cite_critic ----------------------------------------------------------
    def cite_critic(self, state: dict) -> dict:
        critic: CriticResult = self._llm.structured(
            CRITIC_SYSTEM,
            build_critic_prompt(state["original_question"], state["validated"], state["chunks"]),
            CriticResult,
        )
        # Keep-best: rank this draft and adopt it only if it STRICTLY beats the best
        # so far (ties keep the earlier draft). This makes revisions pure upside —
        # the final answer can never be worse than the first draft.
        quality = _draft_quality(state["validated"], critic)
        updates: dict = {"critic": critic.model_dump()}
        if quality > state.get("best_quality", -1.0):
            updates["best_validated"] = state["validated"]
            updates["best_critic"] = critic.model_dump()
            updates["best_quality"] = quality
            became_best = True
        else:
            became_best = False
        updates["trace"] = [
            {
                "node": "cite_critic",
                "revision_round": state.get("revision_round", 0),
                "supported": critic.supported,
                "critic_score": critic.score,
                "n_unsupported": len(critic.unsupported_claims),
                "draft_quality": quality,
                "became_best": became_best,
            }
        ]
        return updates

    # 5. output_guard ---------------------------------------------------------
    def output_guard(self, state: dict) -> dict:
        """Final gate over the BEST draft: structure + abstain + grounding/confidence."""
        best = state.get("best_validated") or state.get("answer")
        best_critic = state.get("best_critic") or state.get("critic")
        decision = self._guards.check_output(best, best_critic)
        entry = {
            "node": "output_guard",
            "action": decision.action,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "failed_checks": [c.name for c in decision.checks if not c.passed],
        }
        # Promote the best draft to the final answer (it may not be the last one).
        return {"guardrail": decision.model_dump(), "answer": best, "trace": [entry]}


# --- keep-best ranking -------------------------------------------------------


def _draft_quality(validated, critic) -> float:
    """Rank a draft so keep-best can pick the strongest answer seen.

    Order (worst to best): ungrounded (fabricated citations) < honest refusal <
    a real grounded answer scored by the critic's supported-claim fraction. So a
    grounded answer always beats a refusal, and a refusal always beats fabrication.
    """
    if validated is None:
        return -1.0
    if not validated.is_grounded:
        return 0.0
    if validated.insufficient_context:
        return 0.1
    return float(getattr(critic, "score", 0.0))


# --- routing (pure) ----------------------------------------------------------


def route_after_grade(state: dict) -> str:
    """Loop back to retrieve if context is weak and we have rounds left; else generate."""
    if (
        state["grade"].get("sufficient")
        or state["retrieval_round"] >= state["max_retrieval_rounds"]
    ):
        return "generate"
    return "retrieve"


def route_after_critic(state: dict) -> str:
    """Revise only if the answer is below the acceptance bar and revisions remain.

    Stop (go to output_guard) when the critic is fully satisfied, OR the
    supported-claim fraction already clears ``accept_score`` ("good enough", so we
    don't churn), OR we've hit the revision cap.
    """
    critic = state.get("critic") or {}
    accept = state.get("accept_score", 1.0)
    if (
        critic.get("supported")
        or critic.get("score", 0.0) >= accept
        or state.get("revision_round", 0) >= state["max_revision_rounds"]
    ):
        return "end"
    return "generate"
