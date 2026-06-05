"""Streamlit demo UI for the agentic-RAG service.

A thin HTTP client of the FastAPI ``/query`` endpoint (so the UI *demonstrates* the
service, and either can run alone). Shows the cited answer, its sources, and a
panel tracing the agent's steps (retrieve → grade → generate → cite-critic →
guardrail) plus cost/latency — surfacing the reasoning is what sells the demo.

    pip install -e ".[ui]"
    streamlit run ui/streamlit_app.py        # API_URL defaults to http://localhost:8000
"""

from __future__ import annotations

import os

import requests
import streamlit as st

DEFAULT_API = os.getenv("API_URL", "http://localhost:8000")
STEP_ICON = {
    "retrieve": "🔎",
    "grade_context": "⚖️",
    "generate": "✍️",
    "cite_critic": "🔍",
    "output_guard": "🛡️",
}

st.set_page_config(page_title="Agentic RAG · arXiv", layout="wide")
st.title("🔬 Agentic RAG over the transformer-lineage papers")
st.caption(
    "Ask a question about ~20 ML papers. The agent retrieves, grades the context, "
    "generates a cited answer, audits its own citations, and a guardrail decides "
    "whether to answer or decline."
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", DEFAULT_API).rstrip("/")
    k = st.slider("Chunks per round (k)", 3, 10, 5)
    max_rounds = st.slider("Max retrieval rounds", 1, 3, 3)
    try:
        health = requests.get(f"{api_url}/health", timeout=3).json()
        (st.success if health.get("agent_ready") else st.warning)(
            "agent ready" if health.get("agent_ready") else "agent not ready"
        )
    except Exception:
        st.error("API unreachable — start it with `uvicorn agentic_rag.api.app:app`")

question = st.text_input(
    "Your question",
    placeholder="How does ELECTRA's pre-training objective differ from BERT's?",
)
ask = st.button("Ask", type="primary")

if ask and question.strip():
    with st.spinner("The agent is retrieving, generating and self-checking…"):
        try:
            resp = requests.post(
                f"{api_url}/query",
                json={"question": question, "k": k, "max_retrieval_rounds": max_rounds},
                timeout=180,
            )
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    if resp.status_code != 200:
        body = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        st.error(f"{resp.status_code}: {body.get('detail') or body.get('error') or resp.text}")
        st.stop()

    data = resp.json()
    answer_col, steps_col = st.columns([3, 2], gap="large")

    with answer_col:
        if data["action"] == "answer":
            grounded = "grounded ✅" if data["grounded"] else "ungrounded ⚠️"
            st.markdown(f"**Answer** · confidence {data['confidence']:.0%} · {grounded}")
        else:
            st.warning(f"The agent **declined** (confidence {data['confidence']:.0%}).")
        st.markdown(data["answer"])

        if data["citations"]:
            st.subheader("Sources")
            for c in data["citations"]:
                pages = (
                    f"p.{c['page']}"
                    if c["page"] == c["page_end"]
                    else f"p.{c['page']}-{c['page_end']}"
                )
                st.markdown(
                    f"- **[{c['source_id']}]** [{c['title']}]({c['url']}) · §{c['section']} · {pages}"
                )

    with steps_col:
        st.subheader("How the agent got here")
        for s in data["steps"]:
            st.markdown(f"{STEP_ICON.get(s['node'], '•')} **{s['node']}** — {s['summary']}")
        m = data["metering"]
        st.divider()
        st.caption(
            f"💸 ${m['cost_usd']:.5f} · {m['llm_calls']} LLM calls · "
            f"{m['cache_hits']} cache hits · {m['latency_ms'] / 1000:.1f}s"
        )
        st.caption(
            f"retrieval rounds: {data['retrieval_rounds']} · revisions: {data['revision_rounds']}"
        )
