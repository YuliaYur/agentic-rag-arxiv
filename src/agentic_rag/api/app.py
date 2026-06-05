"""FastAPI app exposing the agent.

``create_app(service=None)`` is a factory so tests can inject a fake service (no
models/API) while the real entrypoint (``app = create_app()``) builds the heavy
agent once at startup via the lifespan. Endpoints are sync ``def`` so FastAPI runs
them in its threadpool; the service serializes the actual agent call.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agentic_rag.llm.client import LLMError

from .schemas import HealthResponse, QueryRequest, QueryResponse
from .service import AgentService, build_service

DEFAULT_RATE_LIMIT = "10/minute"


def _rate_limit_value(*_args) -> str:
    # Read per-request so tests (and ops) can tune it via env without re-import.
    return os.getenv("API_RATE_LIMIT", DEFAULT_RATE_LIMIT)


limiter = Limiter(key_func=get_remote_address)


def get_service(request: Request) -> AgentService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Agent is not ready (startup failed?).")
    return service


def create_app(service: AgentService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build the real agent once, unless a service was injected (tests).
        if getattr(app.state, "service", None) is None:
            try:
                # Load .env so `uvicorn agentic_rag.api.app:app` finds the API key etc.
                # (real env wins — override=False).
                from dotenv import load_dotenv

                from agentic_rag.ingest.config import REPO_ROOT

                load_dotenv(REPO_ROOT / ".env", override=False)
                app.state.service = build_service()
            except Exception as exc:  # don't crash the server — /health reports it
                app.state.service = None
                app.state.startup_error = str(exc)
        yield

    app = FastAPI(title="agentic-rag-arxiv", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(LLMError)
    async def _llm_error(_request: Request, exc: LLMError):
        # A known agent/LLM failure — clean error, not a stack trace.
        return JSONResponse(status_code=502, content={"error": "agent_failure", "detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception):
        # Anything else (e.g. an upstream provider quota/rate-limit, retrieval outage)
        # — surface a clean JSON error with a short detail, never a stack trace.
        return JSONResponse(
            status_code=502,
            content={"error": "agent_failure", "detail": f"{type(exc).__name__}: {exc}"[:400]},
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok", agent_ready=getattr(request.app.state, "service", None) is not None
        )

    @app.post("/query", response_model=QueryResponse)
    @limiter.limit(_rate_limit_value)
    def query(
        request: Request,
        body: QueryRequest,
        service: Annotated[AgentService, Depends(get_service)],
    ) -> QueryResponse:
        return service.query(body)

    return app


app = create_app()
