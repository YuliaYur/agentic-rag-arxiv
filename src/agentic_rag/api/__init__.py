"""FastAPI service exposing the agent over a clean ``/query`` endpoint."""

from .app import create_app

__all__ = ["create_app"]
