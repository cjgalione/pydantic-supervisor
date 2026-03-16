"""Compatibility entrypoint for the Modal eval server."""

from src.eval_server import app, braintrust_eval_server

__all__ = ["app", "braintrust_eval_server"]
