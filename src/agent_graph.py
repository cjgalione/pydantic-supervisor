"""Compatibility module that exports the PydanticAI-backed supervisor.

This keeps the original import path stable for scripts and evals.
"""

# Import the PydanticAI-backed supervisor implementation
from src.agents.deep_agent import get_supervisor, run_supervisor_with_critic

# Export for backward compatibility
__all__ = ["get_supervisor", "run_supervisor_with_critic"]
