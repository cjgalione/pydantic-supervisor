"""Shared state definitions for compatibility with older imports."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Optional typed state container for agent runs."""

    user_id: str
    session_id: str
