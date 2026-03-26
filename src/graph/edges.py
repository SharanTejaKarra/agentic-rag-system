"""Conditional edge functions for the LangGraph agent.

The routing logic is what makes this system agentic. After each step,
the system decides whether to explore discovered sections, try more
retrieval strategies, or proceed to synthesis.

The flow:
  retrieve -> evaluate -> [resolve | synthesize]
  resolve  -> evaluate -> [resolve | synthesize]

The evaluate node sets discovered_sections and pending_cross_refs.
The edge functions read those to decide what happens next.
"""

from __future__ import annotations

from src.schema.state import AgentState


def has_sections_to_explore(state: AgentState) -> bool:
    """True if there are discovered sections or pending cross-refs to resolve."""
    discovered = state.get("discovered_sections") or []
    pending = state.get("pending_cross_refs") or []
    explored = set(state.get("explored_sections") or [])

    # Filter discovered to only those not yet explored
    unexplored = [s for s in discovered if s not in explored]

    # Check iteration limit
    iteration = state.get("iteration_count") or 0
    max_iter = state.get("max_iterations") or 3
    if iteration >= max_iter:
        return False

    return len(unexplored) > 0 or len(pending) > 0


def route_after_evaluate(state: AgentState) -> str:
    """Decide the next step after the evaluate node.

    Returns "resolve" if there are sections to explore, "synthesize" otherwise.
    """
    if has_sections_to_explore(state):
        return "resolve"
    return "synthesize"


def route_after_resolve(state: AgentState) -> str:
    """Decide the next step after the resolve node.

    Returns "evaluate" to let the LLM decide if we need more, or
    "synthesize" if we have hit iteration limits.
    """
    iteration = state.get("iteration_count") or 0
    max_iter = state.get("max_iterations") or 3

    if iteration >= max_iter:
        return "synthesize"

    # Go back to evaluate so the LLM can decide if enough context exists
    return "evaluate"
