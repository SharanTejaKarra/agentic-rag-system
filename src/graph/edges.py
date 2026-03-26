"""Conditional edge functions for the LangGraph agent."""

from __future__ import annotations

from src.schema.state import AgentState


def should_resolve_refs(state: AgentState) -> bool:
    """True if there are pending unresolved cross-references within iteration limit."""
    pending = state.get("pending_cross_refs") or []
    iteration = state.get("iteration_count") or 0
    max_iter = state.get("max_iterations") or 3
    return len(pending) > 0 and iteration < max_iter


def needs_more_retrieval(state: AgentState) -> bool:
    """True if retrieved results are sparse and secondary strategies remain untried."""
    results = state.get("retrieved_results") or []
    plan = state.get("retrieval_plan")
    if not plan:
        return False

    total_chunks = sum(len(r.chunks) for r in results)
    strategies_used = {r.strategy_used for r in results}
    untried = [s for s in plan.secondary_strategies if s not in strategies_used]

    return total_chunks < 3 and len(untried) > 0


def route_after_retrieval(state: AgentState) -> str:
    """Decide the next step after retrieval or resolution.

    Returns one of: "resolve", "synthesize", "retrieve".
    """
    if should_resolve_refs(state):
        return "resolve"
    if needs_more_retrieval(state):
        return "retrieve"
    return "synthesize"
