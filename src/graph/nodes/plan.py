"""Plan retrieval strategy based on the classified query type."""

from __future__ import annotations

from src.schema.enums import QueryType, RetrievalStrategy
from src.schema.models import QueryPlan
from src.schema.state import AgentState

# Maps each query type to (primary_strategy, [secondary_strategies])
_STRATEGY_TABLE: dict[QueryType, tuple[RetrievalStrategy, list[RetrievalStrategy]]] = {
    QueryType.DEFINITIONAL: (
        RetrievalStrategy.VECTOR_SEARCH,
        [RetrievalStrategy.GRAPH_QUERY, RetrievalStrategy.HIERARCHICAL],
    ),
    QueryType.PROCEDURAL: (
        RetrievalStrategy.VECTOR_SEARCH,
        [RetrievalStrategy.GRAPH_QUERY],
    ),
    QueryType.STRUCTURAL: (
        RetrievalStrategy.GRAPH_QUERY,
        [RetrievalStrategy.VECTOR_SEARCH],
    ),
    QueryType.COMPLIANCE: (
        RetrievalStrategy.PROPOSITIONAL,
        [RetrievalStrategy.GRAPH_QUERY],
    ),
    QueryType.TEMPORAL: (
        RetrievalStrategy.VECTOR_SEARCH,
        [RetrievalStrategy.HIERARCHICAL],
    ),
}


def plan_retrieval(state: AgentState) -> dict:
    """Build a retrieval plan from the query type."""
    query_type = state.get("query_type") or QueryType.DEFINITIONAL
    primary, secondaries = _STRATEGY_TABLE[query_type]

    plan = QueryPlan(
        query_type=query_type.value,
        primary_strategy=primary,
        secondary_strategies=secondaries,
    )

    return {"retrieval_plan": plan}
