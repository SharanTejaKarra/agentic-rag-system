"""Execute retrieval using the planned strategy."""

from __future__ import annotations

from src.schema.enums import RetrievalStrategy
from src.schema.models import RetrievalResult
from src.schema.state import AgentState
from src.tools.cross_reference import cross_reference_search
from src.tools.graph_query import graph_query
from src.tools.hierarchical_lookup import hierarchical_lookup
from src.tools.propositional_search import propositional_search
from src.tools.sub_question import sub_question_search
from src.tools.vector_search import vector_search

# Minimum chunk count before we try secondary strategies
_SPARSE_THRESHOLD = 3

_TOOL_MAP = {
    RetrievalStrategy.VECTOR_SEARCH: vector_search,
    RetrievalStrategy.GRAPH_QUERY: graph_query,
    RetrievalStrategy.CROSS_REFERENCE: cross_reference_search,
    RetrievalStrategy.SUB_QUESTION: sub_question_search,
    RetrievalStrategy.HIERARCHICAL: hierarchical_lookup,
    RetrievalStrategy.PROPOSITIONAL: propositional_search,
}


def execute_retrieval(state: AgentState) -> dict:
    """Run the primary retrieval strategy; fall back to secondaries if sparse."""
    plan = state["retrieval_plan"]
    query = state["original_query"]

    # Run primary strategy
    tool_fn = _TOOL_MAP.get(plan.primary_strategy)
    if tool_fn is None:
        return {"retrieved_results": []}

    primary_result = tool_fn(query)
    results: list[RetrievalResult] = [
        RetrievalResult(
            chunks=primary_result,
            strategy_used=plan.primary_strategy,
        )
    ]

    # If primary results are sparse, try secondary strategies
    if len(primary_result) < _SPARSE_THRESHOLD:
        for strategy in plan.secondary_strategies:
            fallback_fn = _TOOL_MAP.get(strategy)
            if fallback_fn is None:
                continue
            secondary_result = fallback_fn(query)
            results.append(
                RetrievalResult(
                    chunks=secondary_result,
                    strategy_used=strategy,
                )
            )

    return {"retrieved_results": results}
