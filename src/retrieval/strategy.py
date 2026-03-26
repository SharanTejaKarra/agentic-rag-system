"""Strategy selection - maps query types to retrieval strategies."""

from src.schema.enums import QueryType, RetrievalStrategy

# Primary strategy + fallback strategies for each query type.
_STRATEGY_MAP: dict[QueryType, tuple[RetrievalStrategy, list[RetrievalStrategy]]] = {
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


class StrategySelector:
    """Picks primary and secondary retrieval strategies for a query type."""

    def select_strategy(
        self, query_type: QueryType
    ) -> tuple[RetrievalStrategy, list[RetrievalStrategy]]:
        """Return (primary_strategy, secondary_strategies) for the given query type."""
        if query_type in _STRATEGY_MAP:
            return _STRATEGY_MAP[query_type]
        # Default fallback for unknown types
        return (
            RetrievalStrategy.VECTOR_SEARCH,
            [RetrievalStrategy.GRAPH_QUERY],
        )
