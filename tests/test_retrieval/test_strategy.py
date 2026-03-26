"""Tests for the StrategySelector."""

from src.retrieval.strategy import StrategySelector
from src.schema.enums import QueryType, RetrievalStrategy


class TestStrategyDefinitional:
    """DEFINITIONAL queries should map to VECTOR_SEARCH primary."""

    def test_primary_is_vector_search(self):
        selector = StrategySelector()
        primary, _ = selector.select_strategy(QueryType.DEFINITIONAL)
        assert primary == RetrievalStrategy.VECTOR_SEARCH

    def test_secondaries_include_graph_and_hierarchical(self):
        selector = StrategySelector()
        _, secondaries = selector.select_strategy(QueryType.DEFINITIONAL)
        assert RetrievalStrategy.GRAPH_QUERY in secondaries
        assert RetrievalStrategy.HIERARCHICAL in secondaries


class TestStrategyProcedural:
    """PROCEDURAL queries should map to VECTOR_SEARCH primary."""

    def test_primary_is_vector_search(self):
        selector = StrategySelector()
        primary, _ = selector.select_strategy(QueryType.PROCEDURAL)
        assert primary == RetrievalStrategy.VECTOR_SEARCH

    def test_secondaries_include_graph(self):
        selector = StrategySelector()
        _, secondaries = selector.select_strategy(QueryType.PROCEDURAL)
        assert RetrievalStrategy.GRAPH_QUERY in secondaries


class TestStrategyStructural:
    """STRUCTURAL queries should map to GRAPH_QUERY primary."""

    def test_primary_is_graph_query(self):
        selector = StrategySelector()
        primary, _ = selector.select_strategy(QueryType.STRUCTURAL)
        assert primary == RetrievalStrategy.GRAPH_QUERY

    def test_secondaries_include_vector_search(self):
        selector = StrategySelector()
        _, secondaries = selector.select_strategy(QueryType.STRUCTURAL)
        assert RetrievalStrategy.VECTOR_SEARCH in secondaries


class TestStrategyCompliance:
    """COMPLIANCE queries should map to PROPOSITIONAL primary."""

    def test_primary_is_propositional(self):
        selector = StrategySelector()
        primary, _ = selector.select_strategy(QueryType.COMPLIANCE)
        assert primary == RetrievalStrategy.PROPOSITIONAL

    def test_secondaries_include_graph(self):
        selector = StrategySelector()
        _, secondaries = selector.select_strategy(QueryType.COMPLIANCE)
        assert RetrievalStrategy.GRAPH_QUERY in secondaries


class TestStrategyTemporal:
    """TEMPORAL queries should map to VECTOR_SEARCH primary."""

    def test_primary_is_vector_search(self):
        selector = StrategySelector()
        primary, _ = selector.select_strategy(QueryType.TEMPORAL)
        assert primary == RetrievalStrategy.VECTOR_SEARCH

    def test_secondaries_include_hierarchical(self):
        selector = StrategySelector()
        _, secondaries = selector.select_strategy(QueryType.TEMPORAL)
        assert RetrievalStrategy.HIERARCHICAL in secondaries
