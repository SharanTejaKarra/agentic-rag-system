"""Tests for conditional edge functions."""

from src.graph.edges import needs_more_retrieval, route_after_retrieval, should_resolve_refs
from src.schema.enums import RetrievalStrategy
from src.schema.models import CrossReference, QueryPlan, RetrievalResult, Chunk


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", content=f"Chunk {i}", section_ref=f"sec_{i}")
        for i in range(n)
    ]


class TestShouldResolveRefsTrue:
    """When pending cross-refs exist and within iteration limit."""

    def test_returns_true_with_pending_refs(self):
        state = {
            "pending_cross_refs": [
                CrossReference(
                    source_section="1.0",
                    target_section="2.0",
                    reference_text="See Section 2.0",
                ),
            ],
            "iteration_count": 0,
            "max_iterations": 3,
        }
        assert should_resolve_refs(state) is True

    def test_returns_true_at_iteration_boundary(self):
        state = {
            "pending_cross_refs": [
                CrossReference(
                    source_section="1.0",
                    target_section="2.0",
                    reference_text="See Section 2.0",
                ),
            ],
            "iteration_count": 2,
            "max_iterations": 3,
        }
        assert should_resolve_refs(state) is True


class TestShouldResolveRefsFalse:
    """When no pending cross-refs or iteration limit reached."""

    def test_returns_false_with_no_pending(self):
        state = {
            "pending_cross_refs": [],
            "iteration_count": 0,
            "max_iterations": 3,
        }
        assert should_resolve_refs(state) is False

    def test_returns_false_at_max_iterations(self):
        state = {
            "pending_cross_refs": [
                CrossReference(
                    source_section="1.0",
                    target_section="2.0",
                    reference_text="See Section 2.0",
                ),
            ],
            "iteration_count": 3,
            "max_iterations": 3,
        }
        assert should_resolve_refs(state) is False

    def test_returns_false_with_none_pending(self):
        state = {
            "pending_cross_refs": None,
            "iteration_count": 0,
            "max_iterations": 3,
        }
        assert should_resolve_refs(state) is False


class TestRouteAfterRetrieval:
    """Test all three routing outcomes from route_after_retrieval."""

    def test_routes_to_resolve_when_refs_pending(self):
        state = {
            "pending_cross_refs": [
                CrossReference(
                    source_section="1.0",
                    target_section="2.0",
                    reference_text="See Section 2.0",
                ),
            ],
            "iteration_count": 0,
            "max_iterations": 3,
            "retrieved_results": [
                RetrievalResult(
                    chunks=_make_chunks(5),
                    strategy_used=RetrievalStrategy.VECTOR_SEARCH,
                ),
            ],
            "retrieval_plan": QueryPlan(
                query_type="definitional",
                primary_strategy=RetrievalStrategy.VECTOR_SEARCH,
                secondary_strategies=[],
            ),
        }
        assert route_after_retrieval(state) == "resolve"

    def test_routes_to_retrieve_when_sparse_and_untried(self):
        state = {
            "pending_cross_refs": [],
            "iteration_count": 0,
            "max_iterations": 3,
            "retrieved_results": [
                RetrievalResult(
                    chunks=_make_chunks(1),
                    strategy_used=RetrievalStrategy.VECTOR_SEARCH,
                ),
            ],
            "retrieval_plan": QueryPlan(
                query_type="definitional",
                primary_strategy=RetrievalStrategy.VECTOR_SEARCH,
                secondary_strategies=[RetrievalStrategy.GRAPH_QUERY],
            ),
        }
        assert route_after_retrieval(state) == "retrieve"

    def test_routes_to_synthesize_when_enough_chunks(self):
        state = {
            "pending_cross_refs": [],
            "iteration_count": 0,
            "max_iterations": 3,
            "retrieved_results": [
                RetrievalResult(
                    chunks=_make_chunks(5),
                    strategy_used=RetrievalStrategy.VECTOR_SEARCH,
                ),
            ],
            "retrieval_plan": QueryPlan(
                query_type="definitional",
                primary_strategy=RetrievalStrategy.VECTOR_SEARCH,
                secondary_strategies=[],
            ),
        }
        assert route_after_retrieval(state) == "synthesize"

    def test_routes_to_synthesize_when_sparse_but_all_tried(self):
        state = {
            "pending_cross_refs": [],
            "iteration_count": 0,
            "max_iterations": 3,
            "retrieved_results": [
                RetrievalResult(
                    chunks=_make_chunks(1),
                    strategy_used=RetrievalStrategy.VECTOR_SEARCH,
                ),
                RetrievalResult(
                    chunks=_make_chunks(1),
                    strategy_used=RetrievalStrategy.GRAPH_QUERY,
                ),
            ],
            "retrieval_plan": QueryPlan(
                query_type="definitional",
                primary_strategy=RetrievalStrategy.VECTOR_SEARCH,
                secondary_strategies=[RetrievalStrategy.GRAPH_QUERY],
            ),
        }
        assert route_after_retrieval(state) == "synthesize"
