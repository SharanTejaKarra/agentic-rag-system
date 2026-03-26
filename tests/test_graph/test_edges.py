"""Tests for conditional edge functions."""

from src.graph.edges import has_sections_to_explore, route_after_evaluate, route_after_resolve
from src.schema.enums import RetrievalStrategy
from src.schema.models import CrossReference, QueryPlan, RetrievalResult, Chunk


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", content=f"Chunk {i}", section_ref=f"sec_{i}")
        for i in range(n)
    ]


class TestHasSectionsToExplore:
    """When there are discovered sections or pending cross-refs."""

    def test_true_with_discovered_sections(self):
        state = {
            "discovered_sections": ["31.020", "31.060"],
            "explored_sections": [],
            "pending_cross_refs": [],
            "iteration_count": 0,
            "max_iterations": 3,
        }
        assert has_sections_to_explore(state) is True

    def test_false_when_all_explored(self):
        state = {
            "discovered_sections": ["31.020"],
            "explored_sections": ["31.020"],
            "pending_cross_refs": [],
            "iteration_count": 0,
            "max_iterations": 3,
        }
        assert has_sections_to_explore(state) is False

    def test_true_with_pending_crossrefs(self):
        state = {
            "discovered_sections": [],
            "explored_sections": [],
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
        assert has_sections_to_explore(state) is True

    def test_false_at_max_iterations(self):
        state = {
            "discovered_sections": ["31.020"],
            "explored_sections": [],
            "pending_cross_refs": [],
            "iteration_count": 3,
            "max_iterations": 3,
        }
        assert has_sections_to_explore(state) is False

    def test_false_with_nothing_pending(self):
        state = {
            "discovered_sections": [],
            "explored_sections": [],
            "pending_cross_refs": [],
            "iteration_count": 0,
            "max_iterations": 3,
        }
        assert has_sections_to_explore(state) is False


class TestRouteAfterEvaluate:
    """Routing decisions after the evaluate node."""

    def test_routes_to_resolve_when_sections_discovered(self):
        state = {
            "discovered_sections": ["31.020"],
            "explored_sections": [],
            "pending_cross_refs": [],
            "iteration_count": 0,
            "max_iterations": 3,
        }
        assert route_after_evaluate(state) == "resolve"

    def test_routes_to_synthesize_when_nothing_to_explore(self):
        state = {
            "discovered_sections": [],
            "explored_sections": ["31.020"],
            "pending_cross_refs": [],
            "iteration_count": 1,
            "max_iterations": 3,
        }
        assert route_after_evaluate(state) == "synthesize"


class TestRouteAfterResolve:
    """Routing decisions after the resolve node."""

    def test_routes_to_evaluate_within_limit(self):
        state = {
            "iteration_count": 1,
            "max_iterations": 3,
        }
        assert route_after_resolve(state) == "evaluate"

    def test_routes_to_synthesize_at_max_iterations(self):
        state = {
            "iteration_count": 3,
            "max_iterations": 3,
        }
        assert route_after_resolve(state) == "synthesize"
