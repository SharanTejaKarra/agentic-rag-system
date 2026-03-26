"""Tests for LangGraph node functions."""

from unittest.mock import patch

import pytest

from src.schema.enums import Confidence, QueryType, RetrievalStrategy
from src.schema.models import Chunk, CrossReference, QueryPlan, RetrievalResult


class TestParseQueryClassification:
    """Verify parse node correctly classifies different query types."""

    def test_parse_definitional(self):
        from src.graph.nodes.parse import parse_query

        response = '{"query_type": "definitional", "key_concepts": ["qualified beneficiary"]}'
        with patch("src.graph.nodes.parse.get_llm_response", return_value=response):
            state = {"original_query": "What is a qualified beneficiary?"}
            result = parse_query(state)

        assert result["query_type"] == QueryType.DEFINITIONAL

    def test_parse_procedural(self):
        from src.graph.nodes.parse import parse_query

        response = '{"query_type": "procedural", "key_concepts": ["file an appeal"]}'
        with patch("src.graph.nodes.parse.get_llm_response", return_value=response):
            state = {"original_query": "How do I file an appeal?"}
            result = parse_query(state)

        assert result["query_type"] == QueryType.PROCEDURAL

    def test_parse_structural(self):
        from src.graph.nodes.parse import parse_query

        response = '{"query_type": "structural", "key_concepts": ["Section 12", "Section 31"]}'
        with patch("src.graph.nodes.parse.get_llm_response", return_value=response):
            state = {"original_query": "How do Section 12 and Section 31 relate?"}
            result = parse_query(state)

        assert result["query_type"] == QueryType.STRUCTURAL

    def test_parse_compliance(self):
        from src.graph.nodes.parse import parse_query

        response = '{"query_type": "compliance", "key_concepts": ["notice requirement"]}'
        with patch("src.graph.nodes.parse.get_llm_response", return_value=response):
            state = {"original_query": "Am I violating the notice requirement?"}
            result = parse_query(state)

        assert result["query_type"] == QueryType.COMPLIANCE

    def test_parse_temporal(self):
        from src.graph.nodes.parse import parse_query

        response = '{"query_type": "temporal", "key_concepts": ["amendment", "effective date"]}'
        with patch("src.graph.nodes.parse.get_llm_response", return_value=response):
            state = {"original_query": "When does the amendment take effect?"}
            result = parse_query(state)

        assert result["query_type"] == QueryType.TEMPORAL

    def test_parse_falls_back_on_bad_json(self):
        from src.graph.nodes.parse import parse_query

        with patch("src.graph.nodes.parse.get_llm_response", return_value="not valid json"):
            state = {"original_query": "random query"}
            result = parse_query(state)

        assert result["query_type"] == QueryType.DEFINITIONAL


class TestPlanRetrievalStrategy:
    """Verify plan node selects correct strategies via fallback table.

    plan_retrieval now calls get_llm_response. We mock it to raise so the
    fallback table is exercised, which is the code path these tests cover.
    """

    def _plan_with_fallback(self, state):
        from src.graph.nodes.plan import plan_retrieval

        with patch("src.graph.nodes.plan.get_llm_response", side_effect=RuntimeError("mock")):
            return plan_retrieval(state)

    def test_plan_definitional(self):
        state = {"query_type": QueryType.DEFINITIONAL, "original_query": "What is X?"}
        result = self._plan_with_fallback(state)

        plan = result["retrieval_plan"]
        assert plan.primary_strategy == RetrievalStrategy.VECTOR_SEARCH
        assert RetrievalStrategy.GRAPH_QUERY in plan.secondary_strategies
        assert RetrievalStrategy.HIERARCHICAL in plan.secondary_strategies

    def test_plan_procedural(self):
        state = {"query_type": QueryType.PROCEDURAL, "original_query": "How do I do X?"}
        result = self._plan_with_fallback(state)

        plan = result["retrieval_plan"]
        assert plan.primary_strategy == RetrievalStrategy.VECTOR_SEARCH
        assert RetrievalStrategy.GRAPH_QUERY in plan.secondary_strategies

    def test_plan_structural(self):
        state = {"query_type": QueryType.STRUCTURAL, "original_query": "How do A and B relate?"}
        result = self._plan_with_fallback(state)

        plan = result["retrieval_plan"]
        assert plan.primary_strategy == RetrievalStrategy.VECTOR_SEARCH
        # vector_search is always included as a safety net
        assert RetrievalStrategy.GRAPH_QUERY in plan.secondary_strategies

    def test_plan_compliance(self):
        state = {"query_type": QueryType.COMPLIANCE, "original_query": "Am I violating X?"}
        result = self._plan_with_fallback(state)

        plan = result["retrieval_plan"]
        assert plan.primary_strategy == RetrievalStrategy.VECTOR_SEARCH
        assert RetrievalStrategy.PROPOSITIONAL in plan.secondary_strategies

    def test_plan_temporal(self):
        state = {"query_type": QueryType.TEMPORAL, "original_query": "When does X happen?"}
        result = self._plan_with_fallback(state)

        plan = result["retrieval_plan"]
        assert plan.primary_strategy == RetrievalStrategy.VECTOR_SEARCH
        assert RetrievalStrategy.HIERARCHICAL in plan.secondary_strategies

    def test_plan_defaults_to_definitional(self):
        state = {"query_type": None, "original_query": "Something"}
        result = self._plan_with_fallback(state)

        plan = result["retrieval_plan"]
        assert plan.primary_strategy == RetrievalStrategy.VECTOR_SEARCH

    def test_plan_uses_llm_when_available(self):
        """When LLM responds with valid JSON, use its choices."""
        from src.graph.nodes.plan import plan_retrieval

        llm_response = (
            '{"primary_strategy": "graph_query", '
            '"secondary_strategies": ["vector_search"], '
            '"reasoning": "test"}'
        )
        state = {"query_type": QueryType.STRUCTURAL, "original_query": "How do A and B relate?"}
        with patch("src.graph.nodes.plan.get_llm_response", return_value=llm_response):
            result = plan_retrieval(state)

        plan = result["retrieval_plan"]
        assert plan.primary_strategy == RetrievalStrategy.GRAPH_QUERY
        assert RetrievalStrategy.VECTOR_SEARCH in plan.secondary_strategies


class TestRetrieveCallsCorrectTool:
    """Verify retrieve node dispatches to the right tool."""

    def test_dispatches_vector_search(self, sample_chunks):
        from unittest.mock import MagicMock
        from src.graph.nodes import retrieve as retrieve_mod

        mock_vs = MagicMock(return_value=sample_chunks[:3])

        plan = QueryPlan(
            query_type="definitional",
            primary_strategy=RetrievalStrategy.VECTOR_SEARCH,
            secondary_strategies=[],
        )

        patched_map = {**retrieve_mod._TOOL_MAP, RetrievalStrategy.VECTOR_SEARCH: mock_vs}
        with patch.dict(retrieve_mod._TOOL_MAP, patched_map):
            state = {
                "original_query": "What is a qualified beneficiary?",
                "retrieval_plan": plan,
            }
            result = retrieve_mod.execute_retrieval(state)

            mock_vs.assert_called_once_with("What is a qualified beneficiary?")
            assert len(result["retrieved_results"]) == 1
            assert result["retrieved_results"][0].strategy_used == RetrievalStrategy.VECTOR_SEARCH

    def test_dispatches_graph_query(self, sample_chunks):
        from unittest.mock import MagicMock
        from src.graph.nodes import retrieve as retrieve_mod

        mock_gq = MagicMock(return_value=sample_chunks[:1])

        plan = QueryPlan(
            query_type="structural",
            primary_strategy=RetrievalStrategy.GRAPH_QUERY,
            secondary_strategies=[],
        )

        patched_map = {**retrieve_mod._TOOL_MAP, RetrievalStrategy.GRAPH_QUERY: mock_gq}
        with patch.dict(retrieve_mod._TOOL_MAP, patched_map):
            state = {
                "original_query": "How do Section 12 and Section 31 relate?",
                "retrieval_plan": plan,
            }
            result = retrieve_mod.execute_retrieval(state)

            mock_gq.assert_called_once()
            assert result["retrieved_results"][0].strategy_used == RetrievalStrategy.GRAPH_QUERY

    def test_falls_back_to_secondary_when_sparse(self, sample_chunks):
        from unittest.mock import MagicMock
        from src.graph.nodes import retrieve as retrieve_mod

        mock_vs = MagicMock(return_value=sample_chunks[:1])
        mock_gq = MagicMock(return_value=sample_chunks[1:3])

        plan = QueryPlan(
            query_type="definitional",
            primary_strategy=RetrievalStrategy.VECTOR_SEARCH,
            secondary_strategies=[RetrievalStrategy.GRAPH_QUERY],
        )

        patched_map = {
            **retrieve_mod._TOOL_MAP,
            RetrievalStrategy.VECTOR_SEARCH: mock_vs,
            RetrievalStrategy.GRAPH_QUERY: mock_gq,
        }
        with patch.dict(retrieve_mod._TOOL_MAP, patched_map):
            state = {
                "original_query": "What is X?",
                "retrieval_plan": plan,
            }
            result = retrieve_mod.execute_retrieval(state)

            # Primary returned <3 chunks, so secondary should fire
            assert len(result["retrieved_results"]) == 2


class TestResolveDetectsCrossRefs:
    """Verify resolve node finds section references in chunks."""

    def _patch_resolve_tools(self):
        """Patch all external tool calls in resolve.py."""
        return [
            patch("src.graph.nodes.resolve.cross_reference_search", return_value=[]),
            patch("src.graph.nodes.resolve.graph_query", return_value=[]),
            patch("src.graph.nodes.resolve.hierarchical_lookup", return_value=[]),
        ]

    def test_detects_pending_refs(self, sample_chunks):
        from src.graph.nodes.resolve import resolve_cross_references

        state = {
            "retrieved_results": [
                RetrievalResult(
                    chunks=[sample_chunks[1]],
                    strategy_used=RetrievalStrategy.VECTOR_SEARCH,
                ),
            ],
            "pending_cross_refs": [
                CrossReference(
                    source_section="Section 12.100",
                    target_section="Section 31.020",
                    reference_text="See also Section 31.020(a)(1)",
                ),
            ],
            "resolved_cross_refs": [],
            "iteration_count": 0,
            "max_iterations": 3,
        }

        with patch("src.graph.nodes.resolve.cross_reference_search", return_value=[sample_chunks[0]]), \
             patch("src.graph.nodes.resolve.graph_query", return_value=[]), \
             patch("src.graph.nodes.resolve.hierarchical_lookup", return_value=[]):
            result = resolve_cross_references(state)

            assert len(result["resolved_cross_refs"]) == 1
            assert result["resolved_cross_refs"][0].resolved is True

    def test_increments_iteration(self, sample_state):
        from src.graph.nodes.resolve import resolve_cross_references

        sample_state["iteration_count"] = 1

        patches = self._patch_resolve_tools()
        for p in patches:
            p.start()
        try:
            result = resolve_cross_references(sample_state)
            assert result["iteration_count"] == 2
        finally:
            for p in patches:
                p.stop()

    def test_explores_discovered_sections(self, sample_chunks):
        """When discovered_sections has entries, resolve calls graph and hierarchy tools."""
        from src.graph.nodes.resolve import resolve_cross_references

        state = {
            "discovered_sections": ["Section 31.020"],
            "explored_sections": [],
            "pending_cross_refs": [],
            "resolved_cross_refs": [],
            "retrieved_results": [],
            "iteration_count": 0,
            "max_iterations": 3,
        }

        with patch("src.graph.nodes.resolve.cross_reference_search", return_value=[]), \
             patch("src.graph.nodes.resolve.graph_query", return_value=sample_chunks[:1]) as mock_gq, \
             patch("src.graph.nodes.resolve.hierarchical_lookup", return_value=[]):
            result = resolve_cross_references(state)

            mock_gq.assert_called_once_with("Section 31.020")
            assert "Section 31.020" in result["explored_sections"]

    def test_skips_already_explored_sections(self, sample_chunks):
        """Sections in explored_sections are not queried again."""
        from src.graph.nodes.resolve import resolve_cross_references

        state = {
            "discovered_sections": ["Section 31.020"],
            "explored_sections": ["Section 31.020"],
            "pending_cross_refs": [],
            "resolved_cross_refs": [],
            "retrieved_results": [],
            "iteration_count": 0,
            "max_iterations": 3,
        }

        with patch("src.graph.nodes.resolve.cross_reference_search", return_value=[]), \
             patch("src.graph.nodes.resolve.graph_query", return_value=[]) as mock_gq, \
             patch("src.graph.nodes.resolve.hierarchical_lookup", return_value=[]):
            result = resolve_cross_references(state)

            mock_gq.assert_not_called()


class TestEvaluateRetrieval:
    """Verify the evaluate node decides when to synthesize vs explore more."""

    def test_proceeds_to_synthesis_at_max_iterations(self):
        from src.graph.nodes.evaluate import evaluate_retrieval

        state = {
            "original_query": "What is X?",
            "iteration_count": 3,
            "max_iterations": 3,
            "retrieved_results": [],
        }
        result = evaluate_retrieval(state)

        # At max iterations, should clear sections and go to synthesis
        assert result["discovered_sections"] == []

    def test_skips_eval_when_no_chunks(self):
        from src.graph.nodes.evaluate import evaluate_retrieval

        state = {
            "original_query": "What is X?",
            "iteration_count": 0,
            "max_iterations": 3,
            "retrieved_results": [],
        }
        result = evaluate_retrieval(state)

        assert result["discovered_sections"] == []

    def test_proceeds_when_enough_chunks_and_no_undiscovered(self, sample_state):
        from src.graph.nodes.evaluate import evaluate_retrieval

        # sample_state has 3 chunks and no discovered_sections
        result = evaluate_retrieval(sample_state)

        assert result["discovered_sections"] == []

    def test_asks_llm_when_undiscovered_sections_exist(self, sample_state):
        from src.graph.nodes.evaluate import evaluate_retrieval

        sample_state["discovered_sections"] = ["Section 45.200"]
        sample_state["explored_sections"] = []

        llm_response = (
            '{"sufficient": false, '
            '"explore_sections": ["Section 45.200"], '
            '"reasoning": "need penalty details"}'
        )
        with patch("src.graph.nodes.evaluate.get_llm_response", return_value=llm_response):
            result = evaluate_retrieval(sample_state)

        assert "Section 45.200" in result["discovered_sections"]

    def test_falls_back_to_heuristic_when_llm_fails(self, sample_state):
        from src.graph.nodes.evaluate import evaluate_retrieval

        sample_state["discovered_sections"] = ["Section 45.200"]
        sample_state["explored_sections"] = []
        # Reduce chunks so we don't hit the quick path
        sample_state["retrieved_results"] = [
            RetrievalResult(
                chunks=sample_state["retrieved_results"][0].chunks[:2],
                strategy_used=RetrievalStrategy.VECTOR_SEARCH,
            ),
        ]

        with patch("src.graph.nodes.evaluate.get_llm_response", side_effect=RuntimeError("mock")):
            result = evaluate_retrieval(sample_state)

        # Heuristic fallback: should suggest exploring undiscovered sections
        assert "Section 45.200" in result["discovered_sections"]

    def test_sufficient_llm_response_clears_sections(self, sample_state):
        from src.graph.nodes.evaluate import evaluate_retrieval

        sample_state["discovered_sections"] = ["Section 45.200"]
        sample_state["explored_sections"] = []
        sample_state["retrieved_results"] = [
            RetrievalResult(
                chunks=sample_state["retrieved_results"][0].chunks[:2],
                strategy_used=RetrievalStrategy.VECTOR_SEARCH,
            ),
        ]

        llm_response = '{"sufficient": true, "explore_sections": [], "reasoning": "enough context"}'
        with patch("src.graph.nodes.evaluate.get_llm_response", return_value=llm_response):
            result = evaluate_retrieval(sample_state)

        assert result["discovered_sections"] == []


class TestSynthesizeSetsConfidence:
    """Verify synthesize node assigns confidence levels."""

    def test_high_confidence_with_many_chunks(self, sample_state, mock_llm):
        from src.graph.nodes.synthesize import synthesize_answer

        # 3 chunks from first result, add another result with 3 more
        extra = RetrievalResult(
            chunks=sample_state["retrieved_results"][0].chunks[:3],
            strategy_used=RetrievalStrategy.GRAPH_QUERY,
        )
        sample_state["retrieved_results"].append(extra)

        result = synthesize_answer(sample_state)

        assert result["confidence"] == Confidence.HIGH

    def test_medium_confidence_with_few_chunks(self, sample_state, mock_llm):
        from src.graph.nodes.synthesize import synthesize_answer

        # Keep only 2 chunks
        sample_state["retrieved_results"] = [
            RetrievalResult(
                chunks=sample_state["retrieved_results"][0].chunks[:2],
                strategy_used=RetrievalStrategy.VECTOR_SEARCH,
            ),
        ]

        result = synthesize_answer(sample_state)

        assert result["confidence"] == Confidence.MEDIUM

    def test_low_confidence_with_one_chunk(self, sample_state, mock_llm):
        from src.graph.nodes.synthesize import synthesize_answer

        sample_state["retrieved_results"] = [
            RetrievalResult(
                chunks=sample_state["retrieved_results"][0].chunks[:1],
                strategy_used=RetrievalStrategy.VECTOR_SEARCH,
            ),
        ]

        result = synthesize_answer(sample_state)

        assert result["confidence"] == Confidence.LOW

    def test_sets_synthesis_text(self, sample_state, mock_llm):
        from src.graph.nodes.synthesize import synthesize_answer

        result = synthesize_answer(sample_state)

        assert isinstance(result["synthesis"], str)
        assert len(result["synthesis"]) > 0
