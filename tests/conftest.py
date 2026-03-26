"""Shared pytest fixtures for the agentic RAG test suite."""

from unittest.mock import MagicMock, patch

import pytest

from src.schema.enums import Confidence, QueryType, RetrievalStrategy
from src.schema.models import Chunk, CrossReference, QueryPlan, RetrievalResult
from src.schema.state import AgentState


@pytest.fixture()
def sample_chunks() -> list[Chunk]:
    """A set of test Chunk objects with legal document content."""
    return [
        Chunk(
            id="chunk_001",
            content=(
                "Section 31.020(a)(1) defines a 'qualified beneficiary' as any individual "
                "who has been enrolled in the plan for at least 12 consecutive months."
            ),
            section_ref="Section 31.020",
            metadata={"source": "benefits_act.pdf", "chunk_type": "hierarchical"},
            score=0.92,
        ),
        Chunk(
            id="chunk_002",
            content=(
                "Under Section 12.100(b), employers must provide written notice within "
                "30 calendar days of a qualifying event. See also Section 31.020(a)(1) "
                "for the definition of qualifying beneficiary."
            ),
            section_ref="Section 12.100",
            metadata={"source": "employer_obligations.pdf", "chunk_type": "hierarchical"},
            score=0.87,
        ),
        Chunk(
            id="chunk_003",
            content=(
                "Penalties under Section 45.200 apply when an employer fails to comply "
                "with the notice requirements set forth in Section 12.100(b). Fines range "
                "from $100 to $1,000 per day of non-compliance."
            ),
            section_ref="Section 45.200",
            metadata={"source": "penalties.pdf", "chunk_type": "propositional", "fact_type": "penalty"},
            score=0.81,
        ),
        Chunk(
            id="chunk_004",
            content=(
                "Article 5 establishes the appeals process. Any party aggrieved by a "
                "determination under this chapter may file an appeal within 60 days."
            ),
            section_ref="Article 5",
            metadata={"source": "appeals.pdf", "chunk_type": "hierarchical"},
            score=0.75,
        ),
        Chunk(
            id="chunk_005",
            content=(
                "The effective date of amendments to Chapter 3 is January 1, 2025, "
                "as enacted by Public Law 118-42."
            ),
            section_ref="Chapter 3",
            metadata={"source": "amendments.pdf", "chunk_type": "hierarchical"},
            score=0.70,
        ),
    ]


@pytest.fixture()
def sample_state(sample_chunks: list[Chunk]) -> AgentState:
    """A pre-populated AgentState for testing graph nodes."""
    plan = QueryPlan(
        query_type="definitional",
        primary_strategy=RetrievalStrategy.VECTOR_SEARCH,
        secondary_strategies=[RetrievalStrategy.GRAPH_QUERY],
    )
    return {
        "original_query": "What is a qualified beneficiary?",
        "query_type": QueryType.DEFINITIONAL,
        "retrieval_plan": plan,
        "retrieved_results": [
            RetrievalResult(
                chunks=sample_chunks[:3],
                strategy_used=RetrievalStrategy.VECTOR_SEARCH,
            ),
        ],
        "pending_cross_refs": [],
        "resolved_cross_refs": [],
        "synthesis": "",
        "citations": [],
        "confidence": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "messages": [],
    }


@pytest.fixture()
def mock_qdrant(sample_chunks: list[Chunk]):
    """Patch QdrantManager so tests never touch a real Qdrant instance."""
    hits = [
        {
            "id": c.id,
            "score": c.score,
            "payload": {
                "content": c.content,
                "section_ref": c.section_ref,
                **c.metadata,
            },
        }
        for c in sample_chunks
    ]

    mock_manager = MagicMock()
    mock_manager.search.return_value = hits

    with patch(
        "src.retrieval.qdrant_client.QdrantManager", return_value=mock_manager
    ) as patched:
        patched.return_value = mock_manager
        yield mock_manager


@pytest.fixture()
def mock_neo4j():
    """Patch Neo4jManager so tests never touch a real Neo4j instance."""
    mock_manager = MagicMock()

    mock_manager.find_entity.return_value = [
        {
            "n": {
                "id": "sec_31_020",
                "name": "31.020",
                "content": "Defines a qualified beneficiary as any individual enrolled for 12+ months.",
                "section_ref": "Section 31.020",
            }
        }
    ]

    mock_manager.find_relationships.return_value = [
        {
            "n": {
                "id": "sec_31_020",
                "name": "31.020",
                "content": "Qualified beneficiary definition",
            },
            "r": {"type": "REFERENCES"},
            "m": {
                "id": "sec_12_100",
                "name": "12.100",
                "content": "Employer notice requirements",
            },
        }
    ]

    with patch(
        "src.retrieval.neo4j_client.Neo4jManager", return_value=mock_manager
    ) as patched:
        patched.return_value = mock_manager
        yield mock_manager


@pytest.fixture()
def mock_llm():
    """Patch get_llm_response to return canned responses without calling an LLM.

    Patches both the original module and the modules that import the function
    so that local references are also replaced.
    """
    responses = {
        "classify": '{"query_type": "definitional", "key_concepts": ["qualified beneficiary"]}',
        "synthesize": (
            "A qualified beneficiary is defined under Section 31.020(a)(1) as any "
            "individual who has been enrolled in the plan for at least 12 consecutive months."
        ),
        "default": "This is a test response from the mock LLM.",
    }

    def _side_effect(prompt: str, **kwargs) -> str:
        lower = prompt.lower()
        if "classify" in lower:
            return responses["classify"]
        if "synthesize" in lower or "context" in lower:
            return responses["synthesize"]
        return responses["default"]

    with patch("src.llm.client.get_llm_response", side_effect=_side_effect) as mock, \
         patch("src.graph.nodes.parse.get_llm_response", side_effect=_side_effect), \
         patch("src.graph.nodes.synthesize.get_llm_response", side_effect=_side_effect):
        mock.responses = responses
        yield mock
