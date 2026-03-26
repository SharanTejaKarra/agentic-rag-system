"""Execute retrieval and discover sections for further exploration.

After running each tool, this node scans the returned chunks for section
references. Those sections go into discovered_sections so the resolve
node (or the evaluate node) can decide to explore them via graph/hierarchy.

This is the key to making the system agentic: vector search finds text,
text mentions sections, sections get explored via the knowledge graph.
"""

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
from src.utils.logging import get_logger
from src.utils.references import extract_section_refs

logger = get_logger(__name__)

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
    """Run planned retrieval strategies and discover sections in the results."""
    plan = state["retrieval_plan"]
    query = state["original_query"]

    # Track what has already been run across iterations
    already_used: set[RetrievalStrategy] = set()
    for prev_result in (state.get("retrieved_results") or []):
        already_used.add(prev_result.strategy_used)

    results: list[RetrievalResult] = []
    total_chunks = sum(
        len(r.chunks) for r in (state.get("retrieved_results") or [])
    )

    # Run primary strategy if not already done
    if plan.primary_strategy not in already_used:
        tool_fn = _TOOL_MAP.get(plan.primary_strategy)
        if tool_fn is not None:
            primary_result = tool_fn(query)
            results.append(
                RetrievalResult(
                    chunks=primary_result,
                    strategy_used=plan.primary_strategy,
                )
            )
            total_chunks += len(primary_result)
            already_used.add(plan.primary_strategy)

    # If results are sparse, try secondary strategies
    if total_chunks < _SPARSE_THRESHOLD:
        for strategy in plan.secondary_strategies:
            if strategy in already_used:
                continue
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
            total_chunks += len(secondary_result)
            already_used.add(strategy)

    # Scan all new chunks for section references we haven't explored yet
    already_explored = set(state.get("explored_sections") or [])
    newly_discovered: set[str] = set()

    for result in results:
        for chunk in result.chunks:
            refs = extract_section_refs(chunk.content)
            for ref in refs:
                if ref not in already_explored:
                    newly_discovered.add(ref)

    # Also pick up sections from the query itself (if user mentioned them)
    query_refs = extract_section_refs(query)
    for ref in query_refs:
        if ref not in already_explored:
            newly_discovered.add(ref)

    logger.info(
        "Retrieval: %d new chunks, %d sections discovered",
        sum(len(r.chunks) for r in results),
        len(newly_discovered),
    )

    return {
        "retrieved_results": results,
        "discovered_sections": list(newly_discovered),
    }
