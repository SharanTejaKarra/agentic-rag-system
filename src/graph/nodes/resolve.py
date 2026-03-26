"""Explore discovered sections and resolve cross-references.

This node does two things:
1. Takes sections discovered by vector search (or flagged by the evaluator)
   and explores them via the knowledge graph and hierarchy tools.
2. Resolves any pending explicit cross-references.

This is what makes the RAG system agentic: vector search finds text that
mentions "Section 105", this node goes to the knowledge graph to pull
Section 105's content, relationships, and surrounding hierarchy.
"""

from __future__ import annotations

from src.schema.enums import RetrievalStrategy
from src.schema.models import CrossReference, RetrievalResult
from src.schema.state import AgentState
from src.tools.cross_reference import cross_reference_search
from src.tools.graph_query import graph_query
from src.tools.hierarchical_lookup import hierarchical_lookup
from src.utils.logging import get_logger
from src.utils.references import extract_section_refs

logger = get_logger(__name__)


def resolve_cross_references(state: AgentState) -> dict:
    """Explore discovered sections and resolve cross-references."""
    new_results: list[RetrievalResult] = []
    resolved: list[CrossReference] = []
    newly_explored: list[str] = []
    already_explored = set(state.get("explored_sections") or [])

    # --- Phase 1: Explore discovered sections via graph + hierarchy ---
    discovered = state.get("discovered_sections") or []
    for section in discovered:
        if section in already_explored:
            continue

        logger.info("Exploring discovered section: %s", section)
        newly_explored.append(section)

        # Try graph query for this section (gets relationships, penalties, etc.)
        graph_chunks = graph_query(section)
        if graph_chunks:
            new_results.append(
                RetrievalResult(
                    chunks=graph_chunks,
                    strategy_used=RetrievalStrategy.GRAPH_QUERY,
                )
            )

        # Try hierarchy lookup (gets parent context and siblings)
        hierarchy_chunks = hierarchical_lookup(section, direction="children")
        parent_chunks = hierarchical_lookup(section, direction="parent")
        all_hierarchy = hierarchy_chunks + parent_chunks
        if all_hierarchy:
            new_results.append(
                RetrievalResult(
                    chunks=all_hierarchy,
                    strategy_used=RetrievalStrategy.HIERARCHICAL,
                )
            )

    # --- Phase 2: Resolve pending cross-references ---
    pending = list(state.get("pending_cross_refs") or [])
    for ref in pending:
        if ref.resolved:
            continue
        if ref.target_section in already_explored:
            ref_resolved = ref.model_copy(update={"resolved": True})
            resolved.append(ref_resolved)
            continue

        chunks = cross_reference_search(ref.target_section)
        if chunks:
            ref_resolved = ref.model_copy(update={"resolved": True})
            resolved.append(ref_resolved)
            newly_explored.append(ref.target_section)
            new_results.append(
                RetrievalResult(
                    chunks=chunks,
                    cross_references=[ref_resolved],
                    strategy_used=RetrievalStrategy.CROSS_REFERENCE,
                )
            )

    # --- Phase 3: Scan new chunks for further cross-references ---
    new_pending: list[CrossReference] = []
    all_explored = already_explored | set(newly_explored)

    for result in new_results:
        for chunk in result.chunks:
            target_refs = extract_section_refs(chunk.content)
            for target_sec in target_refs:
                if target_sec not in all_explored:
                    new_pending.append(
                        CrossReference(
                            source_section=chunk.section_ref,
                            target_section=target_sec,
                            reference_text=chunk.content[:200],
                        )
                    )

    iteration = (state.get("iteration_count") or 0) + 1

    logger.info(
        "Resolve: explored %d sections, resolved %d cross-refs, "
        "found %d new cross-refs, iteration %d",
        len(newly_explored), len(resolved), len(new_pending), iteration,
    )

    return {
        "retrieved_results": new_results,
        "explored_sections": newly_explored,
        "discovered_sections": [],  # clear after exploring
        "pending_cross_refs": new_pending,
        "resolved_cross_refs": resolved,
        "iteration_count": iteration,
    }
