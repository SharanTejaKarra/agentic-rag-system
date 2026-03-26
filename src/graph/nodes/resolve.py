"""Resolve cross-references found in retrieved chunks."""

from __future__ import annotations

from src.schema.enums import RetrievalStrategy
from src.schema.models import CrossReference, RetrievalResult
from src.schema.state import AgentState
from src.tools.cross_reference import cross_reference_search
from src.utils.references import extract_section_refs


def resolve_cross_references(state: AgentState) -> dict:
    """Scan chunks for section references, resolve them, and add to results."""
    pending = list(state.get("pending_cross_refs") or [])
    resolved: list[CrossReference] = []
    new_results: list[RetrievalResult] = []

    for ref in pending:
        if ref.resolved:
            continue
        chunks = cross_reference_search(ref.target_section)
        if chunks:
            ref_resolved = ref.model_copy(update={"resolved": True})
            resolved.append(ref_resolved)
            new_results.append(
                RetrievalResult(
                    chunks=chunks,
                    cross_references=[ref_resolved],
                    strategy_used=RetrievalStrategy.CROSS_REFERENCE,
                )
            )

    # Scan all existing chunks for new cross-references
    new_pending: list[CrossReference] = []
    all_results = list(state.get("retrieved_results") or []) + new_results
    resolved_targets = {r.target_section for r in resolved}

    for result in all_results:
        for chunk in result.chunks:
            target_refs = extract_section_refs(chunk.content)
            for target_sec in target_refs:
                if target_sec not in resolved_targets:
                    new_pending.append(
                        CrossReference(
                            source_section=chunk.section_ref,
                            target_section=target_sec,
                            reference_text=chunk.content[:200],
                        )
                    )

    iteration = (state.get("iteration_count") or 0) + 1

    return {
        "retrieved_results": new_results,
        "pending_cross_refs": new_pending,
        "resolved_cross_refs": resolved,
        "iteration_count": iteration,
    }
