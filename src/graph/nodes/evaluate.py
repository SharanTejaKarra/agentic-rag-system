"""Evaluate whether enough context has been retrieved.

This is the brain of the agentic loop. After retrieval (and optionally
after resolve), the LLM looks at what we have so far and decides:
  - "I have enough to answer" -> go to synthesize
  - "I should explore these sections via graph/hierarchy" -> go to resolve
  - "I need to search for something else" -> go to retrieve

This is what separates a true agent from a static RAG pipeline.
"""

from __future__ import annotations

import json

from config.prompts import EVALUATE_PROMPT
from src.llm.client import get_llm_response
from src.schema.state import AgentState
from src.utils.logging import get_logger
from src.utils.references import extract_section_refs

logger = get_logger(__name__)


def _format_context(state: AgentState) -> str:
    """Flatten retrieved chunks into a summary for the evaluator."""
    lines: list[str] = []
    idx = 0
    for result in state.get("retrieved_results") or []:
        for chunk in result.chunks:
            idx += 1
            source = chunk.section_ref or chunk.id
            # Truncate each chunk so we don't blow up the context
            text = chunk.content[:500]
            lines.append(f"[{idx}] (source: {source})\n{text}\n")
    return "\n".join(lines) if lines else "(no context retrieved yet)"


def _get_undiscovered(state: AgentState) -> list[str]:
    """Sections discovered in retrieval that haven't been explored yet."""
    discovered = set(state.get("discovered_sections") or [])
    explored = set(state.get("explored_sections") or [])
    return sorted(discovered - explored)


def evaluate_retrieval(state: AgentState) -> dict:
    """Decide whether to synthesize, explore sections, or retrieve more.

    Sets discovered_sections based on what the LLM thinks is worth
    exploring. The edge function reads this to route accordingly.
    """
    query = state["original_query"]
    iteration = state.get("iteration_count") or 0
    max_iter = state.get("max_iterations") or 3

    # If we have hit the iteration limit, just go to synthesis
    if iteration >= max_iter:
        logger.info("Hit max iterations (%d), proceeding to synthesis", max_iter)
        return {"discovered_sections": [], "pending_cross_refs": []}

    context = _format_context(state)
    undiscovered = _get_undiscovered(state)

    # Quick path: if there's no context at all, don't bother with LLM eval
    total_chunks = sum(
        len(r.chunks) for r in (state.get("retrieved_results") or [])
    )
    if total_chunks == 0:
        logger.info("No chunks retrieved, skipping evaluation")
        return {"discovered_sections": [], "pending_cross_refs": []}

    # Quick path: if no undiscovered sections and we have decent context, synthesize
    if not undiscovered and total_chunks >= 3:
        logger.info("Enough context (%d chunks) and no undiscovered sections", total_chunks)
        return {"discovered_sections": [], "pending_cross_refs": []}

    # Ask the LLM whether we need more context
    try:
        prompt = EVALUATE_PROMPT.format(
            query=query,
            context=context,
            undiscovered=", ".join(undiscovered) if undiscovered else "(none)",
        )
        raw = get_llm_response(prompt)
        parsed = json.loads(raw)

        sufficient = parsed.get("sufficient", True)
        explore = parsed.get("explore_sections", [])
        reasoning = parsed.get("reasoning", "")

        logger.info(
            "Evaluator: sufficient=%s, explore=%s, reasoning=%s",
            sufficient, explore, reasoning,
        )

        if sufficient and not explore:
            # LLM says we have enough
            return {"discovered_sections": [], "pending_cross_refs": []}

        # LLM wants to explore specific sections
        already_explored = set(state.get("explored_sections") or [])
        new_to_explore = [s for s in explore if s not in already_explored]

        if new_to_explore:
            return {"discovered_sections": new_to_explore}

        # LLM suggested sections but we already explored them all
        return {"discovered_sections": [], "pending_cross_refs": []}

    except Exception:
        logger.warning("LLM evaluation failed, checking heuristics")
        # Fallback: if there are undiscovered sections, explore them
        if undiscovered:
            return {"discovered_sections": undiscovered[:5]}
        return {"discovered_sections": [], "pending_cross_refs": []}
