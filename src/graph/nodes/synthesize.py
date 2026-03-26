"""Synthesize a coherent answer from retrieved chunks."""

from __future__ import annotations

from src.llm.client import get_llm_response
from src.schema.enums import Confidence
from src.schema.state import AgentState
from config.prompts import SYNTHESIS_PROMPT


def _format_context(state: AgentState) -> str:
    """Flatten all retrieved chunks into a numbered context block."""
    lines: list[str] = []
    idx = 0
    for result in state.get("retrieved_results") or []:
        for chunk in result.chunks:
            idx += 1
            source = chunk.section_ref or chunk.id
            lines.append(f"[{idx}] (source: {source})\n{chunk.content}\n")
    return "\n".join(lines)


def _assess_confidence(state: AgentState) -> Confidence:
    """Pick a confidence level based on how many chunks we have."""
    total_chunks = sum(
        len(r.chunks) for r in (state.get("retrieved_results") or [])
    )
    if total_chunks >= 5:
        return Confidence.HIGH
    if total_chunks >= 2:
        return Confidence.MEDIUM
    return Confidence.LOW


def synthesize_answer(state: AgentState) -> dict:
    """Use the LLM to synthesize retrieved chunks into a final answer."""
    query = state["original_query"]
    context = _format_context(state)

    prompt = SYNTHESIS_PROMPT.format(question=query, context=context)
    synthesis = get_llm_response(prompt)
    confidence = _assess_confidence(state)

    return {
        "synthesis": synthesis,
        "confidence": confidence,
    }
