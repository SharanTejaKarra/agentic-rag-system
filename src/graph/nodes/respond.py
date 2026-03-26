"""Format the final response with citations and reasoning chain."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.schema.enums import Confidence
from src.schema.models import Citation
from src.schema.state import AgentState


def _build_citations(state: AgentState) -> list[Citation]:
    """Collect citations from all retrieved chunks."""
    citations: list[Citation] = []
    confidence = state.get("confidence") or Confidence.LOW

    for result in state.get("retrieved_results") or []:
        for chunk in result.chunks:
            citations.append(
                Citation(
                    section_ref=chunk.section_ref,
                    quote=chunk.content[:300],
                    context=f"Retrieved via {result.strategy_used.value}",
                    confidence=confidence,
                )
            )
    return citations


def _build_reasoning_chain(state: AgentState) -> str:
    """Summarize the retrieval chain for multi-hop queries."""
    steps: list[str] = []
    for i, result in enumerate(state.get("retrieved_results") or [], start=1):
        n_chunks = len(result.chunks)
        steps.append(f"{i}. {result.strategy_used.value} - {n_chunks} chunk(s)")

    resolved = state.get("resolved_cross_refs") or []
    if resolved:
        steps.append(f"Cross-references resolved: {len(resolved)}")

    return "\n".join(steps)


def format_response(state: AgentState) -> dict:
    """Assemble the final user-facing response."""
    synthesis = state.get("synthesis") or ""
    confidence = state.get("confidence") or Confidence.LOW
    citations = _build_citations(state)
    reasoning = _build_reasoning_chain(state)

    parts: list[str] = [synthesis]

    if confidence == Confidence.LOW:
        parts.append(
            "\n[Note: Confidence is low - the retrieved context may be "
            "insufficient to fully answer this question.]"
        )

    if reasoning:
        parts.append(f"\n---\nRetrieval chain:\n{reasoning}")

    response_text = "\n".join(parts)

    return {
        "citations": citations,
        "messages": [AIMessage(content=response_text)],
    }
