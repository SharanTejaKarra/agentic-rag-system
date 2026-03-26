"""Plan retrieval strategy using LLM reasoning, not a static table.

The planner looks at the actual query content and decides which tools
to use. For general questions (no specific section numbers), it always
starts with vector_search so the system can discover relevant sections
before trying graph lookups. This is what makes the system agentic
rather than a rigid pipeline.
"""

from __future__ import annotations

import json

from config.prompts import PLANNING_PROMPT
from src.llm.client import get_llm_response
from src.schema.enums import QueryType, RetrievalStrategy
from src.schema.models import QueryPlan
from src.schema.state import AgentState
from src.utils.logging import get_logger
from src.utils.references import extract_section_refs

logger = get_logger(__name__)

# Fallback table in case the LLM call fails
_FALLBACK_TABLE: dict[QueryType, tuple[RetrievalStrategy, list[RetrievalStrategy]]] = {
    QueryType.DEFINITIONAL: (
        RetrievalStrategy.VECTOR_SEARCH,
        [RetrievalStrategy.GRAPH_QUERY, RetrievalStrategy.HIERARCHICAL],
    ),
    QueryType.PROCEDURAL: (
        RetrievalStrategy.VECTOR_SEARCH,
        [RetrievalStrategy.GRAPH_QUERY],
    ),
    QueryType.STRUCTURAL: (
        RetrievalStrategy.VECTOR_SEARCH,
        [RetrievalStrategy.GRAPH_QUERY],
    ),
    QueryType.COMPLIANCE: (
        RetrievalStrategy.VECTOR_SEARCH,
        [RetrievalStrategy.PROPOSITIONAL, RetrievalStrategy.GRAPH_QUERY],
    ),
    QueryType.TEMPORAL: (
        RetrievalStrategy.VECTOR_SEARCH,
        [RetrievalStrategy.HIERARCHICAL],
    ),
}

_STRATEGY_NAMES = {s.value: s for s in RetrievalStrategy}


def plan_retrieval(state: AgentState) -> dict:
    """Use the LLM to decide which tools to call and in what order.

    Falls back to a sensible default (vector_search first) if the LLM
    call fails or returns garbage.
    """
    query = state["original_query"]
    query_type = state.get("query_type") or QueryType.DEFINITIONAL

    # Check if the user mentioned specific section numbers
    mentioned_sections = extract_section_refs(query)

    # Try LLM-based planning
    try:
        prompt = PLANNING_PROMPT.format(
            query=query,
            query_type=query_type.value,
        )
        raw = get_llm_response(prompt)
        parsed = json.loads(raw)

        primary_name = parsed.get("primary_strategy", "vector_search")
        secondary_names = parsed.get("secondary_strategies", [])

        primary = _STRATEGY_NAMES.get(primary_name, RetrievalStrategy.VECTOR_SEARCH)
        secondaries = [
            _STRATEGY_NAMES[s] for s in secondary_names if s in _STRATEGY_NAMES
        ]

        logger.info(
            "LLM planner chose: primary=%s, secondaries=%s, reasoning=%s",
            primary.value,
            [s.value for s in secondaries],
            parsed.get("reasoning", ""),
        )
    except Exception:
        logger.warning("LLM planning failed, using fallback table")
        primary, secondaries = _FALLBACK_TABLE.get(
            query_type,
            (RetrievalStrategy.VECTOR_SEARCH, [RetrievalStrategy.GRAPH_QUERY]),
        )

    # Safety net: if user mentioned specific sections and the planner
    # didn't include graph/hierarchy tools, add them as secondaries
    if mentioned_sections:
        graph_tools = {RetrievalStrategy.GRAPH_QUERY, RetrievalStrategy.HIERARCHICAL}
        has_graph = primary in graph_tools or any(s in graph_tools for s in secondaries)
        if not has_graph:
            secondaries.append(RetrievalStrategy.GRAPH_QUERY)
            secondaries.append(RetrievalStrategy.HIERARCHICAL)

    # Safety net: vector_search should always be available as a fallback
    all_strategies = {primary} | set(secondaries)
    if RetrievalStrategy.VECTOR_SEARCH not in all_strategies:
        secondaries.insert(0, RetrievalStrategy.VECTOR_SEARCH)

    plan = QueryPlan(
        query_type=query_type.value,
        primary_strategy=primary,
        secondary_strategies=secondaries,
    )

    return {"retrieval_plan": plan}
