"""Build and compile the LangGraph agent graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.edges import route_after_retrieval
from src.graph.nodes import (
    execute_retrieval,
    format_response,
    parse_query,
    plan_retrieval,
    resolve_cross_references,
    synthesize_answer,
)
from src.schema.state import AgentState


def build_graph() -> CompiledStateGraph:
    """Construct the agentic RAG state graph and compile it."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("parse", parse_query)
    builder.add_node("plan", plan_retrieval)
    builder.add_node("retrieve", execute_retrieval)
    builder.add_node("resolve", resolve_cross_references)
    builder.add_node("synthesize", synthesize_answer)
    builder.add_node("respond", format_response)

    # Linear edges
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "plan")
    builder.add_edge("plan", "retrieve")

    # After retrieval, route conditionally
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"resolve": "resolve", "synthesize": "synthesize", "retrieve": "retrieve"},
    )

    # After resolving cross-refs, route again (may loop back)
    builder.add_conditional_edges(
        "resolve",
        route_after_retrieval,
        {"resolve": "resolve", "synthesize": "synthesize", "retrieve": "retrieve"},
    )

    # Final synthesis -> response -> end
    builder.add_edge("synthesize", "respond")
    builder.add_edge("respond", END)

    return builder.compile()
