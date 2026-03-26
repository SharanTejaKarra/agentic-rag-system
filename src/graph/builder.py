"""Build and compile the LangGraph agent graph.

The graph structure:

  START -> parse -> plan -> retrieve -> evaluate -+-> synthesize -> respond -> END
                                                  |
                                                  +-> resolve -> evaluate (loops)

The evaluate node is the decision point. It uses the LLM to decide
whether to explore discovered sections (via resolve) or proceed to
synthesis. This creates a true agentic loop where vector search
discovers sections, the graph/hierarchy tools explore them, and the
LLM decides when enough context has been gathered.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.edges import route_after_evaluate, route_after_resolve
from src.graph.nodes import (
    execute_retrieval,
    format_response,
    parse_query,
    plan_retrieval,
    resolve_cross_references,
    synthesize_answer,
)
from src.graph.nodes.evaluate import evaluate_retrieval
from src.schema.state import AgentState


def build_graph() -> CompiledStateGraph:
    """Construct the agentic RAG state graph and compile it."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("parse", parse_query)
    builder.add_node("plan", plan_retrieval)
    builder.add_node("retrieve", execute_retrieval)
    builder.add_node("evaluate", evaluate_retrieval)
    builder.add_node("resolve", resolve_cross_references)
    builder.add_node("synthesize", synthesize_answer)
    builder.add_node("respond", format_response)

    # Linear: parse -> plan -> retrieve -> evaluate
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "evaluate")

    # After evaluate: either explore sections or synthesize
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"resolve": "resolve", "synthesize": "synthesize"},
    )

    # After resolve: go back to evaluate (the LLM decides if more is needed)
    builder.add_conditional_edges(
        "resolve",
        route_after_resolve,
        {"evaluate": "evaluate", "synthesize": "synthesize"},
    )

    # Final: synthesize -> respond -> end
    builder.add_edge("synthesize", "respond")
    builder.add_edge("respond", END)

    return builder.compile()
