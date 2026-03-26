"""Parse the user query to classify type and extract intent."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from src.llm.client import get_llm_response
from src.schema.enums import QueryType
from src.schema.state import AgentState

_CLASSIFY_PROMPT = """\
Classify the following legal question into exactly one of these categories:
- definitional: asks "What is X?"
- procedural: asks "What steps to do X?" or "How do I do X?"
- structural: asks about relationships between entities, e.g. "How do A and B relate?"
- compliance: asks whether something violates a rule, e.g. "Am I violating X?"
- temporal: asks about timing, deadlines, or effective dates

Also extract a short list of key legal concepts or entities mentioned.

Return JSON with keys "query_type" (one of the categories above) and "key_concepts" (list of strings).

Question: {query}
"""


def parse_query(state: AgentState) -> dict:
    """Classify the query type and extract key concepts using the LLM."""
    query = state["original_query"]

    prompt = _CLASSIFY_PROMPT.format(query=query)
    raw = get_llm_response(prompt)

    try:
        parsed = json.loads(raw)
        query_type = QueryType(parsed["query_type"])
    except (json.JSONDecodeError, KeyError, ValueError):
        # Default to definitional if classification fails
        query_type = QueryType.DEFINITIONAL

    return {
        "query_type": query_type,
        "messages": [HumanMessage(content=query)],
    }
