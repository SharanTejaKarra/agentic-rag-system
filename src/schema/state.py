from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.schema.enums import Confidence, QueryType
from src.schema.models import Citation, CrossReference, QueryPlan, RetrievalResult


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    original_query: str
    query_type: QueryType | None
    retrieval_plan: QueryPlan | None
    retrieved_results: Annotated[list[RetrievalResult], add]
    pending_cross_refs: list[CrossReference]
    resolved_cross_refs: Annotated[list[CrossReference], add]
    synthesis: str
    citations: list[Citation]
    confidence: Confidence | None
    iteration_count: int
    max_iterations: int
