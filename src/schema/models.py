from __future__ import annotations

from pydantic import BaseModel, Field

from src.schema.enums import Confidence, RetrievalStrategy


class Chunk(BaseModel):
    id: str
    content: str
    section_ref: str
    metadata: dict[str, str] = Field(default_factory=dict)
    score: float = 0.0


class CrossReference(BaseModel):
    source_section: str
    target_section: str
    reference_text: str
    resolved: bool = False


class Citation(BaseModel):
    section_ref: str
    quote: str
    context: str
    confidence: Confidence


class SubQuestion(BaseModel):
    question: str
    suggested_strategy: RetrievalStrategy
    answer: str | None = None


class QueryPlan(BaseModel):
    query_type: str
    primary_strategy: RetrievalStrategy
    secondary_strategies: list[RetrievalStrategy] = Field(default_factory=list)
    expected_cross_refs: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunks: list[Chunk] = Field(default_factory=list)
    cross_references: list[CrossReference] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    strategy_used: RetrievalStrategy
