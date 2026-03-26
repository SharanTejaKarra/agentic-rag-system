"""Sub-question decomposition - breaks complex queries into simpler parts."""

import json
import logging
import re

import anthropic

from config.settings import settings
from config.prompts import QUERY_DECOMPOSITION_PROMPT
from src.schema.enums import RetrievalStrategy
from src.schema.models import Chunk, SubQuestion
from src.tools.vector_search import vector_search

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None

# Heuristics for picking a retrieval strategy per sub-question.
_STRATEGY_KEYWORDS: list[tuple[list[str], RetrievalStrategy]] = [
    (["define", "meaning", "what is", "what does"], RetrievalStrategy.VECTOR_SEARCH),
    (["relate", "connection", "between", "linked"], RetrievalStrategy.GRAPH_QUERY),
    (["section", "article", "subsection", "clause"], RetrievalStrategy.HIERARCHICAL),
    (["rule", "exception", "penalty", "condition"], RetrievalStrategy.PROPOSITIONAL),
    (["refer", "reference", "see also", "cross"], RetrievalStrategy.CROSS_REFERENCE),
]


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def decompose_query(complex_query: str) -> list[SubQuestion]:
    """Use an LLM to break a complex question into independent sub-questions.

    Each sub-question is tagged with a suggested retrieval strategy based
    on keyword heuristics.
    """
    client = _get_client()

    prompt = QUERY_DECOMPOSITION_PROMPT.format(question=complex_query)

    try:
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            temperature=settings.anthropic_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
    except Exception:
        logger.exception("LLM call failed during query decomposition")
        # Fall back to returning the original query as a single sub-question
        return [
            SubQuestion(
                question=complex_query,
                suggested_strategy=RetrievalStrategy.VECTOR_SEARCH,
            )
        ]

    sub_questions = _parse_response(raw_text)
    if not sub_questions:
        return [
            SubQuestion(
                question=complex_query,
                suggested_strategy=RetrievalStrategy.VECTOR_SEARCH,
            )
        ]
    return sub_questions


def _parse_response(raw_text: str) -> list[SubQuestion]:
    """Parse the LLM response (expected JSON list of strings) into SubQuestions."""
    # Try to extract a JSON array from the response
    try:
        questions = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to find a JSON array inside markdown fences
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            try:
                questions = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(questions, list):
        return []

    results = []
    for q in questions:
        text = str(q).strip()
        if text:
            strategy = _infer_strategy(text)
            results.append(SubQuestion(question=text, suggested_strategy=strategy))
    return results


def _infer_strategy(question: str) -> RetrievalStrategy:
    """Pick a retrieval strategy based on keyword matching."""
    lower = question.lower()
    for keywords, strategy in _STRATEGY_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return strategy
    return RetrievalStrategy.VECTOR_SEARCH


def sub_question_search(query: str) -> list[Chunk]:
    """Decompose a query and run vector search on each sub-question.

    Intended for use as a graph-node tool - accepts a single query string
    and returns aggregated Chunk results.
    """
    sub_questions = decompose_query(query)
    all_chunks: list[Chunk] = []
    for sq in sub_questions:
        results = vector_search(sq.question)
        all_chunks.extend(results)
    return all_chunks
