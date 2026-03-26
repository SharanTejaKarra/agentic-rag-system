"""Query expansion and synthetic query generation using Claude."""

import json

from src.llm.client import get_llm_response
from src.utils.logging import get_logger

logger = get_logger(__name__)


def expand_query_synonyms(query: str) -> list[str]:
    """Generate alternative phrasings to handle vocabulary mismatch.

    E.g., "permit denial" -> ["permit rejection", "application denied", "license refusal"]
    """
    prompt = (
        "Generate 3-5 alternative phrasings for the following search query. "
        "Each alternative should express the same concept using different vocabulary. "
        "Return ONLY a JSON list of strings, no explanation.\n\n"
        f"Query: {query}"
    )
    raw = get_llm_response(prompt)

    # Parse the JSON list from the response
    try:
        alternatives = json.loads(raw.strip())
        if isinstance(alternatives, list):
            return [str(a) for a in alternatives]
    except json.JSONDecodeError:
        logger.warning("Failed to parse synonym expansion response as JSON")

    # Fallback: split by newlines and clean up
    lines = [line.strip().strip("-").strip("*").strip() for line in raw.strip().splitlines()]
    return [line for line in lines if line and line != query]


def generate_synthetic_queries(query: str, n: int = 3) -> list[str]:
    """Generate synthetic questions that express the same intent differently.

    Useful for multi-query retrieval to improve recall.
    """
    prompt = (
        f"Generate exactly {n} different questions that express the same intent as the query below. "
        "Each question should approach the topic from a slightly different angle. "
        "Return ONLY a JSON list of strings, no explanation.\n\n"
        f"Query: {query}"
    )
    raw = get_llm_response(prompt)

    try:
        questions = json.loads(raw.strip())
        if isinstance(questions, list):
            return [str(q) for q in questions[:n]]
    except json.JSONDecodeError:
        logger.warning("Failed to parse synthetic queries response as JSON")

    lines = [line.strip().strip("-").strip("*").strip() for line in raw.strip().splitlines()]
    return [line for line in lines if line and line != query][:n]
