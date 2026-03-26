"""Reranking - deduplicates and re-scores retrieved chunks."""

import logging
import re

from src.schema.models import Chunk

logger = logging.getLogger(__name__)


def rerank_results(chunks: list[Chunk], query: str, top_k: int = 10) -> list[Chunk]:
    """Deduplicate by chunk ID, re-score, and return the top-k results."""
    if not chunks:
        return []

    # Deduplicate by chunk id, keeping highest-scored version
    seen: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.id not in seen or chunk.score > seen[chunk.id].score:
            seen[chunk.id] = chunk
    unique = list(seen.values())

    query_terms = set(_tokenize(query))

    scored: list[tuple[float, Chunk]] = []
    for chunk in unique:
        final_score = _compute_score(chunk, query_terms)
        scored.append((final_score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for final_score, chunk in scored[:top_k]:
        results.append(chunk.model_copy(update={"score": final_score}))
    return results


def _compute_score(chunk: Chunk, query_terms: set[str]) -> float:
    """Combine relevance signals into a single score."""
    # Original vector similarity score (weight: 60%)
    base = chunk.score * 0.6

    # Keyword overlap with query (weight: 25%)
    chunk_terms = set(_tokenize(chunk.content))
    if query_terms:
        overlap = len(query_terms & chunk_terms) / len(query_terms)
    else:
        overlap = 0.0
    keyword = overlap * 0.25

    # Section reference match bonus (weight: 15%)
    section_bonus = 0.15 if chunk.section_ref and chunk.section_ref.strip() else 0.0

    return base + keyword + section_bonus


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())
