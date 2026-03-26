"""Propositional search - find structured propositions by fact type and subject."""

import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.retrieval.qdrant_client import QdrantManager
from src.schema.models import Chunk

logger = logging.getLogger(__name__)

_encoder: SentenceTransformer | None = None
_qdrant: QdrantManager | None = None

# Keywords used to auto-detect fact type from a query string.
_FACT_TYPE_KEYWORDS = {
    "rule": ["rule", "requirement", "must", "shall", "obligat"],
    "exception": ["exception", "exempt", "unless", "except", "waiver"],
    "penalty": ["penalty", "fine", "sanction", "punish", "violation"],
    "condition": ["condition", "if", "when", "provided that", "prerequisite"],
    "definition": ["definition", "means", "defined as", "refers to"],
}


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(settings.embedding_model)
    return _encoder


def _get_qdrant() -> QdrantManager:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantManager()
    return _qdrant


def propositional_search(
    fact_type_or_query: str,
    subject: str | None = None,
    filters: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Search for structured propositions (rules, exceptions, penalties, etc.).

    Can be called two ways:
      - propositional_search("rule", "overtime pay")  (explicit fact type + subject)
      - propositional_search("What is the penalty for late filing?")  (auto-detect)

    Args:
        fact_type_or_query: Either a fact type ("rule", "exception", ...) or a
            natural-language query to auto-detect from.
        subject: The subject to search for. If None, fact_type_or_query is
            treated as a full query and the fact type is inferred.
        filters: Optional additional metadata filters.

    Returns:
        Chunks tagged with their propositional type.
    """
    if subject is not None:
        # Explicit call: first arg is fact_type
        fact_type = fact_type_or_query
    else:
        # Called from graph node with a single query string
        fact_type = _infer_fact_type(fact_type_or_query)
        subject = fact_type_or_query

    encoder = _get_encoder()
    qdrant = _get_qdrant()

    query_vector = encoder.encode(f"{fact_type}: {subject}").tolist()

    payload_filters: dict[str, Any] = {"fact_type": fact_type}
    if filters:
        if "section" in filters:
            payload_filters["section_ref"] = filters["section"]
        if "date_range" in filters:
            payload_filters["date"] = filters["date_range"]

    hits = qdrant.search(
        collection=settings.qdrant_collection_name,
        query_vector=query_vector,
        filters=payload_filters,
        limit=settings.retrieval_top_k,
    )

    chunks = []
    for hit in hits:
        payload = hit.get("payload", {})
        chunks.append(
            Chunk(
                id=str(hit["id"]),
                content=payload.get("content", ""),
                section_ref=payload.get("section_ref", ""),
                metadata={
                    "fact_type": fact_type,
                    **{k: str(v) for k, v in payload.items()
                       if k not in ("content", "section_ref")},
                },
                score=hit.get("score", 0.0),
            )
        )
    return chunks


def _infer_fact_type(query: str) -> str:
    """Guess the fact type from a query string using keyword matching."""
    lower = query.lower()
    for fact_type, keywords in _FACT_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return fact_type
    return "rule"
