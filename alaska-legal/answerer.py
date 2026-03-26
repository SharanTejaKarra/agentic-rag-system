"""
answerer.py

Accepts a SectionChunk and the original user query, constructs a prompt,
calls the Groq LLM, and returns the answer as a plain string.

No PDF reading, no ChromaDB, no embeddings, no SectionChunk construction.
Uses the Groq SDK directly — no LangChain needed for this prompt structure.
"""

import logging
import time

from groq import Groq, RateLimitError

from chunker import SectionChunk

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1
MAX_TOKENS = 1024
MAX_RETRIES = 3
RETRY_BASE_WAIT = 2  # seconds; doubles each attempt


def _system_prompt(section_id: str) -> str:
    """
    Return the system prompt enforcing grounded, citation-anchored answers.

    Args:
        section_id: Normalised section ID, e.g. "26.080".

    Returns:
        System prompt string.
    """
    return f"""You are a precise legal research assistant. You answer questions strictly using the regulatory text provided to you.

Rules you must follow without exception:
1. Answer only using the text of section 3 AAC {section_id} provided below. Do not infer, extrapolate, or draw on any external knowledge.
2. Always include the citation "3 AAC {section_id}" in your response.
3. Quote the specific subsection or clause that supports your answer wherever applicable.
4. If the answer to the question is not present in the provided section text, respond with exactly: "The provided text of section 3 AAC {section_id} does not address this question."
5. If the section text contains cross-references to other sections, you may mention them as they appear in the text, but do not reason about or interpret those other sections."""


def _user_prompt(chunk: SectionChunk, query: str) -> str:
    """
    Return the user message containing the section text and the question.

    Args:
        chunk: The SectionChunk whose text will be used as context.
        query: The original user query string.

    Returns:
        User prompt string.
    """
    return f"""Section 3 AAC {chunk.section_id} — {chunk.title}

{chunk.text}

Question: {query}"""


def _call_with_retry(client: Groq, messages: list[dict]) -> str:
    """
    Call the Groq API with exponential backoff on rate limit errors.

    Retries up to MAX_RETRIES times. Raises RateLimitError after all attempts
    are exhausted.

    Args:
        client:   Initialised Groq client.
        messages: List of message dicts (system + user).

    Returns:
        The model's response content as a string.

    Raises:
        RateLimitError after MAX_RETRIES failed attempts.
    """
    wait = RETRY_BASE_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == MAX_RETRIES:
                logger.error("Rate limit exceeded after %d attempts.", MAX_RETRIES)
                raise
            logger.warning("Rate limit hit. Retrying in %ds (attempt %d/%d).", wait, attempt, MAX_RETRIES)
            time.sleep(wait)
            wait *= 2


def answer(chunk: SectionChunk, query: str) -> str:
    """
    Generate an answer to the user query grounded in the provided SectionChunk.

    If the section is repealed, returns a repeal notice without calling the LLM.

    Args:
        chunk: A SectionChunk object containing the section text and metadata.
        query: The original user query string.

    Returns:
        A plain string answer.
    """
    if chunk.status == "repealed":
        return (
            f"Section 3 AAC {chunk.section_id} ({chunk.title}) has been repealed. "
            "No substantive regulatory content is available for this section."
        )

    messages = [
        {"role": "system", "content": _system_prompt(chunk.section_id)},
        {"role": "user",   "content": _user_prompt(chunk, query)},
    ]

    client = Groq()  # reads GROQ_API_KEY from environment
    logger.info("Calling LLM for section '%s'.", chunk.section_id)
    return _call_with_retry(client, messages)
