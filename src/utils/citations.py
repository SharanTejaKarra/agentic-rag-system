"""Citation formatting utilities."""


def format_citation(section_ref: str, quote: str, confidence: str) -> str:
    """Format a single citation in standard format.

    Returns: [Source]: Per Section X, "quote". [Confidence]: level
    """
    return f'[Source]: Per {section_ref}, "{quote}". [Confidence]: {confidence}'


def build_citation_chain(citations: list) -> str:
    """Format a chain of citations showing the multi-hop reasoning path.

    Each citation is a dict or Citation model with section_ref, quote, confidence.
    """
    if not citations:
        return ""

    parts: list[str] = []
    for i, cite in enumerate(citations, 1):
        # Support both dicts and Pydantic models
        if hasattr(cite, "section_ref"):
            ref = cite.section_ref
            quote = cite.quote
            conf = cite.confidence if isinstance(cite.confidence, str) else cite.confidence.value
        else:
            ref = cite["section_ref"]
            quote = cite["quote"]
            conf = cite["confidence"]

        parts.append(f"  Step {i}: {format_citation(ref, quote, conf)}")

    header = f"Citation chain ({len(citations)} hop{'s' if len(citations) != 1 else ''}):"
    return header + "\n" + "\n".join(parts)
