"""Utilities for extracting and parsing legal section references."""

import re

# Matches patterns like "31.020(a)(1)", "Section 2.1(b)", "Sec. 12.300(c)(2)(iii)"
SECTION_PATTERN = re.compile(
    r"(?:Section|Sec\.?\s*)"
    r"(\d+(?:\.\d+)*)"
    r"((?:\([a-zA-Z0-9]+\))*)"
    r"|"
    r"(\d+\.\d+)"
    r"((?:\([a-zA-Z0-9]+\))*)",
    re.IGNORECASE,
)


def extract_section_refs(text: str) -> list[str]:
    """Find all section references in a text string."""
    refs: list[str] = []
    for match in SECTION_PATTERN.finditer(text):
        full = match.group(0).strip()
        if full:
            refs.append(full)
    return refs


def parse_section_ref(ref: str) -> dict:
    """Parse '31.020(a)(1)' into structured components.

    Returns dict with keys: section, subsection, paragraph.
    Missing parts are set to empty string.
    """
    # Strip leading "Section" / "Sec." prefix
    cleaned = re.sub(r"^(?:Section|Sec\.?)\s*", "", ref, flags=re.IGNORECASE).strip()

    # Split base number from parenthetical parts
    base_match = re.match(r"([\d.]+)(.*)", cleaned)
    if not base_match:
        return {"section": ref, "subsection": "", "paragraph": ""}

    section = base_match.group(1)
    remainder = base_match.group(2)

    # Extract parenthetical parts: (a), (1), (iii), etc.
    parts = re.findall(r"\(([a-zA-Z0-9]+)\)", remainder)

    return {
        "section": section,
        "subsection": parts[0] if len(parts) > 0 else "",
        "paragraph": parts[1] if len(parts) > 1 else "",
    }
