"""Metadata extraction: hierarchy, references, effective dates."""

import re
from datetime import datetime

from src.utils.logging import get_logger
from src.utils.references import extract_section_refs

logger = get_logger(__name__)

# Section header patterns ordered by hierarchy depth
_HEADER_PATTERNS = [
    (r"(?m)^(?:ARTICLE|Article)\s+(\S+)", "article"),
    (r"(?m)^(?:CHAPTER|Chapter)\s+(\S+)", "chapter"),
    (r"(?m)^(?:SECTION|Section)\s+(\S+)", "section"),
    (r"(?m)^(\d+\.\d+(?:\.\d+)*)\s", "section"),
    (r"(?m)^\s*\(([a-z])\)\s", "subsection"),
    (r"(?m)^\s*\((\d+)\)\s", "paragraph"),
]

# Date patterns
_DATE_PATTERN = re.compile(
    r"(?:effective|enacted|amended|repealed|expires?|sunset)\s+"
    r"(?:on\s+|as of\s+|date[:\s]+)?"
    r"(\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def extract_hierarchy(text: str) -> dict:
    """Parse section headers to build a hierarchy tree.

    Returns a nested dict: {label, level, children: [...]}
    """
    entries: list[dict] = []
    levels = {"article": 0, "chapter": 1, "section": 2, "subsection": 3, "paragraph": 4}

    for pattern, level_name in _HEADER_PATTERNS:
        for match in re.finditer(pattern, text):
            entries.append({
                "label": match.group(0).strip(),
                "identifier": match.group(1),
                "level": levels[level_name],
                "level_name": level_name,
                "offset": match.start(),
            })

    # Sort by document position
    entries.sort(key=lambda e: e["offset"])

    # Build tree
    root: dict = {"label": "root", "level": -1, "children": []}
    stack: list[dict] = [root]

    for entry in entries:
        node = {"label": entry["label"], "level": entry["level"], "level_name": entry["level_name"], "children": []}
        # Pop stack until we find a parent with a lower level
        while stack and stack[-1]["level"] >= entry["level"]:
            stack.pop()
        if not stack:
            stack = [root]
        stack[-1].setdefault("children", []).append(node)
        stack.append(node)

    return root


def extract_references(text: str) -> list[str]:
    """Find all section references in text (e.g., '31.020(a)(1)')."""
    return extract_section_refs(text)


def extract_effective_dates(text: str) -> list[dict]:
    """Find date mentions and associate them with their context.

    Returns list of dicts with "date_text", "context_type", "surrounding_text".
    """
    results: list[dict] = []
    for match in _DATE_PATTERN.finditer(text):
        full_match = match.group(0)
        date_str = match.group(1)

        # Determine context type from the keyword
        lower = full_match.lower()
        if "effective" in lower:
            ctx = "effective"
        elif "enacted" in lower:
            ctx = "enacted"
        elif "amended" in lower:
            ctx = "amended"
        elif "repeal" in lower:
            ctx = "repealed"
        elif "expir" in lower or "sunset" in lower:
            ctx = "expiration"
        else:
            ctx = "mentioned"

        # Grab surrounding text for context
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        surrounding = text[start:end].strip()

        results.append({
            "date_text": date_str,
            "context_type": ctx,
            "surrounding_text": surrounding,
        })

    return results
