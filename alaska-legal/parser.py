"""
parser.py

Extracts raw text from an Alaska Administrative Code PDF and identifies
section boundaries. Returns a list of raw section dictionaries.

No knowledge of ChromaDB, embeddings, LLM, or SectionChunk.
"""

import re
import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

# A genuine section header starts with "3 AAC XX.XXX." followed by a capital letter.
# Cross-references appear mid-sentence and won't have a title on the same line.
SECTION_HEADER_RE = re.compile(r"^3 AAC \d+\.\d+\.\s+[A-Z]")

# Matches "Repealed" optionally followed by a date (e.g. "Repealed 1/1/2000")
REPEALED_RE = re.compile(r"^Repealed[\s\S]*$", re.IGNORECASE)

# Standalone "APPENDIX" — not embedded in a sentence (no lowercase words adjacent)
APPENDIX_RE = re.compile(r"(?<![a-z])APPENDIX(?![a-z])")


def _is_header(line: str) -> bool:
    """Return True if the stripped line matches the section header pattern."""
    return bool(SECTION_HEADER_RE.match(line.strip()))


def _is_continuation(line: str) -> bool:
    """
    Return True if the line could be a title continuation.

    Used only as a safety guard — the primary stop signal is whether the
    accumulated header already ends with a period. This function just prevents
    consuming blank lines or the next section header as continuation.
    Lowercase is allowed: titles like '3 AAC 31.020. Insurance producer, managing
    general agent, ...' have continuation lines that start with lowercase.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if _is_header(stripped):
        return False
    # Stop at lines that are clearly body text: numbered/lettered subsections
    # like "(a)", "(1)", or body sentences starting mid-paragraph.
    if stripped[0].isdigit() or stripped.startswith("("):
        return False
    return True


def _is_repealed(body_lines: list[str]) -> bool:
    """
    Return True if the section is repealed.

    Only checks the first non-empty body line. Repealed sections in these PDFs
    have 'Repealed' or 'Repealed <date>.' as the first content line, followed
    by noise (article headers, page footers) before the next section header.
    Joining all lines and using fullmatch incorrectly marks these as active.
    """
    non_empty = [l.strip() for l in body_lines if l.strip()]
    if not non_empty:
        return False
    # Match "Repealed" optionally followed by a date and/or period
    return bool(re.match(r"Repealed[\s\d/,\.]*$", non_empty[0], re.IGNORECASE))


def _has_appendix(body_lines: list[str]) -> bool:
    """Return True if any body line contains APPENDIX as a standalone word."""
    for line in body_lines:
        if APPENDIX_RE.search(line):
            return True
    return False


def _extract_lines(pdf_path: Path) -> list[str]:
    """Extract all text lines from a PDF, page by page."""
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.splitlines())
    return lines


def parse_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Parse a single AAC PDF into a list of raw section dictionaries.

    Args:
        pdf_path: Path to the PDF file (str or Path).

    Returns:
        List of dicts, each with keys:
            "header_line"  — joined header string
            "body_lines"   — list of raw body text lines
            "is_repealed"  — bool
            "has_appendix" — bool
    """
    pdf_path = Path(pdf_path)
    lines = _extract_lines(pdf_path)

    sections = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if not _is_header(line):
            i += 1
            continue

        # --- Build header (may span multiple lines) ---
        header_parts = [line.strip()]
        i += 1

        # Stop joining once the accumulated header ends with a period —
        # that signals the title is complete. Only join continuation lines
        # for genuinely split titles like "3 AAC 26.080. Additional standards\nfor prompt..."
        while i < len(lines) and not header_parts[-1].endswith(".") and _is_continuation(lines[i]):
            header_parts.append(lines[i].strip())
            i += 1

        header_line = " ".join(header_parts)

        # --- Collect body lines until the next section header ---
        body_lines = []
        while i < len(lines) and not _is_header(lines[i]):
            body_lines.append(lines[i])
            i += 1

        sections.append({
            "header_line": header_line,
            "body_lines": body_lines,
            "is_repealed": _is_repealed(body_lines),
            "has_appendix": _has_appendix(body_lines),
        })

    logger.info("Parsed '%s': %d sections found.", pdf_path.name, len(sections))
    return sections