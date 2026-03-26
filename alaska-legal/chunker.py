"""
chunker.py

Defines the SectionChunk dataclass and converts raw parser dictionaries
into SectionChunk objects with clean metadata.

No PDF reading, no ChromaDB, no embeddings, no LLM.
All other modules import SectionChunk from here.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Extracts chapter and decimal from "3 AAC 26.080." style headers.
# Decimal length validation is handled in _extract_section_id, not here.
# Use re.IGNORECASE on the literal "AAC" — [Aa]{3} was wrong because C is not
# in [Aa], so that pattern could never match "AAC".
SECTION_ID_RE = re.compile(r"3\s*AAC\s*(\d+)\.(\d+)", re.IGNORECASE)

# Matches the full "3 AAC XX.XXX. " prefix to strip from the title.
# Use re.IGNORECASE on literal "AAC" — [Aa]{3} can never match "AAC".
HEADER_PREFIX_RE = re.compile(r"^3\s*AAC\s*\d+\.\d+\.\s*", re.IGNORECASE)


@dataclass
class SectionChunk:
    """
    Canonical data contract for a single AAC section.
    All pipeline modules communicate through this object.
    """
    section_id:     str
    chapter:        str
    title:          str
    hierarchy_path: str
    status:         str          # "active" or "repealed"
    has_appendix:   bool
    source_pdf:     str          # filename only, no path
    text:           str
    subsections:    list = field(default_factory=list)   # reserved, always []
    raw_lines:      list = field(default_factory=list)   # raw body lines from parser


def _extract_section_id(header_line: str) -> tuple[str, str]:
    """
    Extract and normalise section_id and chapter from a PDF header line.

    All sections in the Alaska AAC PDFs in scope use three-digit decimals
    (e.g. "26.080"). Right-padding with ljust applies only when the PDF
    prints a two-digit decimal — a known formatting quirk in this document
    set. One-digit and 4+-digit decimals are not present in these PDFs;
    raise ValueError if encountered so bad parses surface immediately.

    Args:
        header_line: Full header string, e.g. "3 AAC 26.080. Claim deadline."

    Returns:
        Tuple of (section_id, chapter), e.g. ("26.080", "26").

    Raises:
        ValueError if no section ID is found, or the decimal length is
        ambiguous (not 2 or 3 digits).
    """
    m = SECTION_ID_RE.search(header_line)
    if not m:
        raise ValueError(f"No section ID found in header: {header_line!r}")

    chapter = m.group(1)
    decimal_raw = m.group(2)

    if len(decimal_raw) == 3:
        decimal = decimal_raw
    elif len(decimal_raw) == 2:
        # Right-pad: "08" -> "080". Safe for this document set.
        decimal = decimal_raw.ljust(3, "0")
    else:
        raise ValueError(
            f"Ambiguous decimal '{decimal_raw}' in header (expected 2-3 digits): {header_line!r}"
        )

    section_id = f"{chapter}.{decimal}"
    return section_id, chapter


def _extract_title(header_line: str) -> str:
    """
    Strip the '3 AAC XX.XXX. ' prefix from the header and return the title.

    Args:
        header_line: Full header string.

    Returns:
        Title string with leading/trailing whitespace stripped.
        A trailing period is preserved if present.
    """
    title = HEADER_PREFIX_RE.sub("", header_line).strip()
    return title


def _build_text(body_lines: list[str]) -> str:
    """
    Join body lines into section text, stripping leading and trailing blank lines.

    Args:
        body_lines: Raw lines from the parser.

    Returns:
        Single string with internal newlines preserved.
    """
    joined = "\n".join(body_lines)
    return joined.strip("\n")


def build_chunk(raw_section: dict, source_pdf: str) -> SectionChunk:
    """
    Convert a raw parser dictionary into a SectionChunk.

    Args:
        raw_section: Dict with keys 'header_line', 'body_lines',
                     'is_repealed', 'has_appendix' as produced by parser.py.
        source_pdf:  PDF filename (no path), e.g. "Alaska_Admin_Code__chapter_26.pdf".

    Returns:
        A populated SectionChunk object.
    """
    section_id, chapter = _extract_section_id(raw_section["header_line"])
    title = _extract_title(raw_section["header_line"])

    if raw_section["is_repealed"]:
        status = "repealed"
        text = "This section is repealed."
    else:
        status = "active"
        text = _build_text(raw_section["body_lines"])

    return SectionChunk(
        section_id=section_id,
        chapter=chapter,
        title=title,
        hierarchy_path=section_id,
        status=status,
        has_appendix=raw_section["has_appendix"],
        source_pdf=source_pdf,
        text=text,
        subsections=[],
        raw_lines=raw_section["body_lines"],
    )