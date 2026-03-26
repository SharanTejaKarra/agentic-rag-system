"""Document parsing for PDF, HTML, and plain text files."""

import os
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Supported file extensions and their categories
_TEXT_EXTENSIONS = {".txt", ".text", ".md"}
_HTML_EXTENSIONS = {".html", ".htm"}
_PDF_EXTENSIONS = {".pdf"}


def parse_document(file_path: str) -> list[dict]:
    """Parse a document file and extract text with metadata.

    Supports PDF (via LlamaIndex SimpleDirectoryReader), HTML, and plain text.
    Returns list of dicts with "text" and "metadata" keys.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    logger.info("Parsing %s (type: %s)", path.name, ext)

    if ext in _PDF_EXTENSIONS:
        return _parse_pdf(path)
    elif ext in _HTML_EXTENSIONS:
        return _parse_html(path)
    elif ext in _TEXT_EXTENSIONS:
        return _parse_text(path)
    else:
        logger.warning("Unknown file type %s, treating as plain text", ext)
        return _parse_text(path)


def _parse_pdf(path: Path) -> list[dict]:
    """Parse PDF using LlamaIndex SimpleDirectoryReader."""
    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(input_files=[str(path)])
    documents = reader.load_data()

    results = []
    for i, doc in enumerate(documents):
        results.append({
            "text": doc.text,
            "metadata": {
                "source": path.name,
                "page": str(i + 1),
                "file_path": str(path),
                **doc.metadata,
            },
        })
    return results


def _parse_html(path: Path) -> list[dict]:
    """Parse HTML, stripping tags to get raw text."""
    from html.parser import HTMLParser
    from io import StringIO

    class _TagStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []

        def handle_data(self, data: str) -> None:
            self._parts.append(data)

        def get_text(self) -> str:
            return "".join(self._parts)

    raw = path.read_text(encoding="utf-8", errors="replace")
    stripper = _TagStripper()
    stripper.feed(raw)
    text = stripper.get_text()

    return [{
        "text": text,
        "metadata": {
            "source": path.name,
            "file_path": str(path),
            "format": "html",
        },
    }]


def _parse_text(path: Path) -> list[dict]:
    """Parse plain text file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{
        "text": text,
        "metadata": {
            "source": path.name,
            "file_path": str(path),
            "format": "text",
        },
    }]
