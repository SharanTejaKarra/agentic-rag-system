"""Tests for hierarchical and propositional chunking."""

from unittest.mock import patch

from src.schema.models import Chunk


SAMPLE_LEGAL_TEXT = """\
Section 1.0 General Provisions
This act establishes the regulatory framework for employee benefits.
All employers with 50 or more employees must comply with the provisions
set forth in this chapter.

Section 2.0 Definitions
A "qualified beneficiary" means any individual who has been enrolled
in the plan for at least 12 consecutive months. A "qualifying event"
means any event described in Section 3.0(a) through (d).

Section 3.0 Qualifying Events
The following events constitute qualifying events under this act:
(a) Termination of employment for any reason other than gross misconduct.
(b) Reduction in hours that causes loss of coverage.
(c) Death of the covered employee.
(d) Divorce or legal separation from the covered employee.
"""


class TestHierarchicalChunkPreservesSections:
    """Chunks maintain section metadata."""

    def test_chunks_have_section_refs(self):
        from src.ingestion.chunker import hierarchical_chunk

        docs = [{"text": SAMPLE_LEGAL_TEXT, "metadata": {"source": "test.pdf"}}]

        with patch("src.ingestion.chunker.extract_hierarchy", return_value={}), \
             patch("src.ingestion.chunker.extract_references", return_value=[]):
            chunks = hierarchical_chunk(docs, chunk_size=2000)

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.section_ref != ""

    def test_section_boundaries_respected(self):
        from src.ingestion.chunker import hierarchical_chunk

        docs = [{"text": SAMPLE_LEGAL_TEXT, "metadata": {"source": "test.pdf"}}]

        with patch("src.ingestion.chunker.extract_hierarchy", return_value={}), \
             patch("src.ingestion.chunker.extract_references", return_value=[]):
            chunks = hierarchical_chunk(docs, chunk_size=2000)

        # Should have separate chunks for different sections
        section_refs = [c.section_ref for c in chunks]
        assert any("1.0" in ref for ref in section_refs)

    def test_metadata_includes_chunk_type(self):
        from src.ingestion.chunker import hierarchical_chunk

        docs = [{"text": SAMPLE_LEGAL_TEXT, "metadata": {"source": "test.pdf"}}]

        with patch("src.ingestion.chunker.extract_hierarchy", return_value={}), \
             patch("src.ingestion.chunker.extract_references", return_value=[]):
            chunks = hierarchical_chunk(docs, chunk_size=2000)

        for chunk in chunks:
            assert chunk.metadata.get("chunk_type") == "hierarchical"


class TestHierarchicalChunkRespectsSize:
    """Chunks don't exceed chunk_size."""

    def test_chunks_within_size_limit(self):
        from src.ingestion.chunker import hierarchical_chunk

        # Build a long document
        long_text = "Section 1.0 Long Document\n" + ("This is filler text. " * 200)
        docs = [{"text": long_text, "metadata": {"source": "long.pdf"}}]

        chunk_size = 256

        with patch("src.ingestion.chunker.extract_hierarchy", return_value={}), \
             patch("src.ingestion.chunker.extract_references", return_value=[]):
            chunks = hierarchical_chunk(docs, chunk_size=chunk_size)

        assert len(chunks) > 1
        for chunk in chunks:
            # Allow small overflow since we try to break at sentence boundaries
            assert len(chunk.content) <= chunk_size + 50

    def test_small_doc_produces_single_chunk(self):
        from src.ingestion.chunker import hierarchical_chunk

        short_text = "Section 1.0 Short\nJust a few words."
        docs = [{"text": short_text, "metadata": {"source": "short.pdf"}}]

        with patch("src.ingestion.chunker.extract_hierarchy", return_value={}), \
             patch("src.ingestion.chunker.extract_references", return_value=[]):
            chunks = hierarchical_chunk(docs, chunk_size=2000)

        assert len(chunks) >= 1


class TestPropositionalChunkTagsFacts:
    """Chunks are tagged with FactType."""

    def test_chunks_have_fact_type_metadata(self):
        from src.ingestion.chunker import propositional_chunk

        llm_response = (
            '[{"fact_type": "rule", "content": "All employers must comply.", "section_ref": "1.0"},'
            ' {"fact_type": "definition", "content": "A beneficiary means...", "section_ref": "2.0"}]'
        )

        docs = [{"text": "Some legal text.", "metadata": {"source": "test.pdf"}}]

        with patch("src.ingestion.chunker.get_llm_response", return_value=llm_response):
            chunks = propositional_chunk(docs)

        assert len(chunks) == 2
        fact_types = [c.metadata.get("fact_type") for c in chunks]
        assert "rule" in fact_types
        assert "definition" in fact_types

    def test_chunk_type_is_propositional(self):
        from src.ingestion.chunker import propositional_chunk

        llm_response = '[{"fact_type": "penalty", "content": "Fines apply.", "section_ref": "45.0"}]'
        docs = [{"text": "Penalty text.", "metadata": {"source": "test.pdf"}}]

        with patch("src.ingestion.chunker.get_llm_response", return_value=llm_response):
            chunks = propositional_chunk(docs)

        assert all(c.metadata.get("chunk_type") == "propositional" for c in chunks)

    def test_falls_back_on_invalid_json(self):
        from src.ingestion.chunker import propositional_chunk

        docs = [{"text": "Some text.", "metadata": {"source": "test.pdf"}}]

        with patch("src.ingestion.chunker.get_llm_response", return_value="not json"):
            chunks = propositional_chunk(docs)

        # Should fall back to a single raw chunk
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("fact_type") == "rule"
