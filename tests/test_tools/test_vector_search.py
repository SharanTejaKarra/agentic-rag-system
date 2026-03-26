"""Tests for the vector search tool."""

from unittest.mock import MagicMock, patch

from src.schema.models import Chunk


def _make_hits(n: int) -> list[dict]:
    return [
        {
            "id": f"hit_{i}",
            "score": 0.9 - i * 0.1,
            "payload": {
                "content": f"Legal text for hit {i}.",
                "section_ref": f"Section {i}.0",
                "source": "test_doc.pdf",
            },
        }
        for i in range(n)
    ]


class TestVectorSearchReturnsChunks:
    """Basic search returns Chunk objects."""

    def test_returns_list_of_chunks(self):
        hits = _make_hits(3)
        mock_chroma = MagicMock()
        mock_chroma.search.return_value = hits

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        with patch("src.retrieval.shared.get_chroma", return_value=mock_chroma), \
             patch("src.retrieval.shared.get_encoder", return_value=mock_encoder):
            from src.tools.vector_search import vector_search

            result = vector_search("What is a beneficiary?")

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_fields_populated(self):
        hits = _make_hits(1)
        mock_chroma = MagicMock()
        mock_chroma.search.return_value = hits

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        with patch("src.retrieval.shared.get_chroma", return_value=mock_chroma), \
             patch("src.retrieval.shared.get_encoder", return_value=mock_encoder):
            from src.tools.vector_search import vector_search

            result = vector_search("test query")

        chunk = result[0]
        assert chunk.id == "hit_0"
        assert chunk.content == "Legal text for hit 0."
        assert chunk.section_ref == "Section 0.0"
        assert chunk.score == 0.9


class TestVectorSearchWithFilters:
    """Filters are passed to ChromaDB correctly."""

    def test_section_filter_translated(self):
        mock_chroma = MagicMock()
        mock_chroma.search.return_value = []

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        with patch("src.retrieval.shared.get_chroma", return_value=mock_chroma), \
             patch("src.retrieval.shared.get_encoder", return_value=mock_encoder):
            from src.tools.vector_search import vector_search

            vector_search("test", filters={"section": "4.2"})

        call_kwargs = mock_chroma.search.call_args
        assert call_kwargs[1]["filters"] == {"section_ref": "4.2"}

    def test_entity_type_filter_translated(self):
        mock_chroma = MagicMock()
        mock_chroma.search.return_value = []

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        with patch("src.retrieval.shared.get_chroma", return_value=mock_chroma), \
             patch("src.retrieval.shared.get_encoder", return_value=mock_encoder):
            from src.tools.vector_search import vector_search

            vector_search("test", filters={"entity_type": "person"})

        call_kwargs = mock_chroma.search.call_args
        assert call_kwargs[1]["filters"] == {"entity_type": "person"}


class TestVectorSearchEmptyResults:
    """Graceful handling of no results."""

    def test_returns_empty_list(self):
        mock_chroma = MagicMock()
        mock_chroma.search.return_value = []

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        with patch("src.retrieval.shared.get_chroma", return_value=mock_chroma), \
             patch("src.retrieval.shared.get_encoder", return_value=mock_encoder):
            from src.tools.vector_search import vector_search

            result = vector_search("nonexistent concept")

        assert result == []

    def test_no_filters_when_empty_filter_dict(self):
        mock_chroma = MagicMock()
        mock_chroma.search.return_value = []

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        with patch("src.retrieval.shared.get_chroma", return_value=mock_chroma), \
             patch("src.retrieval.shared.get_encoder", return_value=mock_encoder):
            from src.tools.vector_search import vector_search

            vector_search("test", filters={})

        call_kwargs = mock_chroma.search.call_args
        assert call_kwargs[1]["filters"] is None
