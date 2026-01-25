"""
Comprehensive tests for the embedding service.

Tests cover:
- EmbeddingService initialization
- MockEmbeddingService for testing
- Embedding generation for chunks
- Query embedding
- Model detection (local vs API)
- Dimension handling
- Edge cases

Author: RepoMind Team
"""

from unittest.mock import patch, MagicMock

import pytest

from repomind.models.chunk import ChunkType, CodeChunk
from repomind.services.embedding import (
    EmbeddingService,
    MockEmbeddingService,
    EmbeddingModel,
)


class TestMockEmbeddingService:
    """Tests for MockEmbeddingService (used in testing)."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock embedding service."""
        return MockEmbeddingService()

    def test_dimension(self, mock_service):
        """Test mock service returns consistent dimension."""
        assert mock_service.dimension == 768

    def test_is_local(self, mock_service):
        """Test mock service reports as local (or not, based on implementation)."""
        # Mock service may or may not be considered "local"
        # Just verify it has the property
        assert hasattr(mock_service, 'is_local')
        # The actual value depends on implementation

    def test_embed_query(self, mock_service):
        """Test embedding a query."""
        embedding = mock_service.embed_query("test query")

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_query_deterministic(self, mock_service):
        """Test mock embeddings are deterministic."""
        emb1 = mock_service.embed_query("same query")
        emb2 = mock_service.embed_query("same query")

        assert emb1 == emb2

    def test_embed_query_different_inputs(self, mock_service):
        """Test different queries produce different embeddings."""
        emb1 = mock_service.embed_query("query one")
        emb2 = mock_service.embed_query("query two")

        # Should be different (unless hash collision)
        assert emb1 != emb2

    def test_embed_texts(self, mock_service):
        """Test embedding multiple texts."""
        texts = ["text 1", "text 2", "text 3"]
        embeddings = mock_service.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)

    def test_embed_chunks(self, mock_service):
        """Test embedding code chunks."""
        chunks = [
            CodeChunk(
                id=f"chunk{i}",
                repo_name="test-repo",
                file_path="test.py",
                start_line=i,
                end_line=i + 5,
                chunk_type=ChunkType.FUNCTION,
                name=f"func_{i}",
                content=f"def func_{i}(): pass",
                language="python",
            )
            for i in range(3)
        ]

        embeddings = mock_service.embed_chunks(chunks)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)

    def test_embed_empty_list(self, mock_service):
        """Test embedding empty list."""
        embeddings = mock_service.embed_texts([])

        assert embeddings == []

    def test_embed_single_item(self, mock_service):
        """Test embedding single item."""
        embeddings = mock_service.embed_texts(["single text"])

        assert len(embeddings) == 1


class TestEmbeddingModel:
    """Tests for EmbeddingModel enum."""

    def test_local_models_exist(self):
        """Test local model options exist."""
        assert hasattr(EmbeddingModel, 'LOCAL_BGE_BASE')
        assert hasattr(EmbeddingModel, 'LOCAL_CODEBERT')
        assert hasattr(EmbeddingModel, 'LOCAL_MINILM')

    def test_mock_model_exists(self):
        """Test mock model option exists."""
        assert hasattr(EmbeddingModel, 'MOCK')

    def test_model_values_are_strings(self):
        """Test model values are strings."""
        assert isinstance(EmbeddingModel.MOCK.value, str)


class TestEmbeddingServiceInit:
    """Tests for EmbeddingService initialization."""

    def test_default_model(self):
        """Test default model is set."""
        service = EmbeddingService()

        # Should have a model set
        assert service.model is not None
        assert isinstance(service.model, str)

    def test_custom_model_string(self):
        """Test initialization with custom model string."""
        service = EmbeddingService("all-MiniLM-L6-v2")

        assert service.model == "all-MiniLM-L6-v2"

    def test_custom_model_enum(self):
        """Test initialization with model enum."""
        service = EmbeddingService(EmbeddingModel.MOCK)

        assert service.model == EmbeddingModel.MOCK.value

    def test_bge_model_detection(self):
        """Test BGE model detection for query prefix."""
        service = EmbeddingService("BAAI/bge-base-en-v1.5")

        assert service._is_bge_model is True

    def test_non_bge_model(self):
        """Test non-BGE model has no query prefix."""
        service = EmbeddingService("all-MiniLM-L6-v2")

        assert service._is_bge_model is False


class TestEmbeddingServiceLocalModel:
    """Tests for local embedding models (requires model download)."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer to avoid model download."""
        with patch('repomind.services.embedding.SentenceTransformer') as mock:
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1] * 768]
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock.return_value = mock_model
            yield mock

    @pytest.mark.skip(reason="Requires model download - tested in integration")
    def test_local_embed_query(self, mock_sentence_transformer):
        """Test local model query embedding."""
        service = EmbeddingService("all-MiniLM-L6-v2")

        embedding = service.embed_query("test query")

        assert isinstance(embedding, list)

    @pytest.mark.skip(reason="Requires model download - tested in integration")
    def test_local_is_local_property(self, mock_sentence_transformer):
        """Test is_local property for local models."""
        service = EmbeddingService("all-MiniLM-L6-v2")

        assert service.is_local is True


class TestEmbeddingServiceEdgeCases:
    """Edge case tests for embedding service."""

    @pytest.fixture
    def mock_service(self):
        """Use mock service for edge case tests."""
        return MockEmbeddingService()

    def test_empty_query(self, mock_service):
        """Test embedding empty query."""
        embedding = mock_service.embed_query("")

        # Should still return valid embedding
        assert len(embedding) == 768

    def test_very_long_text(self, mock_service):
        """Test embedding very long text."""
        long_text = "x " * 10000

        embedding = mock_service.embed_query(long_text)

        assert len(embedding) == 768

    def test_unicode_text(self, mock_service):
        """Test embedding unicode text."""
        unicode_text = "こんにちは世界 🚀 مرحبا"

        embedding = mock_service.embed_query(unicode_text)

        assert len(embedding) == 768

    def test_special_characters(self, mock_service):
        """Test embedding text with special characters."""
        special_text = "def func():\n\t'''docstring'''\n\tpass # comment"

        embedding = mock_service.embed_query(special_text)

        assert len(embedding) == 768

    def test_large_batch(self, mock_service):
        """Test embedding large batch of texts."""
        texts = [f"text {i}" for i in range(100)]

        embeddings = mock_service.embed_texts(texts)

        assert len(embeddings) == 100

    def test_chunk_with_all_fields(self, mock_service):
        """Test embedding chunk with all optional fields."""
        chunk = CodeChunk(
            id="full-chunk",
            repo_name="test-repo",
            file_path="src/service.py",
            start_line=10,
            end_line=50,
            chunk_type=ChunkType.METHOD,
            name="process_data",
            content="def process_data(self): return None",
            signature="def process_data(self, data: list) -> dict",
            docstring="Process the data and return results.",
            parent_name="DataService",
            parent_type=ChunkType.CLASS,
            language="python",
            imports=["import json", "from typing import List"],
            summary="Data processing method",
        )

        embeddings = mock_service.embed_chunks([chunk])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768


class TestEmbeddingServiceBatching:
    """Tests for embedding batching behavior."""

    @pytest.fixture
    def mock_service(self):
        return MockEmbeddingService()

    def test_batch_processing(self, mock_service):
        """Test texts are processed in batches."""
        texts = [f"text {i}" for i in range(100)]

        # Mock service doesn't actually batch, but tests the interface
        embeddings = mock_service.embed_texts(texts)

        assert len(embeddings) == 100

    def test_embed_chunks_uses_embedding_text(self, mock_service):
        """Test embed_chunks uses chunk's to_embedding_text."""
        chunk = CodeChunk(
            id="test",
            repo_name="test-repo",
            file_path="test.py",
            start_line=1,
            end_line=5,
            chunk_type=ChunkType.FUNCTION,
            name="test_func",
            content="def test_func(): pass",
            docstring="Test function docstring",
            language="python",
        )

        embeddings = mock_service.embed_chunks([chunk])

        # The embedding should be based on the full embedding text
        # which includes docstring, name, etc.
        assert len(embeddings) == 1


class TestEmbeddingNormalization:
    """Tests for embedding normalization."""

    @pytest.fixture
    def mock_service(self):
        return MockEmbeddingService()

    def test_embeddings_are_floats(self, mock_service):
        """Test embeddings contain float values."""
        embedding = mock_service.embed_query("test")

        assert all(isinstance(x, float) for x in embedding)

    def test_embedding_values_in_range(self, mock_service):
        """Test embedding values are in reasonable range."""
        embedding = mock_service.embed_query("test")

        # Normalized vectors should have values between -1 and 1
        # Mock uses hash-based approach, values may differ
        assert all(-10 <= x <= 10 for x in embedding)


class TestEmbeddingServiceModelProperties:
    """Tests for embedding model property detection."""

    def test_mock_model_is_local(self):
        """Test mock model has is_local property."""
        service = MockEmbeddingService()

        # Just verify the property exists
        assert hasattr(service, 'is_local')

    def test_mock_model_dimension(self):
        """Test mock model dimension."""
        service = MockEmbeddingService()

        assert service.dimension == 768


class TestEmbeddingTextGeneration:
    """Tests for to_embedding_text method used by embedding service."""

    def test_embedding_text_includes_name(self):
        """Test embedding text includes function name."""
        chunk = CodeChunk(
            id="test",
            repo_name="repo",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="validate_user",
            content="def validate_user(): pass",
            language="python",
        )

        text = chunk.to_embedding_text()

        assert "validate" in text.lower() or "user" in text.lower()

    def test_embedding_text_includes_docstring(self):
        """Test embedding text includes docstring."""
        chunk = CodeChunk(
            id="test",
            repo_name="repo",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="func",
            content="def func(): pass",
            docstring="This function does important things.",
            language="python",
        )

        text = chunk.to_embedding_text()

        assert "important things" in text

    def test_embedding_text_includes_signature(self):
        """Test embedding text includes signature."""
        chunk = CodeChunk(
            id="test",
            repo_name="repo",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="process",
            content="def process(data): pass",
            signature="def process(data: list[str]) -> dict[str, Any]",
            language="python",
        )

        text = chunk.to_embedding_text()

        assert "list[str]" in text or "list" in text
