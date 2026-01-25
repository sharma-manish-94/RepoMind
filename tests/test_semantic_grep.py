"""
Comprehensive tests for the semantic_grep tool.

Tests cover:
- Basic search functionality
- Filter options (repo, type, language)
- Hybrid search
- Threshold handling
- Error cases
- Result formatting

Author: RepoMind Team
"""

from unittest.mock import patch, MagicMock

import pytest

from repomind.config import Config, SearchConfig, set_config, reset_config
from repomind.models.chunk import ChunkType, CodeChunk
from repomind.tools.semantic_grep import semantic_grep


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset config before and after each test."""
    reset_config()
    yield
    reset_config()


class TestSemanticGrepBasics:
    """Basic tests for semantic_grep."""

    def test_empty_query_returns_error(self):
        """Test that empty query returns error."""
        result = semantic_grep(query="")

        assert "error" in result
        assert "empty" in result["error"].lower()

    def test_whitespace_query_returns_error(self):
        """Test that whitespace-only query returns error."""
        result = semantic_grep(query="   ")

        assert "error" in result

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_basic_search(self, mock_embed_class, mock_storage_class):
        """Test basic search returns results."""
        # Setup mocks
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        chunk = CodeChunk(
            id="test1",
            repo_name="test-repo",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_type=ChunkType.FUNCTION,
            name="test_function",
            content="def test_function(): pass",
            language="python",
        )

        mock_storage = MagicMock()
        mock_storage.search.return_value = [(chunk, 0.85)]
        mock_storage_class.return_value = mock_storage

        result = semantic_grep(query="test function")

        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["name"] == "test_function"

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_no_results(self, mock_embed_class, mock_storage_class):
        """Test search with no results."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        result = semantic_grep(query="nonexistent query")

        assert "results" in result
        assert len(result["results"]) == 0
        assert "message" in result


class TestSemanticGrepFilters:
    """Tests for search filters."""

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_repo_filter(self, mock_embed_class, mock_storage_class):
        """Test repository filter is passed to storage."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test", repo_filter="my-repo")

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["repo_filter"] == "my-repo"

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_type_filter_valid(self, mock_embed_class, mock_storage_class):
        """Test valid type filter is applied."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test", type_filter="function")

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["type_filter"] == ChunkType.FUNCTION

    def test_type_filter_invalid(self):
        """Test invalid type filter returns error."""
        result = semantic_grep(query="test", type_filter="invalid_type")

        assert "error" in result
        assert "Invalid type_filter" in result["error"]

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_language_filter(self, mock_embed_class, mock_storage_class):
        """Test language filter is passed to storage."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test", language_filter="python")

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["language_filter"] == "python"

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_multiple_filters(self, mock_embed_class, mock_storage_class):
        """Test multiple filters applied together."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(
            query="test",
            repo_filter="my-repo",
            type_filter="class",
            language_filter="java"
        )

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["repo_filter"] == "my-repo"
        assert call_kwargs["type_filter"] == ChunkType.CLASS
        assert call_kwargs["language_filter"] == "java"


class TestSemanticGrepThresholds:
    """Tests for similarity threshold handling."""

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_custom_threshold(self, mock_embed_class, mock_storage_class):
        """Test custom similarity threshold."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test", similarity_threshold=0.5)

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["similarity_threshold"] == 0.5

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_default_threshold_from_config(self, mock_embed_class, mock_storage_class):
        """Test default threshold comes from config."""
        # Set custom config
        config = Config(search=SearchConfig(similarity_threshold=0.6))
        set_config(config)

        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test")

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["similarity_threshold"] == 0.6


class TestSemanticGrepResults:
    """Tests for result formatting."""

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_result_contains_score(self, mock_embed_class, mock_storage_class):
        """Test results contain similarity score."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        chunk = CodeChunk(
            id="test1",
            repo_name="test-repo",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_type=ChunkType.FUNCTION,
            name="test_func",
            content="def test_func(): pass",
            language="python",
        )

        mock_storage = MagicMock()
        mock_storage.search.return_value = [(chunk, 0.85)]
        mock_storage_class.return_value = mock_storage

        result = semantic_grep(query="test")

        assert result["results"][0]["score"] == 0.85

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_result_contains_location(self, mock_embed_class, mock_storage_class):
        """Test results contain location info."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        chunk = CodeChunk(
            id="test1",
            repo_name="test-repo",
            file_path="src/handler.py",
            start_line=45,
            end_line=60,
            chunk_type=ChunkType.FUNCTION,
            name="handler",
            content="def handler(): pass",
            language="python",
        )

        mock_storage = MagicMock()
        mock_storage.search.return_value = [(chunk, 0.9)]
        mock_storage_class.return_value = mock_storage

        result = semantic_grep(query="test")

        res = result["results"][0]
        assert res["repo"] == "test-repo"
        assert res["file"] == "src/handler.py"
        assert res["line"] == 45

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_result_contains_docstring(self, mock_embed_class, mock_storage_class):
        """Test results include docstring when present."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        chunk = CodeChunk(
            id="test1",
            repo_name="test-repo",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_type=ChunkType.FUNCTION,
            name="documented",
            content="def documented(): pass",
            docstring="This function is documented.",
            language="python",
        )

        mock_storage = MagicMock()
        mock_storage.search.return_value = [(chunk, 0.8)]
        mock_storage_class.return_value = mock_storage

        result = semantic_grep(query="test")

        assert "docstring" in result["results"][0]
        assert "documented" in result["results"][0]["docstring"]

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_result_contains_code_by_default(self, mock_embed_class, mock_storage_class):
        """Test results include code by default."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        chunk = CodeChunk(
            id="test1",
            repo_name="test-repo",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_type=ChunkType.FUNCTION,
            name="func",
            content="def func():\n    return 42",
            language="python",
        )

        mock_storage = MagicMock()
        mock_storage.search.return_value = [(chunk, 0.8)]
        mock_storage_class.return_value = mock_storage

        result = semantic_grep(query="test", show_code=True)

        assert "code" in result["results"][0]
        assert "return 42" in result["results"][0]["code"]

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_result_excludes_code_when_disabled(self, mock_embed_class, mock_storage_class):
        """Test results exclude code when show_code=False."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        chunk = CodeChunk(
            id="test1",
            repo_name="test-repo",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_type=ChunkType.FUNCTION,
            name="func",
            content="def func(): pass",
            language="python",
        )

        mock_storage = MagicMock()
        mock_storage.search.return_value = [(chunk, 0.8)]
        mock_storage_class.return_value = mock_storage

        result = semantic_grep(query="test", show_code=False)

        assert "code" not in result["results"][0]


class TestSemanticGrepMockEmbeddings:
    """Tests for mock embeddings mode."""

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.MockEmbeddingService')
    def test_mock_embeddings_flag(self, mock_embed_class, mock_storage_class):
        """Test mock embeddings service is used when flag is set."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test", use_mock_embeddings=True)

        # MockEmbeddingService should be instantiated
        mock_embed_class.assert_called_once()


class TestSemanticGrepHybridSearch:
    """Tests for hybrid search functionality."""

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_hybrid_search_enabled(self, mock_embed_class, mock_storage_class):
        """Test hybrid search is enabled by default."""
        config = Config(search=SearchConfig(enable_hybrid_search=True))
        set_config(config)

        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test function")

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["enable_hybrid_search"] is True
        assert call_kwargs["query_text"] == "test function"

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_hybrid_search_disabled(self, mock_embed_class, mock_storage_class):
        """Test hybrid search can be disabled via config."""
        config = Config(search=SearchConfig(enable_hybrid_search=False))
        set_config(config)

        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test")

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["enable_hybrid_search"] is False


class TestSemanticGrepNResults:
    """Tests for number of results handling."""

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_default_n_results(self, mock_embed_class, mock_storage_class):
        """Test default number of results is 10."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test")

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["n_results"] == 10

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_custom_n_results(self, mock_embed_class, mock_storage_class):
        """Test custom number of results."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        mock_storage_class.return_value = mock_storage

        semantic_grep(query="test", n_results=25)

        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["n_results"] == 25


class TestSemanticGrepQualifiedNames:
    """Tests for qualified name handling in results."""

    @patch('repomind.tools.semantic_grep.StorageService')
    @patch('repomind.tools.semantic_grep.EmbeddingService')
    def test_method_qualified_name(self, mock_embed_class, mock_storage_class):
        """Test method results include qualified name with class."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_embed_class.return_value = mock_embed

        chunk = CodeChunk(
            id="test1",
            repo_name="test-repo",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_type=ChunkType.METHOD,
            name="process",
            content="def process(self): pass",
            parent_name="DataService",
            language="python",
        )

        mock_storage = MagicMock()
        mock_storage.search.return_value = [(chunk, 0.9)]
        mock_storage_class.return_value = mock_storage

        result = semantic_grep(query="test")

        assert result["results"][0]["name"] == "DataService.process"
