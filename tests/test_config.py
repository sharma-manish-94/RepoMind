"""
Comprehensive tests for the configuration module.

Tests cover:
- IndexConfig with default values and property methods
- SearchConfig with all thresholds
- EmbeddingConfig with model settings
- RepositoryConfig with pattern matching logic
- Config with repository discovery and backward compatibility
- Global config management functions

Author: RepoMind Team
"""

import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from repomind.config import (
    Config,
    IndexConfig,
    SearchConfig,
    EmbeddingConfig,
    RepositoryConfig,
    get_config,
    set_config,
    reset_config,
)


class TestIndexConfig:
    """Tests for IndexConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = IndexConfig()

        assert config.max_chunk_lines == 500
        assert config.include_imports is True
        assert config.include_comments is False
        assert "python" in config.languages
        assert "java" in config.languages
        assert "typescript" in config.languages
        assert "javascript" in config.languages

    def test_chroma_dir_property(self):
        """Test chroma_dir property returns correct path."""
        config = IndexConfig(data_dir=Path("/tmp/test-data"))

        assert config.chroma_dir == Path("/tmp/test-data/chroma")

    def test_metadata_dir_property(self):
        """Test metadata_dir property returns correct path."""
        config = IndexConfig(data_dir=Path("/tmp/test-data"))

        assert config.metadata_dir == Path("/tmp/test-data/metadata")

    def test_custom_data_dir(self):
        """Test custom data directory."""
        custom_path = Path("/custom/path")
        config = IndexConfig(data_dir=custom_path)

        assert config.data_dir == custom_path

    def test_ignore_patterns(self):
        """Test default ignore patterns."""
        config = IndexConfig()

        assert "node_modules/**" in config.ignore_patterns
        assert "**/__pycache__/**" in config.ignore_patterns
        assert "**/.git/**" in config.ignore_patterns

    def test_custom_ignore_patterns(self):
        """Test custom ignore patterns."""
        config = IndexConfig(ignore_patterns=["*.test.py", "*_test.go"])

        assert "*.test.py" in config.ignore_patterns
        assert "*_test.go" in config.ignore_patterns
        assert "node_modules/**" not in config.ignore_patterns


class TestSearchConfig:
    """Tests for SearchConfig."""

    def test_default_values(self):
        """Test default search configuration values."""
        config = SearchConfig()

        assert config.default_results == 10
        assert config.max_results == 100
        assert config.similarity_threshold == 0.35
        assert config.high_confidence_threshold == 0.70
        assert config.enable_hybrid_search is True
        assert config.keyword_boost_factor == 0.15
        assert config.enable_query_expansion is True

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        config = SearchConfig(
            similarity_threshold=0.5,
            high_confidence_threshold=0.9,
        )

        assert config.similarity_threshold == 0.5
        assert config.high_confidence_threshold == 0.9

    def test_hybrid_search_disabled(self):
        """Test disabling hybrid search."""
        config = SearchConfig(enable_hybrid_search=False)

        assert config.enable_hybrid_search is False

    def test_custom_keyword_boost(self):
        """Test custom keyword boost factor."""
        config = SearchConfig(keyword_boost_factor=0.25)

        assert config.keyword_boost_factor == 0.25


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self):
        """Test default embedding configuration."""
        config = EmbeddingConfig()

        assert config.model == "BAAI/bge-base-en-v1.5"
        assert config.batch_size == 32
        assert config.normalize_embeddings is True
        assert "Represent this code search query" in config.query_instruction

    def test_custom_model(self):
        """Test custom embedding model."""
        config = EmbeddingConfig(model="microsoft/codebert-base")

        assert config.model == "microsoft/codebert-base"

    def test_custom_batch_size(self):
        """Test custom batch size."""
        config = EmbeddingConfig(batch_size=64)

        assert config.batch_size == 64


class TestRepositoryConfig:
    """Tests for RepositoryConfig with pattern matching."""

    def test_default_values(self):
        """Test default repository configuration."""
        config = RepositoryConfig()

        assert config.auto_discover is True
        assert config.include_patterns == ["*"]
        assert config.min_files_threshold == 1
        assert ".git" in config.exclude_patterns
        assert "node_modules" in config.exclude_patterns

    def test_matches_include_pattern_wildcard(self):
        """Test include pattern matching with wildcard."""
        config = RepositoryConfig(include_patterns=["*"])

        assert config.matches_include_pattern("any-repo") is True
        assert config.matches_include_pattern("another-repo") is True

    def test_matches_include_pattern_specific(self):
        """Test include pattern matching with specific patterns."""
        config = RepositoryConfig(include_patterns=["service-*", "lib-*"])

        assert config.matches_include_pattern("service-auth") is True
        assert config.matches_include_pattern("lib-utils") is True
        assert config.matches_include_pattern("other-repo") is False

    def test_matches_exclude_pattern(self):
        """Test exclude pattern matching."""
        config = RepositoryConfig(exclude_patterns=["test-*", ".git"])

        assert config.matches_exclude_pattern("test-integration") is True
        assert config.matches_exclude_pattern(".git") is True
        assert config.matches_exclude_pattern("main-app") is False

    def test_should_include_repository_basic(self):
        """Test repository inclusion decision with basic patterns."""
        config = RepositoryConfig(
            include_patterns=["*"],
            exclude_patterns=[".git", "node_modules"],
        )

        assert config.should_include_repository("my-service") is True
        assert config.should_include_repository(".git") is False
        assert config.should_include_repository("node_modules") is False

    def test_should_include_repository_complex(self):
        """Test repository inclusion with complex pattern combinations."""
        config = RepositoryConfig(
            include_patterns=["service-*", "lib-*"],
            exclude_patterns=["*-deprecated", "*-old"],
        )

        # Matches include, doesn't match exclude
        assert config.should_include_repository("service-auth") is True
        assert config.should_include_repository("lib-utils") is True

        # Matches include but also matches exclude
        assert config.should_include_repository("service-deprecated") is False
        assert config.should_include_repository("lib-old") is False

        # Doesn't match include
        assert config.should_include_repository("app-main") is False

    def test_should_include_repository_exclude_takes_priority(self):
        """Test that exclude patterns take priority over include."""
        config = RepositoryConfig(
            include_patterns=["*"],
            exclude_patterns=["secret-*"],
        )

        assert config.should_include_repository("my-repo") is True
        assert config.should_include_repository("secret-keys") is False

    def test_pattern_matching_case_sensitive(self):
        """Test that pattern matching is case-sensitive."""
        config = RepositoryConfig(include_patterns=["Service-*"])

        assert config.matches_include_pattern("Service-Auth") is True
        assert config.matches_include_pattern("service-auth") is False

    def test_pattern_with_glob_characters(self):
        """Test patterns with various glob characters."""
        config = RepositoryConfig(include_patterns=["v[0-9]*", "release-?"])

        assert config.matches_include_pattern("v1-service") is True
        assert config.matches_include_pattern("v2-lib") is True
        assert config.matches_include_pattern("release-a") is True
        assert config.matches_include_pattern("release-ab") is False  # ? matches single char


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()

        assert isinstance(config.index, IndexConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.repository, RepositoryConfig)
        assert config.repos_dir is None
        assert config.repos_list is None

    def test_config_with_repos_dir(self):
        """Test configuration with repos_dir."""
        config = Config(repos_dir=Path("/path/to/repos"))

        assert config.repos_dir == Path("/path/to/repos")

    def test_config_with_repos_list(self):
        """Test configuration with explicit repos_list."""
        config = Config(repos_list=["repo1", "repo2", "repo3"])

        assert config.repos_list == ["repo1", "repo2", "repo3"]

    def test_get_repositories_to_index_explicit_list(self):
        """Test get_repositories_to_index with explicit repos_list."""
        config = Config(repos_list=["explicit-repo"])

        repos = config.get_repositories_to_index()

        assert repos == ["explicit-repo"]

    def test_get_repositories_to_index_no_repos_dir_raises(self):
        """Test that missing repos_dir raises ValueError."""
        config = Config(repos_dir=None, repos_list=None)

        with pytest.raises(ValueError) as exc_info:
            config.get_repositories_to_index()

        assert "repos_dir must be set" in str(exc_info.value)

    def test_discover_repositories_nonexistent_dir(self):
        """Test discovery with non-existent directory."""
        config = Config(repos_dir=Path("/nonexistent/path"))

        repos = config._discover_repositories()

        assert repos == []

    def test_discover_repositories_empty_dir(self, tmp_path):
        """Test discovery with empty directory."""
        config = Config(repos_dir=tmp_path)

        repos = config._discover_repositories()

        assert repos == []

    def test_discover_repositories_with_subdirs(self, tmp_path):
        """Test discovery with valid subdirectories."""
        # Create subdirectories with source files
        (tmp_path / "repo1").mkdir()
        (tmp_path / "repo1" / "main.py").write_text("print('hello')")

        (tmp_path / "repo2").mkdir()
        (tmp_path / "repo2" / "app.ts").write_text("console.log('hello')")

        # Create a file (should be ignored)
        (tmp_path / "not-a-dir.txt").write_text("ignored")

        config = Config(repos_dir=tmp_path)
        repos = config._discover_repositories()

        assert "repo1" in repos
        assert "repo2" in repos
        assert "not-a-dir.txt" not in repos

    def test_discover_repositories_excludes_patterns(self, tmp_path):
        """Test that discovery excludes configured patterns."""
        # Create directories
        (tmp_path / "my-service").mkdir()
        (tmp_path / "my-service" / "main.py").write_text("code")

        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git stuff")

        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.js").write_text("module")

        config = Config(repos_dir=tmp_path)
        repos = config._discover_repositories()

        assert "my-service" in repos
        assert ".git" not in repos
        assert "node_modules" not in repos

    def test_discover_repositories_with_include_patterns(self, tmp_path):
        """Test discovery with specific include patterns."""
        # Create directories
        (tmp_path / "service-auth").mkdir()
        (tmp_path / "service-auth" / "main.py").write_text("code")

        (tmp_path / "lib-utils").mkdir()
        (tmp_path / "lib-utils" / "util.py").write_text("code")

        (tmp_path / "other-app").mkdir()
        (tmp_path / "other-app" / "app.py").write_text("code")

        config = Config(
            repos_dir=tmp_path,
            repository=RepositoryConfig(include_patterns=["service-*"]),
        )
        repos = config._discover_repositories()

        assert "service-auth" in repos
        assert "lib-utils" not in repos
        assert "other-app" not in repos

    def test_is_valid_repository_with_python_files(self, tmp_path):
        """Test repository validation with Python files."""
        repo_path = tmp_path / "python-repo"
        repo_path.mkdir()
        (repo_path / "main.py").write_text("print('hello')")

        config = Config()
        assert config._is_valid_repository(repo_path) is True

    def test_is_valid_repository_with_typescript_files(self, tmp_path):
        """Test repository validation with TypeScript files."""
        repo_path = tmp_path / "ts-repo"
        repo_path.mkdir()
        (repo_path / "app.ts").write_text("console.log('hello')")

        config = Config()
        assert config._is_valid_repository(repo_path) is True

    def test_is_valid_repository_with_java_files(self, tmp_path):
        """Test repository validation with Java files."""
        repo_path = tmp_path / "java-repo"
        repo_path.mkdir()
        (repo_path / "Main.java").write_text("public class Main {}")

        config = Config()
        assert config._is_valid_repository(repo_path) is True

    def test_is_valid_repository_empty_dir(self, tmp_path):
        """Test repository validation with empty directory."""
        repo_path = tmp_path / "empty-repo"
        repo_path.mkdir()

        config = Config()
        assert config._is_valid_repository(repo_path) is False

    def test_is_valid_repository_no_source_files(self, tmp_path):
        """Test repository validation with non-source files only."""
        repo_path = tmp_path / "docs-only"
        repo_path.mkdir()
        (repo_path / "README.md").write_text("# Docs")
        (repo_path / "config.json").write_text("{}")

        config = Config()
        assert config._is_valid_repository(repo_path) is False

    def test_is_valid_repository_nested_source_files(self, tmp_path):
        """Test repository validation with nested source files."""
        repo_path = tmp_path / "nested-repo"
        repo_path.mkdir()
        (repo_path / "src").mkdir()
        (repo_path / "src" / "app.py").write_text("code")

        config = Config()
        assert config._is_valid_repository(repo_path) is True

    def test_is_valid_repository_with_min_threshold(self, tmp_path):
        """Test repository validation with custom minimum file threshold."""
        repo_path = tmp_path / "small-repo"
        repo_path.mkdir()
        (repo_path / "app.py").write_text("code")

        # With min_files_threshold=2, single file shouldn't pass
        config = Config(repository=RepositoryConfig(min_files_threshold=2))
        assert config._is_valid_repository(repo_path) is False

        # Add another file
        (repo_path / "util.py").write_text("code")
        assert config._is_valid_repository(repo_path) is True

    def test_load_default(self):
        """Test load_default class method."""
        config = Config.load_default()

        assert isinstance(config, Config)
        assert config.repos_dir is None


class TestGlobalConfigManagement:
    """Tests for global config management functions."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config_returns_default(self):
        """Test get_config returns default when not set."""
        config = get_config()

        assert isinstance(config, Config)

    def test_set_config(self):
        """Test set_config updates global config."""
        custom_config = Config(repos_list=["custom-repo"])

        set_config(custom_config)
        retrieved = get_config()

        assert retrieved.repos_list == ["custom-repo"]

    def test_reset_config(self):
        """Test reset_config clears global config."""
        custom_config = Config(repos_list=["custom-repo"])
        set_config(custom_config)

        reset_config()

        # Should create new default config
        config = get_config()
        assert config.repos_list is None

    def test_get_config_singleton_behavior(self):
        """Test that get_config returns same instance when not reset."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_set_config_replaces_singleton(self):
        """Test that set_config replaces the singleton."""
        original = get_config()
        new_config = Config(repos_list=["new"])

        set_config(new_config)
        current = get_config()

        assert current is not original
        assert current is new_config
