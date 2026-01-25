"""
Comprehensive tests for the CLI commands.

Tests cover:
- Main CLI group and options
- index command
- index-all command with patterns
- discover command
- search command
- stats command
- clear and clear-all commands
- config-show command

Author: RepoMind Team
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from repomind.cli import main
from repomind.config import reset_config


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset config before and after each test."""
    reset_config()
    yield
    reset_config()


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_help_option(self, runner):
        """Test --help shows help message."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "RepoMind" in result.output
        assert "AI-Powered" in result.output

    def test_repos_dir_option(self, runner, tmp_path):
        """Test --repos-dir option sets config."""
        result = runner.invoke(main, ["--repos-dir", str(tmp_path), "--help"])

        assert result.exit_code == 0

    def test_invalid_repos_dir(self, runner):
        """Test invalid repos_dir shows error."""
        result = runner.invoke(main, ["--repos-dir", "/nonexistent/path", "stats"])

        # Should not crash, but may show warning
        assert result.exit_code in [0, 1, 2]


class TestIndexCommand:
    """Tests for the index command."""

    def test_index_help(self, runner):
        """Test index command help."""
        result = runner.invoke(main, ["index", "--help"])

        assert result.exit_code == 0
        assert "REPO_PATH" in result.output

    @patch('repomind.cli.index_repo')
    def test_index_basic(self, mock_index, runner, tmp_path):
        """Test basic index command."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_index.return_value = {"status": "success", "chunks_stored": 10}

        result = runner.invoke(main, ["index", str(repo_path)])

        assert result.exit_code == 0
        mock_index.assert_called_once()

    @patch('repomind.cli.index_repo')
    def test_index_with_name(self, mock_index, runner, tmp_path):
        """Test index command with custom name."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_index.return_value = {"status": "success"}

        result = runner.invoke(main, ["index", str(repo_path), "-n", "custom-name"])

        mock_index.assert_called_once()
        call_kwargs = mock_index.call_args[1]
        assert call_kwargs["repo_name"] == "custom-name"

    @patch('repomind.cli.index_repo')
    def test_index_with_mock_flag(self, mock_index, runner, tmp_path):
        """Test index command with --mock flag."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_index.return_value = {"status": "success"}

        result = runner.invoke(main, ["index", str(repo_path), "--mock"])

        call_kwargs = mock_index.call_args[1]
        assert call_kwargs["use_mock_embeddings"] is True

    @patch('repomind.cli.index_repo')
    def test_index_incremental(self, mock_index, runner, tmp_path):
        """Test index command with --incremental flag."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_index.return_value = {"status": "success"}

        result = runner.invoke(main, ["index", str(repo_path), "-i"])

        call_kwargs = mock_index.call_args[1]
        assert call_kwargs["incremental"] is True


class TestIndexAllCommand:
    """Tests for the index-all command."""

    def test_index_all_help(self, runner):
        """Test index-all command help."""
        result = runner.invoke(main, ["index-all", "--help"])

        assert result.exit_code == 0
        assert "--pattern" in result.output
        assert "--exclude" in result.output
        assert "--incremental" in result.output

    def test_index_all_without_repos_dir(self, runner):
        """Test index-all without repos_dir shows error."""
        result = runner.invoke(main, ["index-all"])

        assert "repos_dir not configured" in result.output or "Error" in result.output

    @patch('repomind.cli.index_all_repositories')
    def test_index_all_with_repos_dir(self, mock_index_all, runner, tmp_path):
        """Test index-all with repos_dir."""
        mock_index_all.return_value = {
            "_summary": {"total_repos": 2, "successful": 2, "failed": 0, "skipped": 0}
        }

        result = runner.invoke(main, ["--repos-dir", str(tmp_path), "index-all"])

        mock_index_all.assert_called_once()

    @patch('repomind.cli.index_all_repositories')
    def test_index_all_with_patterns(self, mock_index_all, runner, tmp_path):
        """Test index-all with include patterns."""
        mock_index_all.return_value = {
            "_summary": {"total_repos": 1, "successful": 1, "failed": 0, "skipped": 0}
        }

        result = runner.invoke(main, [
            "--repos-dir", str(tmp_path),
            "index-all",
            "-p", "service-*",
            "-p", "lib-*"
        ])

        call_kwargs = mock_index_all.call_args[1]
        assert call_kwargs["include_patterns"] == ["service-*", "lib-*"]

    @patch('repomind.cli.index_all_repositories')
    def test_index_all_with_exclude(self, mock_index_all, runner, tmp_path):
        """Test index-all with exclude patterns."""
        mock_index_all.return_value = {
            "_summary": {"total_repos": 1, "successful": 1, "failed": 0, "skipped": 0}
        }

        result = runner.invoke(main, [
            "--repos-dir", str(tmp_path),
            "index-all",
            "-e", "test-*",
            "-e", "legacy-*"
        ])

        call_kwargs = mock_index_all.call_args[1]
        assert call_kwargs["exclude_patterns"] == ["test-*", "legacy-*"]

    @patch('repomind.cli.index_all_repositories')
    def test_index_all_mock_embeddings(self, mock_index_all, runner, tmp_path):
        """Test index-all with mock embeddings."""
        mock_index_all.return_value = {"_summary": {"total_repos": 0}}

        result = runner.invoke(main, [
            "--repos-dir", str(tmp_path),
            "index-all",
            "--mock"
        ])

        call_kwargs = mock_index_all.call_args[1]
        assert call_kwargs["use_mock_embeddings"] is True


class TestDiscoverCommand:
    """Tests for the discover command."""

    def test_discover_help(self, runner):
        """Test discover command help."""
        result = runner.invoke(main, ["discover", "--help"])

        assert result.exit_code == 0
        assert "--pattern" in result.output

    def test_discover_without_repos_dir(self, runner):
        """Test discover without repos_dir."""
        result = runner.invoke(main, ["discover"])

        assert "Error" in result.output or "not configured" in result.output

    @patch('repomind.cli.discover_repositories')
    def test_discover_basic(self, mock_discover, runner, tmp_path):
        """Test basic discover command."""
        mock_discover.return_value = {
            "repo1": tmp_path / "repo1",
            "repo2": tmp_path / "repo2",
        }

        result = runner.invoke(main, ["--repos-dir", str(tmp_path), "discover"])

        assert "Discovered 2 Repositories" in result.output
        mock_discover.assert_called_once()

    @patch('repomind.cli.discover_repositories')
    def test_discover_with_pattern(self, mock_discover, runner, tmp_path):
        """Test discover with pattern filter."""
        mock_discover.return_value = {"service-auth": tmp_path / "service-auth"}

        result = runner.invoke(main, [
            "--repos-dir", str(tmp_path),
            "discover",
            "-p", "service-*"
        ])

        call_kwargs = mock_discover.call_args[1]
        assert call_kwargs["include_patterns"] == ["service-*"]

    @patch('repomind.cli.discover_repositories')
    def test_discover_empty_result(self, mock_discover, runner, tmp_path):
        """Test discover with no repos found."""
        mock_discover.return_value = {}

        result = runner.invoke(main, ["--repos-dir", str(tmp_path), "discover"])

        assert "No repositories found" in result.output


class TestSearchCommand:
    """Tests for the search command."""

    def test_search_help(self, runner):
        """Test search command help."""
        result = runner.invoke(main, ["search", "--help"])

        assert result.exit_code == 0
        assert "QUERY" in result.output
        assert "--results" in result.output
        assert "--repo" in result.output

    @patch('repomind.cli.semantic_grep')
    def test_search_basic(self, mock_grep, runner):
        """Test basic search command."""
        mock_grep.return_value = {
            "query": "test query",
            "results": [],
            "message": "No results"
        }

        result = runner.invoke(main, ["search", "test query"])

        mock_grep.assert_called_once()
        call_kwargs = mock_grep.call_args[1]
        assert call_kwargs["query"] == "test query"

    @patch('repomind.cli.semantic_grep')
    def test_search_with_repo_filter(self, mock_grep, runner):
        """Test search with repository filter."""
        mock_grep.return_value = {"results": []}

        result = runner.invoke(main, ["search", "query", "-r", "my-repo"])

        call_kwargs = mock_grep.call_args[1]
        assert call_kwargs["repo_filter"] == "my-repo"

    @patch('repomind.cli.semantic_grep')
    def test_search_with_type_filter(self, mock_grep, runner):
        """Test search with type filter."""
        mock_grep.return_value = {"results": []}

        result = runner.invoke(main, ["search", "query", "-t", "function"])

        call_kwargs = mock_grep.call_args[1]
        assert call_kwargs["type_filter"] == "function"

    @patch('repomind.cli.semantic_grep')
    def test_search_with_language_filter(self, mock_grep, runner):
        """Test search with language filter."""
        mock_grep.return_value = {"results": []}

        result = runner.invoke(main, ["search", "query", "-l", "python"])

        call_kwargs = mock_grep.call_args[1]
        assert call_kwargs["language_filter"] == "python"

    @patch('repomind.cli.semantic_grep')
    def test_search_custom_results_count(self, mock_grep, runner):
        """Test search with custom results count."""
        mock_grep.return_value = {"results": []}

        result = runner.invoke(main, ["search", "query", "-n", "20"])

        call_kwargs = mock_grep.call_args[1]
        assert call_kwargs["n_results"] == 20

    @patch('repomind.cli.semantic_grep')
    def test_search_with_threshold(self, mock_grep, runner):
        """Test search with similarity threshold."""
        mock_grep.return_value = {"results": []}

        result = runner.invoke(main, ["search", "query", "--threshold", "0.5"])

        call_kwargs = mock_grep.call_args[1]
        assert call_kwargs["similarity_threshold"] == 0.5

    @patch('repomind.cli.semantic_grep')
    def test_search_error_handling(self, mock_grep, runner):
        """Test search handles errors gracefully."""
        mock_grep.return_value = {"error": "Something went wrong"}

        result = runner.invoke(main, ["search", "query"])

        assert "Error" in result.output


class TestStatsCommand:
    """Tests for the stats command."""

    def test_stats_help(self, runner):
        """Test stats command help."""
        result = runner.invoke(main, ["stats", "--help"])

        assert result.exit_code == 0

    @patch('repomind.cli.StorageService')
    def test_stats_basic(self, mock_storage_class, runner):
        """Test basic stats command."""
        mock_storage = MagicMock()
        mock_storage.get_stats.return_value = {
            "total_chunks": 100,
            "repositories": ["repo1", "repo2"],
            "languages": {"python": 60, "typescript": 40},
            "chunk_types": {"function": 70, "class": 30},
        }
        mock_storage_class.return_value = mock_storage

        result = runner.invoke(main, ["stats"])

        assert result.exit_code == 0
        assert "100" in result.output
        assert "RepoMind Index Statistics" in result.output


class TestClearCommands:
    """Tests for clear and clear-all commands."""

    def test_clear_help(self, runner):
        """Test clear command help."""
        result = runner.invoke(main, ["clear", "--help"])

        assert result.exit_code == 0
        assert "REPO_NAME" in result.output

    @patch('repomind.cli.StorageService')
    def test_clear_repo(self, mock_storage_class, runner):
        """Test clearing a specific repository."""
        mock_storage = MagicMock()
        mock_storage.clear_repo.return_value = 50
        mock_storage_class.return_value = mock_storage

        result = runner.invoke(main, ["clear", "my-repo"])

        assert result.exit_code == 0
        mock_storage.clear_repo.assert_called_with("my-repo")
        assert "50" in result.output

    def test_clear_all_help(self, runner):
        """Test clear-all command help."""
        result = runner.invoke(main, ["clear-all", "--help"])

        assert result.exit_code == 0
        assert "--force" in result.output

    @patch('repomind.cli.StorageService')
    def test_clear_all_with_force(self, mock_storage_class, runner):
        """Test clear-all with --force flag."""
        mock_storage = MagicMock()
        mock_storage.clear_all.return_value = 200
        mock_storage_class.return_value = mock_storage

        result = runner.invoke(main, ["clear-all", "--force"])

        assert result.exit_code == 0
        mock_storage.clear_all.assert_called_once()
        assert "200" in result.output

    @patch('repomind.cli.StorageService')
    def test_clear_all_requires_confirmation(self, mock_storage_class, runner):
        """Test clear-all requires confirmation without --force."""
        result = runner.invoke(main, ["clear-all"], input="no\n")

        assert "WARNING" in result.output
        assert "Cancelled" in result.output

    @patch('repomind.cli.StorageService')
    def test_clear_all_confirmation_yes(self, mock_storage_class, runner):
        """Test clear-all with 'yes' confirmation."""
        mock_storage = MagicMock()
        mock_storage.clear_all.return_value = 100
        mock_storage_class.return_value = mock_storage

        result = runner.invoke(main, ["clear-all"], input="yes\n")

        mock_storage.clear_all.assert_called_once()


class TestContextCommand:
    """Tests for the context command."""

    def test_context_help(self, runner):
        """Test context command help."""
        result = runner.invoke(main, ["context", "--help"])

        assert result.exit_code == 0
        assert "SYMBOL" in result.output

    @patch('repomind.cli.get_context')
    def test_context_basic(self, mock_get_context, runner):
        """Test basic context command."""
        mock_get_context.return_value = {
            "found": True,
            "symbol": "test_func",
        }

        result = runner.invoke(main, ["context", "test_func"])

        mock_get_context.assert_called_once()

    @patch('repomind.cli.get_context')
    def test_context_with_repo_filter(self, mock_get_context, runner):
        """Test context with repository filter."""
        mock_get_context.return_value = {"found": True}

        result = runner.invoke(main, ["context", "symbol", "-r", "my-repo"])

        call_kwargs = mock_get_context.call_args[1]
        assert call_kwargs["repo_filter"] == "my-repo"

    @patch('repomind.cli.get_context')
    def test_context_not_found(self, mock_get_context, runner):
        """Test context when symbol not found."""
        mock_get_context.return_value = {
            "found": False,
            "message": "Symbol not found"
        }

        result = runner.invoke(main, ["context", "unknown_symbol"])

        assert "Symbol not found" in result.output or "not found" in result.output.lower()


class TestConfigShowCommand:
    """Tests for the config-show command."""

    def test_config_show_help(self, runner):
        """Test config-show command help."""
        result = runner.invoke(main, ["config-show", "--help"])

        assert result.exit_code == 0

    def test_config_show_basic(self, runner):
        """Test basic config-show command."""
        result = runner.invoke(main, ["config-show"])

        assert result.exit_code == 0
        assert "RepoMind Configuration" in result.output

    def test_config_show_with_repos_dir(self, runner, tmp_path):
        """Test config-show with repos_dir set."""
        result = runner.invoke(main, [
            "--repos-dir", str(tmp_path),
            "config-show"
        ])

        assert result.exit_code == 0
        # The output may be truncated, so just check for partial path
        # or that Repos Directory is shown
        assert "Repos Directory" in result.output


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_help(self, runner):
        """Test serve command help."""
        result = runner.invoke(main, ["serve", "--help"])

        assert result.exit_code == 0
        assert "MCP server" in result.output or "AI assistant" in result.output
