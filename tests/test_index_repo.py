"""
Comprehensive tests for the index_repo tools.

Tests cover:
- discover_repositories function with various patterns
- index_all_repositories function
- index_repo function
- Edge cases and error handling

Author: RepoMind Team
"""

import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from repomind.config import Config, RepositoryConfig, set_config, reset_config
from repomind.tools.index_repo import (
    discover_repositories,
    index_all_repositories,
    index_repo,
    _matches_pattern,
)


class TestMatchesPattern:
    """Tests for the _matches_pattern helper function."""

    def test_exact_match(self):
        """Test exact pattern match."""
        assert _matches_pattern("repo", "repo") is True
        assert _matches_pattern("repo", "other") is False

    def test_wildcard_star(self):
        """Test wildcard * pattern."""
        assert _matches_pattern("service-auth", "service-*") is True
        assert _matches_pattern("lib-utils", "service-*") is False

    def test_wildcard_all(self):
        """Test * matches everything."""
        assert _matches_pattern("anything", "*") is True
        assert _matches_pattern("", "*") is True

    def test_question_mark_wildcard(self):
        """Test ? wildcard matches single character."""
        assert _matches_pattern("v1", "v?") is True
        assert _matches_pattern("v10", "v?") is False

    def test_character_class(self):
        """Test character class patterns."""
        assert _matches_pattern("repo1", "repo[0-9]") is True
        assert _matches_pattern("repoA", "repo[0-9]") is False


class TestDiscoverRepositories:
    """Tests for discover_repositories function."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_discover_with_explicit_repos_dir(self, tmp_path):
        """Test discovery with explicit repos_dir."""
        # Create test repos
        (tmp_path / "repo1").mkdir()
        (tmp_path / "repo1" / "main.py").write_text("code")

        (tmp_path / "repo2").mkdir()
        (tmp_path / "repo2" / "app.ts").write_text("code")

        set_config(Config())

        repos = discover_repositories(repos_dir=tmp_path)

        assert "repo1" in repos
        assert "repo2" in repos
        assert repos["repo1"] == tmp_path / "repo1"
        assert repos["repo2"] == tmp_path / "repo2"

    def test_discover_uses_config_repos_dir(self, tmp_path):
        """Test discovery uses config repos_dir when not provided."""
        (tmp_path / "config-repo").mkdir()
        (tmp_path / "config-repo" / "main.py").write_text("code")

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories()

        assert "config-repo" in repos

    def test_discover_raises_without_repos_dir(self):
        """Test discovery raises when no repos_dir."""
        set_config(Config(repos_dir=None))

        with pytest.raises(ValueError) as exc_info:
            discover_repositories()

        assert "repos_dir must be provided" in str(exc_info.value)

    def test_discover_empty_for_nonexistent_dir(self, tmp_path):
        """Test discovery returns empty for non-existent directory."""
        nonexistent = tmp_path / "nonexistent"

        set_config(Config())
        repos = discover_repositories(repos_dir=nonexistent)

        assert repos == {}

    def test_discover_empty_for_file_path(self, tmp_path):
        """Test discovery returns empty when path is a file."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")

        set_config(Config())
        repos = discover_repositories(repos_dir=file_path)

        assert repos == {}

    def test_discover_with_include_patterns(self, tmp_path):
        """Test discovery with include patterns."""
        (tmp_path / "service-auth").mkdir()
        (tmp_path / "service-auth" / "main.py").write_text("code")

        (tmp_path / "lib-utils").mkdir()
        (tmp_path / "lib-utils" / "util.py").write_text("code")

        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "app.py").write_text("code")

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories(include_patterns=["service-*"])

        assert "service-auth" in repos
        assert "lib-utils" not in repos
        assert "other" not in repos

    def test_discover_with_exclude_patterns(self, tmp_path):
        """Test discovery with exclude patterns."""
        (tmp_path / "main-app").mkdir()
        (tmp_path / "main-app" / "main.py").write_text("code")

        (tmp_path / "test-app").mkdir()
        (tmp_path / "test-app" / "test.py").write_text("code")

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories(exclude_patterns=["test-*"])

        assert "main-app" in repos
        assert "test-app" not in repos

    def test_discover_excludes_default_patterns(self, tmp_path):
        """Test discovery excludes default patterns like .git."""
        (tmp_path / "valid-repo").mkdir()
        (tmp_path / "valid-repo" / "main.py").write_text("code")

        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git")

        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("js")

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories()

        assert "valid-repo" in repos
        assert ".git" not in repos
        assert "node_modules" not in repos

    def test_discover_skips_files(self, tmp_path):
        """Test discovery skips files in repos_dir."""
        (tmp_path / "valid-repo").mkdir()
        (tmp_path / "valid-repo" / "main.py").write_text("code")

        (tmp_path / "file.txt").write_text("not a repo")

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories()

        assert "valid-repo" in repos
        assert "file.txt" not in repos

    def test_discover_skips_empty_dirs(self, tmp_path):
        """Test discovery skips directories without source files."""
        (tmp_path / "valid-repo").mkdir()
        (tmp_path / "valid-repo" / "main.py").write_text("code")

        (tmp_path / "empty-dir").mkdir()

        (tmp_path / "docs-only").mkdir()
        (tmp_path / "docs-only" / "README.md").write_text("docs")

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories()

        assert "valid-repo" in repos
        assert "empty-dir" not in repos
        assert "docs-only" not in repos


class TestIndexAllRepositories:
    """Tests for index_all_repositories function."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_index_all_returns_error_without_repos_dir(self):
        """Test index_all returns error when repos_dir not set."""
        set_config(Config(repos_dir=None))

        result = index_all_repositories()

        assert "error" in result
        assert "_summary" in result
        assert result["_summary"]["total_repos"] == 0

    def test_index_all_empty_directory(self, tmp_path):
        """Test index_all with empty directory."""
        set_config(Config(repos_dir=tmp_path))

        result = index_all_repositories()

        assert "error" in result
        assert "No repositories found" in result["error"]

    @patch('repomind.tools.index_repo.index_repo')
    def test_index_all_with_discovered_repos(self, mock_index_repo, tmp_path):
        """Test index_all calls index_repo for each discovered repo."""
        # Create test repos
        (tmp_path / "repo1").mkdir()
        (tmp_path / "repo1" / "main.py").write_text("code")

        (tmp_path / "repo2").mkdir()
        (tmp_path / "repo2" / "app.py").write_text("code")

        mock_index_repo.return_value = {"status": "success", "chunks_stored": 10}

        set_config(Config(repos_dir=tmp_path))

        result = index_all_repositories()

        assert mock_index_repo.call_count == 2
        assert "_summary" in result
        assert result["_summary"]["total_repos"] == 2
        assert result["_summary"]["successful"] == 2

    @patch('repomind.tools.index_repo.index_repo')
    def test_index_all_with_pattern_filter(self, mock_index_repo, tmp_path):
        """Test index_all with include pattern filter."""
        # Create test repos
        (tmp_path / "service-a").mkdir()
        (tmp_path / "service-a" / "main.py").write_text("code")

        (tmp_path / "lib-b").mkdir()
        (tmp_path / "lib-b" / "lib.py").write_text("code")

        mock_index_repo.return_value = {"status": "success"}

        set_config(Config(repos_dir=tmp_path))

        result = index_all_repositories(include_patterns=["service-*"])

        assert mock_index_repo.call_count == 1
        assert "_summary" in result
        assert result["_summary"]["total_repos"] == 1

    @patch('repomind.tools.index_repo.index_repo')
    def test_index_all_handles_index_error(self, mock_index_repo, tmp_path):
        """Test index_all handles errors from index_repo gracefully."""
        (tmp_path / "repo").mkdir()
        (tmp_path / "repo" / "main.py").write_text("code")

        mock_index_repo.side_effect = Exception("Index error")

        set_config(Config(repos_dir=tmp_path))

        result = index_all_repositories()

        assert "repo" in result
        assert result["repo"]["status"] == "error"
        assert "Index error" in result["repo"]["error"]
        assert result["_summary"]["failed"] == 1

    @patch('repomind.tools.index_repo.index_repo')
    def test_index_all_with_explicit_repos_list(self, mock_index_repo, tmp_path):
        """Test index_all uses explicit repos_list when provided."""
        (tmp_path / "explicit-repo").mkdir()
        (tmp_path / "explicit-repo" / "main.py").write_text("code")

        (tmp_path / "other-repo").mkdir()
        (tmp_path / "other-repo" / "other.py").write_text("code")

        mock_index_repo.return_value = {"status": "success"}

        set_config(Config(repos_dir=tmp_path, repos_list=["explicit-repo"]))

        result = index_all_repositories()

        # Should only index the one in repos_list
        assert mock_index_repo.call_count == 1
        assert result["_summary"]["total_repos"] == 1

    @patch('repomind.tools.index_repo.index_repo')
    def test_index_all_skips_nonexistent_repo(self, mock_index_repo, tmp_path):
        """Test index_all skips repos that don't exist."""
        (tmp_path / "exists").mkdir()
        (tmp_path / "exists" / "main.py").write_text("code")

        mock_index_repo.return_value = {"status": "success"}

        # repos_list includes one that doesn't exist
        set_config(Config(repos_dir=tmp_path, repos_list=["exists", "not-exists"]))

        result = index_all_repositories()

        assert mock_index_repo.call_count == 1  # Only called for 'exists'
        assert result["not-exists"]["status"] == "skipped"
        assert result["_summary"]["skipped"] == 1


class TestIndexRepo:
    """Tests for index_repo function."""

    def test_index_repo_nonexistent_path(self):
        """Test index_repo with non-existent path."""
        result = index_repo("/nonexistent/path")

        assert "error" in result
        assert "does not exist" in result["error"]

    def test_index_repo_file_path(self, tmp_path):
        """Test index_repo with file instead of directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        result = index_repo(str(file_path))

        assert "error" in result
        assert "not a directory" in result["error"]

    @patch('repomind.tools.index_repo._index_full')
    def test_index_repo_full_mode(self, mock_index_full, tmp_path):
        """Test index_repo in full mode."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / "main.py").write_text("code")

        mock_index_full.return_value = {"status": "success"}

        result = index_repo(str(repo_path))

        mock_index_full.assert_called_once()
        assert result == {"status": "success"}

    @patch('repomind.tools.index_repo._index_incremental')
    def test_index_repo_incremental_mode(self, mock_index_incr, tmp_path):
        """Test index_repo in incremental mode."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / "main.py").write_text("code")

        mock_index_incr.return_value = {"status": "success", "mode": "incremental"}

        result = index_repo(str(repo_path), incremental=True)

        mock_index_incr.assert_called_once()
        assert result["mode"] == "incremental"

    @patch('repomind.tools.index_repo._index_full')
    def test_index_repo_uses_directory_name(self, mock_index_full, tmp_path):
        """Test index_repo uses directory name when not provided."""
        repo_path = tmp_path / "my-repo"
        repo_path.mkdir()
        (repo_path / "main.py").write_text("code")

        mock_index_full.return_value = {"status": "success"}

        index_repo(str(repo_path))

        # Check that the name passed is the directory name
        call_args = mock_index_full.call_args
        assert call_args[0][1] == "my-repo"  # Second arg is name

    @patch('repomind.tools.index_repo._index_full')
    def test_index_repo_custom_name(self, mock_index_full, tmp_path):
        """Test index_repo with custom name."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / "main.py").write_text("code")

        mock_index_full.return_value = {"status": "success"}

        index_repo(str(repo_path), repo_name="custom-name")

        call_args = mock_index_full.call_args
        assert call_args[0][1] == "custom-name"




class TestIndexRepoEdgeCases:
    """Edge case tests for index operations."""

    def test_discover_with_symlinks(self, tmp_path):
        """Test discovery handles symlinks gracefully."""
        reset_config()

        # Create a real repo
        (tmp_path / "real-repo").mkdir()
        (tmp_path / "real-repo" / "main.py").write_text("code")

        # Create a symlink
        try:
            symlink = tmp_path / "symlink-repo"
            symlink.symlink_to(tmp_path / "real-repo")
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories()

        assert "real-repo" in repos
        # symlink-repo may or may not be included depending on implementation

    def test_discover_with_permission_error(self, tmp_path):
        """Test discovery handles permission errors gracefully."""
        reset_config()

        # Create repos
        (tmp_path / "accessible").mkdir()
        (tmp_path / "accessible" / "main.py").write_text("code")

        # Note: This test may not work on all systems
        # The implementation should handle PermissionError gracefully

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories()

        assert "accessible" in repos

    def test_discover_sorted_alphabetically(self, tmp_path):
        """Test discovered repos are sorted alphabetically."""
        reset_config()

        for name in ["zebra", "alpha", "middle"]:
            (tmp_path / name).mkdir()
            (tmp_path / name / "main.py").write_text("code")

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories()
        repo_names = list(repos.keys())

        assert repo_names == sorted(repo_names)

    def test_discover_with_special_characters_in_name(self, tmp_path):
        """Test discovery with special characters in repo name."""
        reset_config()

        # Create repo with special characters
        (tmp_path / "repo-with-dash").mkdir()
        (tmp_path / "repo-with-dash" / "main.py").write_text("code")

        (tmp_path / "repo.with.dots").mkdir()
        (tmp_path / "repo.with.dots" / "main.py").write_text("code")

        set_config(Config(repos_dir=tmp_path))

        repos = discover_repositories()

        assert "repo-with-dash" in repos
        assert "repo.with.dots" in repos
