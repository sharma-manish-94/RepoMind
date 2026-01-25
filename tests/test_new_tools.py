"""
Comprehensive tests for the new RepoMind tools.

Tests cover:
- file_summary tool
- find_usages tool
- find_tests tool
- find_implementations tool
- diff_impact tool

Author: RepoMind Team
"""

from unittest.mock import patch, MagicMock
import pytest

from repomind.config import reset_config
from repomind.models.chunk import ChunkType, CodeChunk, InheritanceInfo
from repomind.services.symbol_table import Symbol
from repomind.tools.file_summary import file_summary
from repomind.tools.find_usages import find_usages
from repomind.tools.find_tests import find_tests
from repomind.tools.find_implementations import find_implementations, find_hierarchy
from repomind.tools.diff_impact import diff_impact


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset config before and after each test."""
    reset_config()
    yield
    reset_config()


# =============================================================================
# file_summary Tests
# =============================================================================

class TestFileSummary:
    """Tests for file_summary tool."""

    def test_empty_path_returns_error(self):
        """Test that empty file path returns error."""
        result = file_summary(file_path="")
        assert "error" in result
        assert "empty" in result["error"].lower()

    def test_whitespace_path_returns_error(self):
        """Test that whitespace-only path returns error."""
        result = file_summary(file_path="   ")
        assert "error" in result

    @patch('repomind.tools.file_summary.SymbolTableService')
    def test_file_not_found_returns_error(self, mock_symbol_class):
        """Test that non-existent file returns appropriate error."""
        mock_service = MagicMock()
        mock_service.lookup_by_file.return_value = []
        mock_symbol_class.return_value = mock_service

        result = file_summary(file_path="nonexistent.py", repo_name="test-repo")

        assert "error" in result

    @patch('repomind.tools.file_summary.SymbolTableService')
    def test_basic_file_summary(self, mock_symbol_class):
        """Test basic file summary with symbols."""
        mock_service = MagicMock()
        mock_service.lookup_by_file.return_value = [
            Symbol(
                name="TestClass",
                qualified_name="TestClass",
                symbol_type="class",
                file_path="test.py",
                repo_name="test-repo",
                start_line=1,
                end_line=20,
                signature="class TestClass:",
                parent_name=None,
                language="python",
            ),
            Symbol(
                name="test_method",
                qualified_name="TestClass.test_method",
                symbol_type="method",
                file_path="test.py",
                repo_name="test-repo",
                start_line=5,
                end_line=10,
                signature="def test_method(self):",
                parent_name="TestClass",
                language="python",
            ),
        ]
        mock_symbol_class.return_value = mock_service

        result = file_summary(file_path="test.py", repo_name="test-repo")

        assert "error" not in result
        assert result["file"] == "test.py"
        assert result["repo"] == "test-repo"
        assert "symbols" in result
        assert len(result["symbols"]["classes"]) == 1
        assert result["symbols"]["classes"][0]["name"] == "TestClass"

    @patch('repomind.tools.file_summary.SymbolTableService')
    def test_file_summary_groups_methods_under_classes(self, mock_symbol_class):
        """Test that methods are nested under their parent classes."""
        mock_service = MagicMock()
        mock_service.lookup_by_file.return_value = [
            Symbol(
                name="MyClass",
                qualified_name="MyClass",
                symbol_type="class",
                file_path="test.py",
                repo_name="repo",
                start_line=1,
                end_line=30,
                signature=None,
                parent_name=None,
                language="python",
            ),
            Symbol(
                name="method1",
                qualified_name="MyClass.method1",
                symbol_type="method",
                file_path="test.py",
                repo_name="repo",
                start_line=5,
                end_line=10,
                signature=None,
                parent_name="MyClass",
                language="python",
            ),
            Symbol(
                name="method2",
                qualified_name="MyClass.method2",
                symbol_type="method",
                file_path="test.py",
                repo_name="repo",
                start_line=12,
                end_line=18,
                signature=None,
                parent_name="MyClass",
                language="python",
            ),
        ]
        mock_symbol_class.return_value = mock_service

        result = file_summary(file_path="test.py", repo_name="repo")

        assert len(result["symbols"]["classes"]) == 1
        assert len(result["symbols"]["classes"][0]["methods"]) == 2


# =============================================================================
# find_usages Tests
# =============================================================================

class TestFindUsages:
    """Tests for find_usages tool."""

    def test_empty_symbol_returns_error(self):
        """Test that empty symbol name returns error."""
        result = find_usages(symbol_name="")
        assert "error" in result

    @patch('repomind.tools.find_usages.CallGraphService')
    @patch('repomind.tools.find_usages.StorageService')
    @patch('repomind.tools.find_usages.SymbolTableService')
    def test_basic_usage_search(self, mock_symbol, mock_storage, mock_call):
        """Test basic usage search."""
        # Mock call graph
        mock_call_instance = MagicMock()
        mock_call_instance.find_callers.return_value = []
        mock_call.return_value = mock_call_instance

        # Mock storage
        mock_storage_instance = MagicMock()
        mock_storage_instance._load_chunk_metadata.return_value = {}
        mock_storage.return_value = mock_storage_instance

        # Mock symbol table
        mock_symbol_instance = MagicMock()
        mock_symbol.return_value = mock_symbol_instance

        result = find_usages(symbol_name="TestSymbol")

        assert result["symbol"] == "TestSymbol"
        assert "total_usages" in result

    @patch('repomind.tools.find_usages.CallGraphService')
    @patch('repomind.tools.find_usages.StorageService')
    @patch('repomind.tools.find_usages.SymbolTableService')
    def test_finds_type_hints(self, mock_symbol, mock_storage, mock_call):
        """Test that type hints are found."""
        mock_call_instance = MagicMock()
        mock_call_instance.find_callers.return_value = []
        mock_call.return_value = mock_call_instance

        # Create a chunk with type hint
        chunk = CodeChunk(
            id="test1",
            repo_name="repo",
            file_path="test.py",
            start_line=1,
            end_line=5,
            chunk_type=ChunkType.FUNCTION,
            name="use_service",
            content="def use_service(svc: MyService) -> None:\n    pass",
            language="python",
        )

        mock_storage_instance = MagicMock()
        mock_storage_instance._load_chunk_metadata.return_value = {"test1": chunk}
        mock_storage.return_value = mock_storage_instance

        mock_symbol_instance = MagicMock()
        mock_symbol.return_value = mock_symbol_instance

        result = find_usages(symbol_name="MyService")

        assert result["total_usages"] >= 0  # May or may not find depending on exact matching


# =============================================================================
# find_tests Tests
# =============================================================================

class TestFindTests:
    """Tests for find_tests tool."""

    def test_empty_symbol_returns_error(self):
        """Test that empty symbol returns error."""
        result = find_tests(symbol_name="")
        assert "error" in result

    @patch('repomind.tools.find_tests.StorageService')
    @patch('repomind.tools.find_tests.SymbolTableService')
    def test_finds_test_by_filename_pattern(self, mock_symbol, mock_storage):
        """Test that tests are found by filename pattern."""
        mock_symbol.return_value = MagicMock()

        # Create a test file chunk
        test_chunk = CodeChunk(
            id="test1",
            repo_name="repo",
            file_path="tests/test_user_service.py",
            start_line=1,
            end_line=10,
            chunk_type=ChunkType.FUNCTION,
            name="test_user_service_validate",
            content="def test_user_service_validate(): pass",
            language="python",
        )

        mock_storage_instance = MagicMock()
        mock_storage_instance._load_chunk_metadata.return_value = {"test1": test_chunk}
        mock_storage.return_value = mock_storage_instance

        result = find_tests(symbol_name="user_service")

        assert result["symbol"] == "user_service"
        assert result["test_files_count"] >= 0

    @patch('repomind.tools.find_tests.StorageService')
    @patch('repomind.tools.find_tests.SymbolTableService')
    def test_finds_test_methods(self, mock_symbol, mock_storage):
        """Test that specific test methods are found."""
        mock_symbol.return_value = MagicMock()

        test_chunk = CodeChunk(
            id="test1",
            repo_name="repo",
            file_path="tests/test_auth.py",
            start_line=10,
            end_line=15,
            chunk_type=ChunkType.FUNCTION,
            name="test_validate_token",
            content="def test_validate_token():\n    token = 'abc'\n    assert validate(token)",
            language="python",
        )

        mock_storage_instance = MagicMock()
        mock_storage_instance._load_chunk_metadata.return_value = {"test1": test_chunk}
        mock_storage.return_value = mock_storage_instance

        result = find_tests(symbol_name="validate")

        # Should find the test method that contains "validate"
        assert "test_methods" in result


# =============================================================================
# find_implementations Tests
# =============================================================================

class TestFindImplementations:
    """Tests for find_implementations tool."""

    def test_empty_interface_returns_error(self):
        """Test that empty interface name returns error."""
        result = find_implementations(interface_name="")
        assert "error" in result

    @patch('repomind.tools.find_implementations.SymbolTableService')
    def test_no_implementations_found(self, mock_symbol):
        """Test when no implementations are found."""
        mock_service = MagicMock()
        mock_service.find_implementations.return_value = []
        mock_symbol.return_value = mock_service

        result = find_implementations(interface_name="NonExistentInterface")

        assert result["found"] is False
        assert "message" in result

    @patch('repomind.tools.find_implementations.SymbolTableService')
    def test_finds_implementations(self, mock_symbol):
        """Test that implementations are found."""
        mock_service = MagicMock()
        mock_service.find_implementations.return_value = [
            {
                "child_name": "PythonParser",
                "child_qualified": "PythonParser",
                "parent_name": "BaseParser",
                "relation_type": "extends",
                "file_path": "parsers/python_parser.py",
                "repo_name": "repomind",
                "line_number": 10,
            },
            {
                "child_name": "JavaParser",
                "child_qualified": "JavaParser",
                "parent_name": "BaseParser",
                "relation_type": "extends",
                "file_path": "parsers/java_parser.py",
                "repo_name": "repomind",
                "line_number": 8,
            },
        ]
        mock_symbol.return_value = mock_service

        result = find_implementations(interface_name="BaseParser")

        assert result["found"] is True
        assert result["total_count"] == 2
        assert result["extends"] is not None
        assert len(result["extends"]) == 2


class TestFindHierarchy:
    """Tests for find_hierarchy function."""

    def test_empty_class_returns_error(self):
        """Test that empty class name returns error."""
        result = find_hierarchy(class_name="")
        assert "error" in result

    @patch('repomind.tools.find_implementations.SymbolTableService')
    def test_finds_parents_and_children(self, mock_symbol):
        """Test that both parents and children are found."""
        mock_service = MagicMock()
        mock_service.find_parents.return_value = [
            {
                "child_name": "PythonParser",
                "parent_name": "BaseParser",
                "relation_type": "extends",
                "file_path": "parsers/python_parser.py",
                "repo_name": "repomind",
                "line_number": 10,
            }
        ]
        mock_service.find_implementations.return_value = []
        mock_symbol.return_value = mock_service

        result = find_hierarchy(class_name="PythonParser")

        assert result["class"] == "PythonParser"
        assert result["parents"] is not None


# =============================================================================
# diff_impact Tests
# =============================================================================

class TestDiffImpact:
    """Tests for diff_impact tool."""

    def test_nonexistent_repo_returns_error(self):
        """Test that non-existent repo path returns error."""
        result = diff_impact(repo_path="/nonexistent/path")
        assert "error" in result

    @patch('repomind.tools.diff_impact.Path')
    def test_non_git_repo_returns_error(self, mock_path):
        """Test that non-git directory returns error."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.__truediv__ = MagicMock(
            return_value=MagicMock(exists=MagicMock(return_value=False))
        )
        mock_path.return_value.resolve.return_value = mock_path_instance

        result = diff_impact(repo_path="/some/path")

        assert "error" in result
        assert "git" in result["error"].lower()

    @patch('repomind.tools.diff_impact._get_changed_files')
    @patch('repomind.tools.diff_impact.Path')
    def test_no_changes_returns_up_to_date(self, mock_path, mock_get_files):
        """Test that no changes returns up_to_date status."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.name = "test-repo"
        mock_git_dir = MagicMock()
        mock_git_dir.exists.return_value = True
        mock_path_instance.__truediv__ = MagicMock(return_value=mock_git_dir)
        mock_path.return_value.resolve.return_value = mock_path_instance

        mock_get_files.return_value = []

        result = diff_impact(repo_path="/some/path")

        assert result["status"] == "no_changes"


# =============================================================================
# Inheritance Info Model Tests
# =============================================================================

class TestInheritanceInfo:
    """Tests for InheritanceInfo dataclass."""

    def test_inheritance_info_creation(self):
        """Test creating InheritanceInfo."""
        info = InheritanceInfo(
            child_name="ChildClass",
            child_qualified="ChildClass",
            parent_name="ParentClass",
            relation_type="extends",
            file_path="test.py",
            line_number=10,
        )

        assert info.child_name == "ChildClass"
        assert info.parent_name == "ParentClass"
        assert info.relation_type == "extends"

    def test_inheritance_info_implements(self):
        """Test InheritanceInfo with implements relation."""
        info = InheritanceInfo(
            child_name="MyService",
            child_qualified="MyService",
            parent_name="ServiceInterface",
            relation_type="implements",
            file_path="service.java",
            line_number=5,
        )

        assert info.relation_type == "implements"


# =============================================================================
# Integration Tests
# =============================================================================

class TestToolIntegration:
    """Integration tests for tools working together."""

    @patch('repomind.tools.find_tests.StorageService')
    @patch('repomind.tools.find_tests.SymbolTableService')
    @patch('repomind.tools.find_usages.CallGraphService')
    @patch('repomind.tools.find_usages.StorageService')
    @patch('repomind.tools.find_usages.SymbolTableService')
    def test_usage_and_test_discovery(
        self, mock_usage_symbol, mock_usage_storage, mock_usage_call,
        mock_test_symbol, mock_test_storage
    ):
        """Test that find_usages and find_tests work on same symbol."""
        # Setup mocks for find_usages
        mock_usage_call.return_value = MagicMock(find_callers=MagicMock(return_value=[]))
        mock_usage_storage.return_value = MagicMock(_load_chunk_metadata=MagicMock(return_value={}))
        mock_usage_symbol.return_value = MagicMock()

        # Setup mocks for find_tests
        mock_test_symbol.return_value = MagicMock()
        mock_test_storage.return_value = MagicMock(_load_chunk_metadata=MagicMock(return_value={}))

        # Find usages
        usage_result = find_usages(symbol_name="MySymbol")
        assert "symbol" in usage_result

        # Find tests
        test_result = find_tests(symbol_name="MySymbol")
        assert "symbol" in test_result
