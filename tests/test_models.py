"""
Comprehensive tests for the chunk models.

Tests cover:
- CallInfo dataclass
- ParseResult dataclass
- ChunkType enum
- CodeChunk model with all methods
- Edge cases for embedding text generation
- Keyword extraction from various naming conventions

Author: RepoMind Team
"""

import pytest

from repomind.models.chunk import (
    CallInfo,
    ParseResult,
    ChunkType,
    CodeChunk,
)


class TestCallInfo:
    """Tests for CallInfo dataclass."""

    def test_basic_creation(self):
        """Test basic CallInfo creation."""
        call = CallInfo(
            caller_qualified_name="MyClass.my_method",
            callee_name="helper_function",
            caller_file="src/my_class.py",
            caller_line=25,
        )

        assert call.caller_qualified_name == "MyClass.my_method"
        assert call.callee_name == "helper_function"
        assert call.caller_file == "src/my_class.py"
        assert call.caller_line == 25
        assert call.call_type == "direct"  # default

    def test_with_custom_call_type(self):
        """Test CallInfo with custom call type."""
        call = CallInfo(
            caller_qualified_name="main",
            callee_name="MyClass",
            caller_file="main.py",
            caller_line=10,
            call_type="constructor",
        )

        assert call.call_type == "constructor"

    def test_method_call_type(self):
        """Test CallInfo for method calls."""
        call = CallInfo(
            caller_qualified_name="Service.handle",
            callee_name="self.validate",
            caller_file="service.py",
            caller_line=50,
            call_type="method",
        )

        assert call.call_type == "method"


class TestParseResult:
    """Tests for ParseResult dataclass."""

    def test_empty_parse_result(self):
        """Test empty ParseResult."""
        result = ParseResult()

        assert result.chunks == []
        assert result.calls == []

    def test_parse_result_with_data(self):
        """Test ParseResult with chunks and calls."""
        chunk = CodeChunk(
            id="test123",
            repo_name="test-repo",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_type=ChunkType.FUNCTION,
            name="test_func",
            content="def test_func(): pass",
            language="python",
        )

        call = CallInfo(
            caller_qualified_name="test_func",
            callee_name="print",
            caller_file="test.py",
            caller_line=5,
        )

        result = ParseResult(chunks=[chunk], calls=[call])

        assert len(result.chunks) == 1
        assert len(result.calls) == 1
        assert result.chunks[0].name == "test_func"
        assert result.calls[0].callee_name == "print"


class TestChunkType:
    """Tests for ChunkType enum."""

    def test_all_chunk_types_exist(self):
        """Test all expected chunk types exist."""
        expected_types = [
            "module", "class", "interface", "function",
            "method", "constructor", "property", "constant",
            "import", "comment", "documentation"
        ]

        for type_name in expected_types:
            assert hasattr(ChunkType, type_name.upper())
            assert ChunkType(type_name).value == type_name

    def test_chunk_type_values(self):
        """Test ChunkType enum values."""
        assert ChunkType.FUNCTION.value == "function"
        assert ChunkType.CLASS.value == "class"
        assert ChunkType.METHOD.value == "method"

    def test_invalid_chunk_type_raises(self):
        """Test invalid chunk type raises ValueError."""
        with pytest.raises(ValueError):
            ChunkType("invalid_type")


class TestCodeChunk:
    """Tests for CodeChunk model."""

    @pytest.fixture
    def basic_chunk(self):
        """Create a basic chunk for testing."""
        return CodeChunk(
            id="abc123",
            repo_name="test-repo",
            file_path="src/handler.py",
            start_line=10,
            end_line=20,
            chunk_type=ChunkType.FUNCTION,
            name="handle_request",
            content="def handle_request(): pass",
            language="python",
        )

    @pytest.fixture
    def full_chunk(self):
        """Create a fully populated chunk for testing."""
        return CodeChunk(
            id="full123",
            repo_name="MyService",
            file_path="src/services/auth.py",
            start_line=45,
            end_line=67,
            chunk_type=ChunkType.METHOD,
            name="validate_token",
            content="def validate_token(self, token: str) -> bool:\n    return True",
            signature="def validate_token(self, token: str) -> bool",
            docstring="Validate a JWT token and return validity status.",
            parent_name="AuthService",
            parent_type=ChunkType.CLASS,
            language="python",
            imports=["import jwt", "from typing import Optional"],
            summary="Token validation method",
        )

    def test_basic_chunk_creation(self, basic_chunk):
        """Test basic chunk creation."""
        assert basic_chunk.id == "abc123"
        assert basic_chunk.repo_name == "test-repo"
        assert basic_chunk.name == "handle_request"
        assert basic_chunk.chunk_type == ChunkType.FUNCTION

    def test_optional_fields_default_to_none(self, basic_chunk):
        """Test optional fields default to None."""
        assert basic_chunk.signature is None
        assert basic_chunk.docstring is None
        assert basic_chunk.parent_name is None
        assert basic_chunk.parent_type is None
        assert basic_chunk.summary is None

    def test_imports_default_to_empty_list(self, basic_chunk):
        """Test imports default to empty list."""
        assert basic_chunk.imports == []

    def test_full_chunk_creation(self, full_chunk):
        """Test fully populated chunk."""
        assert full_chunk.docstring == "Validate a JWT token and return validity status."
        assert full_chunk.parent_name == "AuthService"
        assert full_chunk.parent_type == ChunkType.CLASS
        assert len(full_chunk.imports) == 2

    def test_get_qualified_name_with_parent(self, full_chunk):
        """Test qualified name with parent class."""
        assert full_chunk.get_qualified_name() == "AuthService.validate_token"

    def test_get_qualified_name_without_parent(self, basic_chunk):
        """Test qualified name without parent."""
        assert basic_chunk.get_qualified_name() == "handle_request"

    def test_to_embedding_text_basic(self, basic_chunk):
        """Test basic embedding text generation."""
        text = basic_chunk.to_embedding_text()

        assert "handle_request" in text.lower() or "handle" in text.lower()
        assert "handler.py" in text
        assert "python" in text.lower()
        assert "def handle_request(): pass" in text

    def test_to_embedding_text_with_docstring(self, full_chunk):
        """Test embedding text includes docstring."""
        text = full_chunk.to_embedding_text()

        assert "Validate a JWT token" in text

    def test_to_embedding_text_with_signature(self, full_chunk):
        """Test embedding text includes signature."""
        text = full_chunk.to_embedding_text()

        assert "validate_token(self, token: str)" in text

    def test_to_embedding_text_with_summary(self, full_chunk):
        """Test embedding text includes AI-generated summary."""
        text = full_chunk.to_embedding_text()

        assert "Token validation method" in text

    def test_to_embedding_text_long_content_truncation(self):
        """Test that very long content is truncated."""
        long_content = "x = 1\n" * 1000  # Very long content

        chunk = CodeChunk(
            id="long123",
            repo_name="test-repo",
            file_path="long.py",
            start_line=1,
            end_line=1000,
            chunk_type=ChunkType.FUNCTION,
            name="long_function",
            content=long_content,
            language="python",
        )

        text = chunk.to_embedding_text()

        # Should be truncated and contain marker
        assert "truncated" in text.lower()

    def test_extract_keywords_from_snake_case(self):
        """Test keyword extraction from snake_case names."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="validate_user_input",
            content="pass",
            language="python",
        )

        keywords = chunk._extract_keywords_from_name()

        assert "validate" in keywords
        assert "user" in keywords
        assert "input" in keywords

    def test_extract_keywords_from_camelCase(self):
        """Test keyword extraction from camelCase names."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.ts",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="validateUserInput",
            content="pass",
            language="typescript",
        )

        keywords = chunk._extract_keywords_from_name()

        assert "validate" in keywords
        assert "user" in keywords
        assert "input" in keywords

    def test_extract_keywords_from_PascalCase(self):
        """Test keyword extraction from PascalCase names."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.java",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.CLASS,
            name="UserAuthenticationService",
            content="class UserAuthenticationService {}",
            language="java",
        )

        keywords = chunk._extract_keywords_from_name()

        assert "user" in keywords
        assert "authentication" in keywords
        assert "service" in keywords

    def test_extract_keywords_filters_common_words(self):
        """Test that common words are filtered out."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="get_the_user",
            content="pass",
            language="python",
        )

        keywords = chunk._extract_keywords_from_name()

        # 'get' and 'the' should be filtered
        assert "get" not in keywords
        assert "the" not in keywords
        assert "user" in keywords

    def test_extract_keywords_filters_short_words(self):
        """Test that single-character words are filtered."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="a_b_calculate",
            content="pass",
            language="python",
        )

        keywords = chunk._extract_keywords_from_name()

        # 'a' and 'b' should be filtered
        assert "a" not in keywords
        assert "b" not in keywords
        assert "calculate" in keywords

    def test_generate_purpose_description_function(self):
        """Test purpose description for function."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="validate_input",
            content="pass",
            language="python",
        )

        desc = chunk._generate_purpose_description()

        assert "Function" in desc
        assert "validate" in desc.lower()

    def test_generate_purpose_description_method_with_parent(self):
        """Test purpose description for method with parent."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.METHOD,
            name="process_data",
            content="pass",
            language="python",
            parent_name="DataProcessor",
        )

        desc = chunk._generate_purpose_description()

        assert "Method" in desc
        assert "DataProcessor" in desc

    def test_generate_purpose_description_class(self):
        """Test purpose description for class."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.CLASS,
            name="UserService",
            content="class UserService: pass",
            language="python",
        )

        desc = chunk._generate_purpose_description()

        assert "Class" in desc
        assert "user" in desc.lower()

    def test_generate_purpose_description_interface(self):
        """Test purpose description for interface."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.ts",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.INTERFACE,
            name="IUserRepository",
            content="interface IUserRepository {}",
            language="typescript",
        )

        desc = chunk._generate_purpose_description()

        assert "Interface" in desc

    def test_generate_purpose_description_constructor(self):
        """Test purpose description for constructor."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.CONSTRUCTOR,
            name="__init__",
            content="def __init__(self): pass",
            language="python",
            parent_name="MyClass",
        )

        desc = chunk._generate_purpose_description()

        assert "Constructor" in desc
        assert "MyClass" in desc


class TestCodeChunkEdgeCases:
    """Edge case tests for CodeChunk."""

    def test_empty_name(self):
        """Test chunk with empty name."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="",
            content="pass",
            language="python",
        )

        keywords = chunk._extract_keywords_from_name()
        assert keywords == []

    def test_name_with_numbers(self):
        """Test keyword extraction with numbers in name."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="process_v2_data",
            content="pass",
            language="python",
        )

        keywords = chunk._extract_keywords_from_name()

        assert "process" in keywords
        assert "data" in keywords

    def test_name_with_screaming_snake_case(self):
        """Test keyword extraction from SCREAMING_SNAKE_CASE."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.CONSTANT,
            name="MAX_RETRY_COUNT",
            content="MAX_RETRY_COUNT = 3",
            language="python",
        )

        keywords = chunk._extract_keywords_from_name()

        assert "max" in keywords
        assert "retry" in keywords
        assert "count" in keywords

    def test_file_path_without_slash(self):
        """Test embedding text with simple file path."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="main.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="main",
            content="pass",
            language="python",
        )

        text = chunk.to_embedding_text()

        assert "main.py" in text

    def test_unicode_in_content(self):
        """Test chunk with unicode characters."""
        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            name="greet",
            content="def greet(): return '你好世界'",
            docstring="Greet in Chinese: 你好",
            language="python",
        )

        text = chunk.to_embedding_text()

        assert "你好" in text

    def test_multiline_docstring(self):
        """Test chunk with multiline docstring."""
        docstring = """This is a multiline docstring.
        
        It has multiple paragraphs.
        
        Args:
            x: Something
        """

        chunk = CodeChunk(
            id="test",
            repo_name="test",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_type=ChunkType.FUNCTION,
            name="documented_func",
            content="def documented_func(x): pass",
            docstring=docstring,
            language="python",
        )

        text = chunk.to_embedding_text()

        assert "multiline docstring" in text
        assert "multiple paragraphs" in text
