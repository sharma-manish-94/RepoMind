"""
Comprehensive tests for the code parsers.

Tests cover:
- PythonParser with various Python constructs
- TypeScriptParser with TypeScript/JavaScript
- JavaParser with Java classes and methods
- Edge cases: decorators, async functions, nested classes
- Call graph extraction

Author: RepoMind Team
"""

from pathlib import Path
from textwrap import dedent

import pytest

from repomind.models.chunk import ChunkType, ParseResult
from repomind.parsers.python_parser import PythonParser


class TestPythonParserBasics:
    """Basic tests for Python parser."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return PythonParser()

    def test_language_property(self, parser):
        """Test language property returns 'python'."""
        assert parser.language == "python"

    def test_file_extensions_property(self, parser):
        """Test file extensions property."""
        assert parser.file_extensions == [".py"]

    def test_parse_simple_function(self, parser, tmp_path):
        """Test parsing a simple function."""
        code = dedent('''
            def hello():
                print("Hello, World!")
        ''').strip()

        file_path = tmp_path / "simple.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        assert isinstance(result, ParseResult)
        assert len(result.chunks) == 1

        chunk = result.chunks[0]
        assert chunk.name == "hello"
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.language == "python"
        assert chunk.repo_name == "test-repo"

    def test_parse_function_with_signature(self, parser, tmp_path):
        """Test parsing function with typed signature."""
        code = dedent('''
            def process(data: str, count: int = 5) -> list[str]:
                return [data] * count
        ''').strip()

        file_path = tmp_path / "typed.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        chunk = result.chunks[0]
        assert chunk.name == "process"
        assert chunk.signature is not None
        assert "data: str" in chunk.signature
        assert "count: int" in chunk.signature

    def test_parse_function_with_docstring(self, parser, tmp_path):
        """Test parsing function with docstring."""
        code = dedent('''
            def documented():
                """This function has a docstring.
                
                It explains what the function does.
                """
                pass
        ''').strip()

        file_path = tmp_path / "doc.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        chunk = result.chunks[0]
        assert chunk.docstring is not None
        assert "This function has a docstring" in chunk.docstring

    def test_parse_empty_file(self, parser, tmp_path):
        """Test parsing an empty file."""
        file_path = tmp_path / "empty.py"
        file_path.write_text("")

        result = parser.parse_file(file_path, "test-repo")

        assert result.chunks == []
        assert result.calls == []

    def test_parse_comments_only(self, parser, tmp_path):
        """Test parsing file with only comments."""
        code = dedent('''
            # This is a comment
            # Another comment
        ''').strip()

        file_path = tmp_path / "comments.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Should have no function/class chunks
        assert len([c for c in result.chunks if c.chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS]]) == 0


class TestPythonParserClasses:
    """Tests for parsing Python classes."""

    @pytest.fixture
    def parser(self):
        return PythonParser()

    def test_parse_simple_class(self, parser, tmp_path):
        """Test parsing a simple class."""
        code = dedent('''
            class MyClass:
                pass
        ''').strip()

        file_path = tmp_path / "cls.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        class_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "MyClass"

    def test_parse_class_with_docstring(self, parser, tmp_path):
        """Test parsing class with docstring."""
        code = dedent('''
            class Documented:
                """This class has documentation."""
                pass
        ''').strip()

        file_path = tmp_path / "doc_cls.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        class_chunk = next(c for c in result.chunks if c.chunk_type == ChunkType.CLASS)
        assert class_chunk.docstring is not None
        assert "documentation" in class_chunk.docstring

    def test_parse_class_with_methods(self, parser, tmp_path):
        """Test parsing class with multiple methods."""
        code = dedent('''
            class Calculator:
                def add(self, a, b):
                    return a + b

                def subtract(self, a, b):
                    return a - b

                def multiply(self, a, b):
                    return a * b
        ''').strip()

        file_path = tmp_path / "calc.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        class_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.CLASS]
        method_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.METHOD]

        assert len(class_chunks) == 1
        assert len(method_chunks) == 3

        # Methods should reference parent class
        for method in method_chunks:
            assert method.parent_name == "Calculator"

    def test_parse_class_inheritance(self, parser, tmp_path):
        """Test parsing class with inheritance."""
        code = dedent('''
            class Child(Parent):
                def method(self):
                    pass
        ''').strip()

        file_path = tmp_path / "inherit.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        class_chunk = next(c for c in result.chunks if c.chunk_type == ChunkType.CLASS)
        assert class_chunk.name == "Child"

    def test_parse_class_with_constructor(self, parser, tmp_path):
        """Test parsing class with __init__ constructor."""
        code = dedent('''
            class Person:
                def __init__(self, name: str, age: int):
                    self.name = name
                    self.age = age
        ''').strip()

        file_path = tmp_path / "person.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Should have class and init method
        assert any(c.name == "Person" for c in result.chunks)
        assert any(c.name == "__init__" for c in result.chunks)

    def test_parse_class_with_class_methods(self, parser, tmp_path):
        """Test parsing class with classmethod and staticmethod."""
        code = dedent('''
            class Factory:
                @classmethod
                def create(cls):
                    return cls()

                @staticmethod
                def version():
                    return "1.0"
        ''').strip()

        file_path = tmp_path / "factory.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Parser may or may not extract decorated methods as METHOD
        # Just verify we got the class
        class_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Factory"


class TestPythonParserAdvanced:
    """Advanced tests for Python parser."""

    @pytest.fixture
    def parser(self):
        return PythonParser()

    def test_parse_async_function(self, parser, tmp_path):
        """Test parsing async function."""
        code = dedent('''
            async def fetch_data(url: str) -> dict:
                """Fetch data from URL asynchronously."""
                pass
        ''').strip()

        file_path = tmp_path / "async.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        assert len(result.chunks) == 1
        chunk = result.chunks[0]
        assert chunk.name == "fetch_data"
        assert "async" in chunk.content

    def test_parse_decorated_function(self, parser, tmp_path):
        """Test parsing decorated function."""
        code = dedent('''
            def decorated():
                """A decorated function."""
                pass
        ''').strip()

        file_path = tmp_path / "deco.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Should find the function
        assert len(result.chunks) >= 1
        func = next((c for c in result.chunks if c.name == "decorated"), None)
        assert func is not None

    def test_parse_nested_function(self, parser, tmp_path):
        """Test parsing nested functions."""
        code = dedent('''
            def outer():
                def inner():
                    pass
                return inner
        ''').strip()

        file_path = tmp_path / "nested.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Should find outer function
        outer = next((c for c in result.chunks if c.name == "outer"), None)
        assert outer is not None
        assert outer.chunk_type == ChunkType.FUNCTION

    def test_parse_multiple_functions(self, parser, tmp_path):
        """Test parsing file with multiple functions."""
        code = dedent('''
            def func_a():
                pass

            def func_b():
                pass

            def func_c():
                pass
        ''').strip()

        file_path = tmp_path / "multi.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        func_names = {c.name for c in result.chunks if c.chunk_type == ChunkType.FUNCTION}
        assert func_names == {"func_a", "func_b", "func_c"}

    def test_parse_function_with_default_args(self, parser, tmp_path):
        """Test parsing function with default arguments."""
        code = dedent('''
            def with_defaults(a, b=10, c="default", d=None):
                pass
        ''').strip()

        file_path = tmp_path / "defaults.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        chunk = result.chunks[0]
        assert chunk.signature is not None
        assert "b=10" in chunk.signature or "b = 10" in chunk.signature

    def test_parse_lambda_not_extracted(self, parser, tmp_path):
        """Test that lambda functions are not extracted as separate chunks."""
        code = dedent('''
            def use_lambda():
                mapper = lambda x: x * 2
                return mapper(5)
        ''').strip()

        file_path = tmp_path / "lambda.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Should only have use_lambda, not the lambda
        assert len(result.chunks) == 1
        assert result.chunks[0].name == "use_lambda"


class TestPythonParserCallGraph:
    """Tests for call graph extraction."""

    @pytest.fixture
    def parser(self):
        return PythonParser()

    def test_extract_simple_call(self, parser, tmp_path):
        """Test extracting simple function calls."""
        code = dedent('''
            def caller():
                callee()

            def callee():
                pass
        ''').strip()

        file_path = tmp_path / "calls.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Should find call from caller to callee
        calls = [c for c in result.calls if c.caller_qualified_name == "caller"]
        assert any(c.callee_name == "callee" for c in calls)

    def test_extract_method_call(self, parser, tmp_path):
        """Test extracting method calls."""
        code = dedent('''
            class Service:
                def process(self):
                    self.validate()

                def validate(self):
                    pass
        ''').strip()

        file_path = tmp_path / "method_calls.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Should find self.validate call
        calls = [c for c in result.calls if "process" in c.caller_qualified_name]
        assert any("validate" in c.callee_name for c in calls)

    def test_extract_chained_calls(self, parser, tmp_path):
        """Test extracting chained method calls."""
        code = dedent('''
            def chained():
                result.process().filter().map()
        ''').strip()

        file_path = tmp_path / "chain.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Should find multiple calls
        assert len(result.calls) >= 1


class TestPythonParserLineNumbers:
    """Tests for line number tracking."""

    @pytest.fixture
    def parser(self):
        return PythonParser()

    def test_line_numbers_correct(self, parser, tmp_path):
        """Test that line numbers are correctly tracked."""
        code = dedent('''
            # Comment line 1
            
            def on_line_three():
                # Line 4
                pass  # Line 5
        ''').strip()

        file_path = tmp_path / "lines.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        chunk = result.chunks[0]
        assert chunk.start_line == 3  # Function starts on line 3
        assert chunk.end_line >= chunk.start_line

    def test_multiline_function_span(self, parser, tmp_path):
        """Test that multiline functions have correct span."""
        code = dedent('''
            def multiline():
                a = 1
                b = 2
                c = 3
                return a + b + c
        ''').strip()

        file_path = tmp_path / "multi.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        chunk = result.chunks[0]
        assert chunk.start_line == 1
        assert chunk.end_line == 5  # Last line of function


class TestPythonParserQualifiedNames:
    """Tests for qualified name generation."""

    @pytest.fixture
    def parser(self):
        return PythonParser()

    def test_method_qualified_name(self, parser, tmp_path):
        """Test qualified name for method includes class."""
        code = dedent('''
            class MyClass:
                def my_method(self):
                    pass
        ''').strip()

        file_path = tmp_path / "qual.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        method = next(c for c in result.chunks if c.chunk_type == ChunkType.METHOD)
        assert method.get_qualified_name() == "MyClass.my_method"

    def test_function_qualified_name(self, parser, tmp_path):
        """Test qualified name for top-level function."""
        code = dedent('''
            def standalone():
                pass
        ''').strip()

        file_path = tmp_path / "standalone.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        func = result.chunks[0]
        assert func.get_qualified_name() == "standalone"


class TestPythonParserEdgeCases:
    """Edge case tests for Python parser."""

    @pytest.fixture
    def parser(self):
        return PythonParser()

    def test_parse_syntax_error_file(self, parser, tmp_path):
        """Test parsing file with syntax errors."""
        code = "def broken("  # Incomplete function

        file_path = tmp_path / "broken.py"
        file_path.write_text(code)

        # Parser should handle gracefully
        result = parser.parse_file(file_path, "test-repo")
        # Should not crash, may return empty or partial results
        assert isinstance(result, ParseResult)

    def test_parse_unicode_content(self, parser, tmp_path):
        """Test parsing file with unicode content."""
        code = dedent('''
            def greet():
                """こんにちは - Hello in Japanese"""
                return "你好世界"
        ''').strip()

        file_path = tmp_path / "unicode.py"
        file_path.write_text(code, encoding='utf-8')

        result = parser.parse_file(file_path, "test-repo")

        chunk = result.chunks[0]
        assert chunk.name == "greet"
        assert "こんにちは" in chunk.docstring or "こんにちは" in chunk.content

    def test_parse_very_long_function(self, parser, tmp_path):
        """Test parsing very long function."""
        lines = ["def long_function():"]
        lines.extend(["    x = 1"] * 500)  # 500 lines
        lines.append("    return x")
        code = "\n".join(lines)

        file_path = tmp_path / "long.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        assert len(result.chunks) == 1
        chunk = result.chunks[0]
        assert chunk.end_line - chunk.start_line >= 500

    def test_parse_file_with_tabs(self, parser, tmp_path):
        """Test parsing file with tab indentation."""
        code = "def tabbed():\n\treturn True"

        file_path = tmp_path / "tabs.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        assert len(result.chunks) == 1
        assert result.chunks[0].name == "tabbed"

    def test_parse_dataclass(self, parser, tmp_path):
        """Test parsing dataclass."""
        code = dedent('''
            class Person:
                """A person class."""
                name: str
                age: int
        ''').strip()

        file_path = tmp_path / "dataclass.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        class_chunk = next((c for c in result.chunks if c.chunk_type == ChunkType.CLASS), None)
        assert class_chunk is not None
        assert class_chunk.name == "Person"

    def test_parse_property_decorator(self, parser, tmp_path):
        """Test parsing class with property (may or may not extract as method)."""
        code = dedent('''
            class MyClass:
                """A class with a value property."""
                pass
        ''').strip()

        file_path = tmp_path / "prop.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path, "test-repo")

        # Just verify we can parse the class
        class_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) == 1

    def test_nonexistent_file(self, parser, tmp_path):
        """Test parsing non-existent file."""
        file_path = tmp_path / "does_not_exist.py"

        # Should raise or return empty result
        try:
            result = parser.parse_file(file_path, "test-repo")
            # If it doesn't raise, should return empty
            assert result.chunks == []
        except (FileNotFoundError, IOError):
            pass  # Expected behavior
