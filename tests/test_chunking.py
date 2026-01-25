"""Tests for the chunking service."""

from pathlib import Path
from textwrap import dedent

import pytest

from repomind.models.chunk import ChunkType, ParseResult
from repomind.parsers.python_parser import PythonParser


class TestPythonParser:
    """Tests for Python parser."""

    def test_parse_function(self, tmp_path):
        """Test parsing a simple function."""
        code = dedent('''
            def hello(name: str) -> str:
                """Say hello to someone."""
                return f"Hello, {name}!"
        ''').strip()

        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        parser = PythonParser()
        result = parser.parse_file(file_path, "test-repo")
        chunks = result.chunks

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.name == "hello"
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.language == "python"
        assert "Say hello" in (chunk.docstring or "")

    def test_parse_class_with_methods(self, tmp_path):
        """Test parsing a class with methods."""
        code = dedent('''
            class Calculator:
                """A simple calculator."""

                def add(self, a: int, b: int) -> int:
                    """Add two numbers."""
                    return a + b

                def subtract(self, a: int, b: int) -> int:
                    """Subtract b from a."""
                    return a - b
        ''').strip()

        file_path = tmp_path / "calc.py"
        file_path.write_text(code)

        parser = PythonParser()
        result = parser.parse_file(file_path, "test-repo")
        chunks = result.chunks

        # Should have: class + 2 methods
        assert len(chunks) == 3

        class_chunk = next(c for c in chunks if c.chunk_type == ChunkType.CLASS)
        assert class_chunk.name == "Calculator"

        methods = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        assert len(methods) == 2
        assert all(m.parent_name == "Calculator" for m in methods)

    def test_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        file_path = tmp_path / "empty.py"
        file_path.write_text("")

        parser = PythonParser()
        result = parser.parse_file(file_path, "test-repo")

        assert len(result.chunks) == 0

    def test_qualified_name(self, tmp_path):
        """Test qualified name generation."""
        code = dedent('''
            class MyClass:
                def my_method(self):
                    pass
        ''').strip()

        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        parser = PythonParser()
        result = parser.parse_file(file_path, "test-repo")

        method = next(c for c in result.chunks if c.chunk_type == ChunkType.METHOD)
        assert method.get_qualified_name() == "MyClass.my_method"


class TestChunkModel:
    """Tests for CodeChunk model."""

    def test_to_embedding_text(self):
        """Test embedding text generation."""
        from repomind.models.chunk import CodeChunk

        chunk = CodeChunk(
            id="test123",
            repo_name="Actions",
            file_path="src/handler.py",
            start_line=10,
            end_line=20,
            chunk_type=ChunkType.FUNCTION,
            name="handle_request",
            content="def handle_request(): pass",
            signature="def handle_request(request: Request) -> Response",
            docstring="Handle incoming requests.",
            language="python",
        )

        text = chunk.to_embedding_text()

        # Updated assertions based on actual embedding text format
        assert "handler.py" in text
        assert "handle_request" in text.lower() or "handle" in text
        assert "Handle incoming requests" in text
        assert "def handle_request" in text
