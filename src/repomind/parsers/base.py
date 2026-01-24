"""Base parser interface for code parsing."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..models.chunk import CodeChunk, ParseResult


class BaseParser(ABC):
    """Abstract base class for language-specific parsers.

    Each parser uses tree-sitter to parse source code and extract
    semantic chunks (functions, classes, methods, etc.) plus call relationships.
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """The language this parser handles."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """File extensions this parser can handle."""
        pass

    @abstractmethod
    def parse_file(self, file_path: Path, repo_name: str) -> ParseResult:
        """Parse a file and extract code chunks and call relationships.

        Args:
            file_path: Path to the file to parse
            repo_name: Name of the repository containing the file

        Returns:
            ParseResult containing CodeChunks and CallInfo objects
        """
        pass

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        return any(str(file_path).endswith(ext) for ext in self.file_extensions)

    def _generate_chunk_id(
        self, repo_name: str, file_path: str, name: str, chunk_type: str, start_line: int = 0
    ) -> str:
        """Generate a unique ID for a chunk."""
        import hashlib

        # Include start_line to ensure uniqueness for same-named items
        content = f"{repo_name}:{file_path}:{name}:{chunk_type}:{start_line}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
