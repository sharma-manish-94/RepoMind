"""Chunking service for extracting code chunks from repositories."""

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import IndexConfig, get_config
from ..models.chunk import CallInfo, CodeChunk, ParseResult
from ..parsers import get_language_for_file, get_parser

console = Console()


@dataclass
class ChunkingResult:
    """Result of chunking a repository or file.

    Contains both code chunks (for embedding/symbol table) and
    call information (for call graph).
    """

    chunks: list[CodeChunk] = field(default_factory=list)
    calls: list[CallInfo] = field(default_factory=list)
    files_processed: int = 0
    files_skipped: int = 0


class GitIgnoreParser:
    """Simple .gitignore parser."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.patterns: list[tuple[str, bool]] = []  # (pattern, is_negation)
        self._load_gitignore()

    def _load_gitignore(self):
        """Load .gitignore patterns from repo root."""
        gitignore_path = self.repo_root / ".gitignore"
        if not gitignore_path.exists():
            return

        try:
            content = gitignore_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Handle negation
                is_negation = line.startswith("!")
                if is_negation:
                    line = line[1:]

                self.patterns.append((line, is_negation))
        except Exception:
            pass

    def is_ignored(self, relative_path: str) -> bool:
        """Check if a path should be ignored based on .gitignore rules."""
        if not self.patterns:
            return False

        # Normalize path separators
        relative_path = relative_path.replace("\\", "/")
        ignored = False

        for pattern, is_negation in self.patterns:
            if self._matches_pattern(relative_path, pattern):
                ignored = not is_negation

        return ignored

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches gitignore pattern."""
        # Handle directory-only patterns (ending with /)
        if pattern.endswith("/"):
            pattern = pattern[:-1]
            # Only match directories
            parts = path.split("/")
            return any(fnmatch.fnmatch(part, pattern) for part in parts[:-1])

        # Handle patterns with /
        if "/" in pattern:
            # Pattern is relative to repo root
            if pattern.startswith("/"):
                pattern = pattern[1:]
            return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern + "/*")
        else:
            # Pattern matches any path component
            parts = path.split("/")
            return any(fnmatch.fnmatch(part, pattern) for part in parts)


class ChunkingService:
    """Service for extracting code chunks from source files."""

    def __init__(self, config: IndexConfig | None = None):
        self.config = config or get_config().index
        self._gitignore_cache: dict[Path, GitIgnoreParser] = {}

    def chunk_repository(self, repo_path: Path, repo_name: str | None = None) -> list[CodeChunk]:
        """Extract all code chunks from a repository (legacy interface).

        Args:
            repo_path: Path to the repository root
            repo_name: Name of the repository (defaults to directory name)

        Returns:
            List of CodeChunks extracted from the repository
        """
        result = self.chunk_repository_full(repo_path, repo_name)
        return result.chunks

    def chunk_repository_full(
        self, repo_path: Path, repo_name: str | None = None
    ) -> ChunkingResult:
        """Extract all code chunks and call information from a repository.

        Args:
            repo_path: Path to the repository root
            repo_name: Name of the repository (defaults to directory name)

        Returns:
            ChunkingResult with chunks, calls, and processing stats
        """
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

        if repo_name is None:
            repo_name = repo_path.name

        result = ChunkingResult()

        # Find all source files
        source_files = self._find_source_files(repo_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Chunking {repo_name}...", total=len(source_files))

            for file_path in source_files:
                progress.update(task, advance=1, description=f"Processing {file_path.name}")

                language = get_language_for_file(str(file_path))
                if not language:
                    result.files_skipped += 1
                    continue

                parser = get_parser(language)
                if not parser:
                    result.files_skipped += 1
                    continue

                try:
                    # Make path relative to repo root
                    relative_path = file_path.relative_to(repo_path)
                    parse_result = parser.parse_file(file_path, repo_name)

                    # Update file paths to be relative
                    for chunk in parse_result.chunks:
                        chunk.file_path = str(relative_path)

                    # Update caller_file paths in calls to be relative
                    for call in parse_result.calls:
                        call.caller_file = str(relative_path)

                    result.chunks.extend(parse_result.chunks)
                    result.calls.extend(parse_result.calls)
                    result.files_processed += 1
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to parse {file_path}: {e}[/yellow]")
                    result.files_skipped += 1

        console.print(
            f"[green]Processed {result.files_processed} files, "
            f"extracted {len(result.chunks)} chunks, {len(result.calls)} call relations[/green]"
        )
        if result.files_skipped:
            console.print(f"[yellow]Skipped {result.files_skipped} files[/yellow]")

        return result

    def chunk_file(self, file_path: Path, repo_name: str) -> list[CodeChunk]:
        """Extract code chunks from a single file (legacy interface).

        Args:
            file_path: Path to the source file
            repo_name: Name of the repository containing the file

        Returns:
            List of CodeChunks extracted from the file
        """
        result = self.chunk_file_full(file_path, repo_name)
        return result.chunks

    def chunk_file_full(self, file_path: Path, repo_name: str) -> ParseResult:
        """Extract code chunks and calls from a single file.

        Args:
            file_path: Path to the source file
            repo_name: Name of the repository containing the file

        Returns:
            ParseResult with chunks and call information
        """
        language = get_language_for_file(str(file_path))
        if not language:
            return ParseResult()

        parser = get_parser(language)
        if not parser:
            return ParseResult()

        return parser.parse_file(file_path, repo_name)

    def _find_source_files(self, repo_path: Path) -> list[Path]:
        """Find all source files in a repository."""
        source_files = []
        extensions = {".py", ".java", ".ts", ".tsx", ".js", ".jsx", ".mjs"}

        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix not in extensions:
                continue

            # Check against ignore patterns
            if self._should_ignore(file_path, repo_path):
                continue

            source_files.append(file_path)

        return sorted(source_files)

    def _get_gitignore(self, repo_root: Path) -> GitIgnoreParser:
        """Get or create GitIgnoreParser for a repository."""
        if repo_root not in self._gitignore_cache:
            self._gitignore_cache[repo_root] = GitIgnoreParser(repo_root)
        return self._gitignore_cache[repo_root]

    def _should_ignore(self, file_path: Path, repo_root: Path) -> bool:
        """Check if a file should be ignored based on patterns and .gitignore."""
        relative = file_path.relative_to(repo_root)
        relative_str = str(relative).replace("\\", "/")
        parts = relative.parts

        # Fast path: check for common directories to always ignore
        always_ignore_dirs = {
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            "dist", "build", "target", ".next", ".nuxt", "coverage",
            ".idea", ".vscode", "vendor", "out", ".gradle", ".mvn",
            "test-output", "bin", "lib", "generated",
        }
        if any(part in always_ignore_dirs for part in parts):
            return True

        # Check filename patterns
        always_ignore_files = {".min.js", ".bundle.js", ".chunk.js", ".d.ts"}
        if any(relative_str.endswith(suffix) for suffix in always_ignore_files):
            return True

        # Check .gitignore patterns
        gitignore = self._get_gitignore(repo_root)
        if gitignore.is_ignored(relative_str):
            return True

        # Check glob patterns from config for remaining cases
        for pattern in self.config.ignore_patterns:
            if "**" in pattern:
                if pattern.startswith("**/") and "*" not in pattern[3:].replace("*.", ""):
                    suffix = pattern[3:]
                    if relative_str.endswith(suffix.replace("*", "")):
                        return True
            else:
                if fnmatch.fnmatch(relative_str, pattern):
                    return True

        return False
