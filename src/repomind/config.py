"""
Configuration Module for RepoMind.

This module provides the configuration system for the RepoMind application,
including settings for indexing, embedding, search, and repository discovery.

The configuration follows a hierarchical structure:
    - IndexConfig: Indexing behavior settings
    - EmbeddingConfig: Embedding model configuration
    - SearchConfig: Search quality tuning
    - RepositoryConfig: Repository discovery and filtering
    - Config: Main configuration aggregating all sub-configs

Example Usage:
    >>> from repomind.config import get_config, set_config, Config
    >>> config = Config(repos_dir=Path("/path/to/repos"))
    >>> set_config(config)
    >>> current_config = get_config()

Author: RepoMind Team
"""

from pathlib import Path
from typing import Optional
import fnmatch
import warnings

from pydantic import BaseModel, Field


class IndexConfig(BaseModel):
    """
    Configuration for indexing behavior.

    Controls how code is parsed, chunked, and indexed including
    language support, file filtering, and storage locations.

    Attributes:
        data_dir: Root directory for all index data storage
        max_chunk_lines: Maximum lines per code chunk
        include_imports: Whether to index import statements
        include_comments: Whether to index standalone comments
        languages: List of programming languages to parse
        ignore_patterns: Glob patterns for files to skip during indexing
    """

    # Paths
    data_dir: Path = Field(
        default=Path.home() / ".repomind",
        description="Directory for storing index data",
    )

    # Chunking
    max_chunk_lines: int = Field(default=500, description="Maximum lines per chunk")
    include_imports: bool = Field(default=True, description="Index import statements")
    include_comments: bool = Field(default=False, description="Index standalone comments")

    # Languages to index
    languages: list[str] = Field(
        default=["python", "java", "typescript", "javascript"],
        description="Languages to parse and index",
    )

    # File patterns to ignore
    ignore_patterns: list[str] = Field(
        default=[
            "node_modules/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/dist/**",
            "**/build/**",
            "**/target/**",
            "**/*.min.js",
            "**/*.bundle.js",
            "**/vendor/**",
            "**/.venv/**",
            "**/venv/**",
        ],
        description="Glob patterns to ignore during indexing",
    )

    @property
    def chroma_dir(self) -> Path:
        """Directory for ChromaDB storage."""
        return self.data_dir / "chroma"

    @property
    def metadata_dir(self) -> Path:
        """Directory for metadata storage."""
        return self.data_dir / "metadata"


class SearchConfig(BaseModel):
    """Configuration for search behavior and quality tuning."""

    # Result configuration
    default_results: int = Field(default=10, description="Default number of search results")
    max_results: int = Field(default=100, description="Maximum allowed search results")

    # Quality thresholds
    similarity_threshold: float = Field(
        default=0.35,
        description="Minimum similarity score (0-1) to include in results",
    )
    high_confidence_threshold: float = Field(
        default=0.70,
        description="Threshold for high-confidence matches",
    )

    # Hybrid search configuration
    enable_hybrid_search: bool = Field(
        default=True,
        description="Combine semantic search with keyword matching",
    )
    keyword_boost_factor: float = Field(
        default=0.15,
        description="Boost factor per keyword match in hybrid search",
    )

    # Query expansion
    enable_query_expansion: bool = Field(
        default=True,
        description="Expand queries with synonyms and related terms",
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    model: str = Field(
        default="BAAI/bge-base-en-v1.5",  # Best balance of quality and speed for code
        description="Embedding model to use",
    )
    batch_size: int = Field(default=32, description="Batch size for embedding requests")
    normalize_embeddings: bool = Field(
        default=True,
        description="L2 normalize embeddings for cosine similarity",
    )
    query_instruction: str = Field(
        default="Represent this code search query: ",
        description="Instruction prefix for query embedding (BGE models)",
    )


class RouteLinkingConfig(BaseModel):
    """Configuration for cross-language API route linking.

    Controls how frontend API calls are matched to backend endpoints,
    including confidence thresholds, ambiguity detection, and matching behavior.

    Attributes:
        min_confidence: Minimum confidence score to consider a match valid
        high_confidence: Threshold for high-confidence matches (no ambiguity check)
        enable_fuzzy_matching: Allow fuzzy matching for parameter names
        max_matches_per_call: Maximum matches to return per API call
        flag_ambiguous: Whether to flag ambiguous matches for review
        openapi_fallback: Use OpenAPI specs as fallback source
        base_url_patterns: Common base URL patterns to strip

    Example:
        >>> config = RouteLinkingConfig(
        ...     min_confidence=0.70,
        ...     flag_ambiguous=True,
        ...     openapi_fallback=True
        ... )
    """

    # Confidence thresholds
    min_confidence: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0-1) to consider a match valid",
    )
    high_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Threshold for high-confidence matches",
    )

    # Matching behavior
    enable_fuzzy_matching: bool = Field(
        default=True,
        description="Allow fuzzy matching for parameter names (userId vs user_id)",
    )
    max_matches_per_call: int = Field(
        default=3,
        ge=1,
        description="Maximum number of matches to return per API call",
    )

    # Ambiguity handling
    flag_ambiguous: bool = Field(
        default=True,
        description="Flag matches with multiple candidates for manual review",
    )
    ambiguity_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Max score difference between matches to consider ambiguous",
    )

    # Fallback sources
    openapi_fallback: bool = Field(
        default=True,
        description="Use OpenAPI/Swagger specs as fallback route source",
    )

    # Base URL handling
    base_url_patterns: list[str] = Field(
        default=[
            r"https?://[^/]+",  # http://api.example.com
            r"/api/v\d+",       # /api/v1, /api/v2
            r"/api",            # /api
        ],
        description="Regex patterns for base URLs to strip during normalization",
    )

    # Scoring weights (must sum to 1.0)
    weight_path_structure: float = Field(
        default=0.40,
        description="Weight for path structure matching",
    )
    weight_param_position: float = Field(
        default=0.25,
        description="Weight for parameter position matching",
    )
    weight_param_name: float = Field(
        default=0.20,
        description="Weight for parameter name similarity",
    )
    weight_context: float = Field(
        default=0.15,
        description="Weight for context matching (same repo, language)",
    )


class RepositoryConfig(BaseModel):
    """
    Configuration for repository discovery and filtering.

    Controls how repositories are discovered in a directory,
    including pattern matching for inclusion/exclusion.

    Attributes:
        auto_discover: Enable automatic repository discovery
        include_patterns: Glob patterns for repos to include
        exclude_patterns: Glob patterns for repos to exclude
        min_files_threshold: Minimum files to consider as valid repo
    """

    auto_discover: bool = Field(
        default=True,
        description="Automatically discover subdirectories as repositories",
    )

    include_patterns: list[str] = Field(
        default=["*"],
        description="Glob patterns to include repositories (e.g., ['service-*', 'lib-*'])",
    )

    exclude_patterns: list[str] = Field(
        default=[
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".idea",
            ".vscode",
            "dist",
            "build",
            "target",
            ".ruff_cache",
            ".pytest_cache",
            ".mypy_cache",
            "__pypackages__",
        ],
        description="Glob patterns to exclude from repository discovery",
    )

    min_files_threshold: int = Field(
        default=1,
        description="Minimum number of source files to consider directory as a repository",
    )

    def matches_include_pattern(self, repo_name: str) -> bool:
        """
        Check if repository name matches any include pattern.

        Args:
            repo_name: Name of the repository to check

        Returns:
            True if repo matches at least one include pattern
        """
        return any(
            fnmatch.fnmatch(repo_name, pattern)
            for pattern in self.include_patterns
        )

    def matches_exclude_pattern(self, repo_name: str) -> bool:
        """
        Check if repository name matches any exclude pattern.

        Args:
            repo_name: Name of the repository to check

        Returns:
            True if repo matches at least one exclude pattern
        """
        return any(
            fnmatch.fnmatch(repo_name, pattern)
            for pattern in self.exclude_patterns
        )

    def should_include_repository(self, repo_name: str) -> bool:
        """
        Determine if a repository should be included based on patterns.

        A repository is included if it matches at least one include pattern
        AND does not match any exclude pattern.

        Args:
            repo_name: Name of the repository to check

        Returns:
            True if repository should be included
        """
        if self.matches_exclude_pattern(repo_name):
            return False
        return self.matches_include_pattern(repo_name)


class Config(BaseModel):
    """
    Main configuration for Code Expert.

    Aggregates all sub-configurations and provides the central
    configuration access point for the application.

    Attributes:
        index: Indexing behavior configuration
        embedding: Embedding model configuration
        search: Search quality tuning configuration
        repository: Repository discovery configuration
        repos_dir: Root directory containing repositories to index
        repos_list: Optional explicit list of repo names (overrides auto-discovery)

    Example:
        >>> config = Config(
        ...     repos_dir=Path("/path/to/repos"),
        ...     repository=RepositoryConfig(
        ...         include_patterns=["service-*", "lib-*"],
        ...         exclude_patterns=["legacy-*"]
        ...     )
        ... )
    """

    index: IndexConfig = Field(default_factory=IndexConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    repository: RepositoryConfig = Field(default_factory=RepositoryConfig)
    route_linking: RouteLinkingConfig = Field(default_factory=RouteLinkingConfig)

    # Repository location
    repos_dir: Optional[Path] = Field(
        default=None,
        description="Root directory containing repositories to index",
    )

    # Optional explicit repository list (overrides auto-discovery)
    repos_list: Optional[list[str]] = Field(
        default=None,
        description="Explicit list of repository names to index (overrides auto-discovery)",
    )

    def get_repositories_to_index(self) -> list[str]:
        """
        Get the list of repository names to index.

        Priority:
            1. Explicit repos_list if provided
            2. Auto-discovery from repos_dir

        Returns:
            List of repository names to index

        Raises:
            ValueError: If repos_dir is not set and no explicit list provided
        """
        # Priority 1: Explicit repos_list
        if self.repos_list is not None:
            return self.repos_list

        # Priority 2: Auto-discovery
        if self.repos_dir is None:
            raise ValueError(
                "repos_dir must be set for auto-discovery, "
                "or provide explicit repos_list"
            )

        return self._discover_repositories()

    def _discover_repositories(self) -> list[str]:
        """
        Discover repositories in repos_dir based on configuration.

        Returns:
            List of discovered repository names
        """
        if self.repos_dir is None or not self.repos_dir.exists():
            return []

        discovered = []

        for item in sorted(self.repos_dir.iterdir()):
            if not item.is_dir():
                continue

            repo_name = item.name

            # Apply include/exclude filters
            if not self.repository.should_include_repository(repo_name):
                continue

            # Check if directory has source files (basic validation)
            if self._is_valid_repository(item):
                discovered.append(repo_name)

        return discovered

    def _is_valid_repository(self, repo_path: Path) -> bool:
        """
        Check if a directory appears to be a valid code repository.

        Args:
            repo_path: Path to the directory to check

        Returns:
            True if directory appears to be a valid repository
        """
        # Count source files
        source_extensions = {'.py', '.java', '.ts', '.tsx', '.js', '.jsx'}
        file_count = 0

        try:
            for item in repo_path.rglob('*'):
                if item.is_file() and item.suffix in source_extensions:
                    file_count += 1
                    if file_count >= self.repository.min_files_threshold:
                        return True
        except PermissionError:
            return False

        return file_count >= self.repository.min_files_threshold

    @classmethod
    def load_default(cls) -> "Config":
        """
        Load default configuration.

        Returns:
            Config instance with default values
        """
        return cls()


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Creates a default configuration if none has been set.

    Returns:
        Current global Config instance
    """
    global _config
    if _config is None:
        _config = Config.load_default()
    return _config


def set_config(config: Config) -> None:
    """
    Set the global configuration instance.

    Args:
        config: Config instance to use globally
    """
    global _config
    _config = config


def reset_config() -> None:
    """
    Reset the global configuration to None.

    Useful for testing or reinitializing configuration.
    """
    global _config
    _config = None

