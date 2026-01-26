"""
Constants and Configuration Values for RepoMind.

This module centralizes all magic strings, numbers, and configuration values
used throughout the application. This provides:

1. Single source of truth for all constants
2. Easy configuration changes without hunting through code
3. Self-documenting values with clear naming
4. Type safety through proper annotations

Usage:
    from repomind.constants import (
        COLLECTION_NAME,
        DEFAULT_EMBEDDING_MODEL,
        SUPPORTED_LANGUAGES,
    )

Naming Conventions:
    - ALL_CAPS for constants
    - Descriptive names that read like English
    - Grouped by category with clear section headers

Author: RepoMind Team
"""

from enum import Enum
from pathlib import Path


# ============================================================================
# Application Metadata
# ============================================================================

APPLICATION_NAME = "RepoMind"
APPLICATION_VERSION = "1.0.0"
APPLICATION_DESCRIPTION = "AI-powered code intelligence platform for any codebase"


# ============================================================================
# File System Paths
# ============================================================================

DEFAULT_DATA_DIRECTORY = Path.home() / ".repomind"
DEFAULT_CHROMA_DIRECTORY = DEFAULT_DATA_DIRECTORY / "chroma"
DEFAULT_METADATA_DIRECTORY = DEFAULT_DATA_DIRECTORY / "metadata"
DEFAULT_LOG_DIRECTORY = DEFAULT_DATA_DIRECTORY / "logs"



# ============================================================================
# ChromaDB Configuration
# ============================================================================

CHROMADB_COLLECTION_NAME = "code_chunks"
CHROMADB_COLLECTION_DESCRIPTION = "Code chunks for semantic search"
CHROMADB_BATCH_SIZE = 500  # Maximum items per upsert operation



# ============================================================================
# Embedding Models
# ============================================================================

class EmbeddingModelName(str, Enum):
    """
    Available embedding models for code vectorization.

    Local models run entirely on your machine with no API calls.
    API models require authentication and make external requests.

    Recommended for code search:
    - BGE_BASE: Best balance of quality and speed (default)
    - CODEBERT: Good for code understanding, slower
    - MPNET: Good for general text, fast
    """

    # Local models (privacy-first, no API required)
    BGE_BASE = "BAAI/bge-base-en-v1.5"      # Best for code search
    BGE_LARGE = "BAAI/bge-large-en-v1.5"    # Higher quality, slower
    CODEBERT = "microsoft/codebert-base"
    MINILM = "all-MiniLM-L6-v2"
    MPNET = "all-mpnet-base-v2"

    # API-based models (require API keys)
    VOYAGE_CODE = "voyage-code-2"
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"

    # Testing only
    MOCK = "mock"


# Default model - BGE for best semantic search quality with code
DEFAULT_EMBEDDING_MODEL = EmbeddingModelName.BGE_BASE.value

# Embedding dimensions for each model
EMBEDDING_DIMENSIONS = {
    EmbeddingModelName.BGE_BASE.value: 768,
    EmbeddingModelName.BGE_LARGE.value: 1024,
    EmbeddingModelName.CODEBERT.value: 768,
    EmbeddingModelName.MINILM.value: 384,
    EmbeddingModelName.MPNET.value: 768,
    EmbeddingModelName.VOYAGE_CODE.value: 1024,
    EmbeddingModelName.OPENAI_SMALL.value: 1536,
    EmbeddingModelName.OPENAI_LARGE.value: 3072,
    EmbeddingModelName.MOCK.value: 768,
}

# Local models that don't require API calls
LOCAL_EMBEDDING_MODELS = {
    EmbeddingModelName.BGE_BASE.value,
    EmbeddingModelName.BGE_LARGE.value,
    EmbeddingModelName.CODEBERT.value,
    EmbeddingModelName.MINILM.value,
    EmbeddingModelName.MPNET.value,
}

# Default embedding dimension (for fallback)
DEFAULT_EMBEDDING_DIMENSION = 768

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = 32


# ============================================================================
# Code Parsing Configuration
# ============================================================================

class SupportedLanguage(str, Enum):
    """
    Programming languages supported for parsing and indexing.

    Each language requires:
    1. A tree-sitter grammar
    2. A parser implementation in parsers/
    """

    PYTHON = "python"
    JAVA = "java"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"


# File extensions mapped to languages
FILE_EXTENSION_TO_LANGUAGE = {
    ".py": SupportedLanguage.PYTHON.value,
    ".java": SupportedLanguage.JAVA.value,
    ".ts": SupportedLanguage.TYPESCRIPT.value,
    ".tsx": SupportedLanguage.TYPESCRIPT.value,
    ".js": SupportedLanguage.JAVASCRIPT.value,
    ".jsx": SupportedLanguage.JAVASCRIPT.value,
}

# Maximum lines for a single code chunk
MAX_CHUNK_LINES = 500

# Minimum lines for a code chunk to be indexed
MIN_CHUNK_LINES = 3


# ============================================================================
# Code Chunk Types
# ============================================================================

class ChunkTypeName(str, Enum):
    """
    Types of code constructs that can be indexed.

    Each type represents a distinct semantic unit in source code
    that is useful for search and analysis.
    """

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    INTERFACE = "interface"
    CONSTRUCTOR = "constructor"
    MODULE = "module"


# ============================================================================
# Call Graph Configuration
# ============================================================================

class CallTypeName(str, Enum):
    """
    Types of function/method calls tracked in the call graph.

    Understanding call relationships enables:
    - Impact analysis ("who calls this?")
    - Dependency analysis ("what does this call?")
    - Refactoring safety checks
    """

    DIRECT = "direct"           # Standard function call
    METHOD = "method"           # Method invocation on object
    CONSTRUCTOR = "constructor" # Object instantiation
    STATIC = "static"          # Static method call
    CALLBACK = "callback"       # Passed as callback/handler


# ============================================================================
# Search Configuration
# ============================================================================

DEFAULT_SEARCH_RESULTS = 10
MAX_SEARCH_RESULTS = 100
MIN_SEARCH_RESULTS = 1


# ============================================================================
# Ignore Patterns
# ============================================================================

# Directories to always skip during indexing
IGNORED_DIRECTORIES = {
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",

    ".env",
    "build",
    "dist",
    ".idea",
    ".vscode",
    "target",  # Java build output
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "coverage",
    ".coverage",
    "htmlcov",
}

# File patterns to always skip
IGNORED_FILE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.class",
    "*.jar",
    "*.war",
    "*.min.js",
    "*.map",
    "*.d.ts",  # TypeScript declaration files
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
}


# ============================================================================
# Security Configuration
# ============================================================================

# Directories that should never be indexed (security)
FORBIDDEN_INDEX_PATHS = [
    "/etc",
    "/var",
    "/usr",
    "/sys",
    "/proc",
    "/dev",
    "/boot",
    "/root",
]


# ============================================================================
# Performance Tuning
# ============================================================================

# Maximum concurrent file operations
MAX_CONCURRENT_FILE_OPERATIONS = 4

# Memory limit for batch operations (MB)
BATCH_MEMORY_LIMIT_MB = 512

# Progress update interval (items)
PROGRESS_UPDATE_INTERVAL = 10


# ============================================================================
# Error Messages
# ============================================================================

class ErrorMessage:
    """
    Standardized error messages for consistent user experience.

    Using a class instead of a dict provides:
    - IDE autocomplete
    - Compile-time checking
    - Easy documentation
    """

    REPOSITORY_NOT_FOUND = "Repository path does not exist: {path}"
    PATH_NOT_DIRECTORY = "Path is not a directory: {path}"
    FORBIDDEN_PATH = "Cannot index system directory: {path}"
    EMPTY_QUERY = "Search query cannot be empty"
    INVALID_CHUNK_TYPE = "Invalid chunk type: {chunk_type}. Valid options: {valid_types}"
    EMBEDDING_DIMENSION_MISMATCH = (
        "Embedding dimension mismatch. "
        "Existing: {existing}D, New: {new}D. "
        "Collection will be recreated."
    )
    INDEX_NOT_FOUND = "No index found. Please run 'repomind index' first."
    SYMBOL_NOT_FOUND = "No symbol found matching: {symbol}"


# ============================================================================
# Success Messages
# ============================================================================

class SuccessMessage:
    """Standardized success messages for consistent user experience."""

    INDEXING_COMPLETE = "Successfully indexed {repo_name}: {chunks} chunks, {symbols} symbols"
    SEARCH_COMPLETE = "Found {count} results for: {query}"
    COLLECTION_MIGRATED = "Collection recreated with new embedding dimension ({dimension}D)"
    INDEX_CLEARED = "Cleared {count} chunks from index"


# ============================================================================
# MCP Tool Names
# ============================================================================

class MCPToolName(str, Enum):
    """
    Names of MCP tools exposed by this server.

    These must match exactly what's registered in server.py.
    """

    INDEX_REPO = "index_repo"
    INDEX_ALL = "index_all_repositories"
    SEMANTIC_GREP = "semantic_grep"
    GET_CONTEXT = "get_context"
    GET_INDEX_STATS = "get_index_stats"
    LOOKUP_SYMBOL = "lookup_symbol"
    FIND_CALLERS = "find_callers"
    FIND_CALLEES = "find_callees"
    # Code analysis tools
    FILE_SUMMARY = "file_summary"
    FIND_USAGES = "find_usages"
    FIND_IMPLEMENTATIONS = "find_implementations"
    FIND_TESTS = "find_tests"
    DIFF_IMPACT = "diff_impact"
    # Compound operations (token-efficient)
    EXPLORE = "explore"
    UNDERSTAND = "understand"
    PREPARE_CHANGE = "prepare_change"
    # Pattern analysis
    ANALYZE_PATTERNS = "analyze_patterns"
    GET_CODING_CONVENTIONS = "get_coding_conventions"
    # Production features
    ANALYZE_OWNERSHIP = "analyze_ownership"
    SCAN_SECRETS = "scan_secrets"
    GET_METRICS = "get_metrics"
