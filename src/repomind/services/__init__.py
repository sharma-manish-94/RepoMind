"""
Services Layer for RepoMind.

This module provides the core business logic services used throughout
the application. Each service encapsulates a specific capability:

**ChunkingService**:
    Parses source files and extracts semantic code chunks.
    Uses tree-sitter for robust AST parsing.

**EmbeddingService**:
    Generates vector embeddings from code chunks.
    Supports local and API-based models.

**StorageService**:
    Persists chunks and embeddings to ChromaDB.
    Handles vector search and metadata filtering.

**SymbolTableService**:
    Maintains a fast lookup table for symbols.
    Enables exact-match symbol queries.

**CallGraphService**:
    Builds and queries the function call graph.
    Enables "who calls this?" impact analysis.

**ManifestService**:
    Tracks file changes for incremental indexing.
    Only re-indexes files that have changed.

**ResponseFormatter**:
    Formats search results with token budgeting and detail levels.
    Supports summary/preview/full modes for token-efficient responses.

**LSPClientManager**:
    Manages Language Server Protocol connections for compiler-grade
    code analysis (find references, go to definition, type hierarchy).

**HybridSymbolResolver**:
    Combines LSP precision with tree-sitter speed. Falls back
    gracefully when LSP is unavailable.

**PatternAnalyzer**:
    Detects code patterns, library usage conventions, and team
    practices. Tracks pattern momentum and identifies golden files.

**OwnershipService**:
    Parses CODEOWNERS files and integrates git blame for file
    ownership and reviewer suggestion.

**SecurityScanner**:
    Scans code for hardcoded secrets, API keys, and credentials
    using 26 detection patterns with confidence scoring.

**MetricsService**:
    Calculates code complexity metrics including cyclomatic,
    cognitive, Halstead, and maintainability index.

Architecture:
    Services are designed to be stateless (except caching) and
    can be instantiated multiple times safely. Each service
    focuses on a single responsibility.

Usage:
    from repomind.services import (
        ChunkingService,
        EmbeddingService,
        StorageService,
    )

    # Parse files
    chunking = ChunkingService()
    chunks = chunking.chunk_repository("/path/to/repo")

    # Generate embeddings
    embedding = EmbeddingService()
    vectors = embedding.embed_chunks(chunks)

    # Store for search
    storage = StorageService()
    storage.store_chunks(chunks, vectors)

Author: RepoMind Team
"""

from .call_graph import CallGraphService, CallNode, CallRelation
from .chunking import ChunkingService
from .embedding import EmbeddingService, LocalEmbeddingService, MockEmbeddingService
from .manifest import FileChange, FileStatus, ManifestService
from .metrics import MetricsService
from .ownership import OwnershipService
from .pattern_analyzer import PatternAnalyzer
from .response_formatter import ResponseFormatter
from .security_scanner import SecurityScanner
from .storage import StorageService
from .symbol_table import Symbol, SymbolTableService

__all__ = [
    # Core services for indexing and search
    "ChunkingService",
    "EmbeddingService",
    "LocalEmbeddingService",
    "MockEmbeddingService",
    "StorageService",

    # Incremental indexing support
    "ManifestService",
    "FileChange",
    "FileStatus",
    # Symbol table for fast lookups
    "SymbolTableService",
    "Symbol",
    # Call graph for relationship tracking
    "CallGraphService",
    "CallRelation",
    "CallNode",
    # Token-efficient response formatting
    "ResponseFormatter",
    # Pattern analysis and conventions
    "PatternAnalyzer",
    # Production features
    "OwnershipService",
    "SecurityScanner",
    "MetricsService",
]
