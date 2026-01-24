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
]
