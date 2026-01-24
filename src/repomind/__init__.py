"""
RepoMind - AI-Powered Code Intelligence Platform for Any Codebase.

This package provides an MCP (Model Context Protocol) server that enables
AI assistants to search, understand, and navigate codebases using semantic
understanding rather than simple text matching.

Key Features:
    - **Semantic Code Search**: Find code by meaning, not just text patterns
    - **Auto-Discovery**: Automatically discover and index repositories
    - **Call Graph Analysis**: Understand function relationships and dependencies
    - **Symbol Lookup**: Fast exact-match lookups for functions, classes, methods
    - **Multi-Language Support**: Python, Java, TypeScript, JavaScript
    - **Privacy-First**: Local embeddings by default, no code leaves your machine

Quick Start:
    1. Install: pip install repomind
    2. Index: repomind --repos-dir ~/projects index-all
    3. Search: repomind search "authentication middleware"

MCP Integration:
    This server integrates with AI assistants via the Model Context Protocol.
    Configure your MCP client (Claude, VS Code Copilot) to connect to this server.

Architecture:
    - server.py: MCP protocol handler and tool registration
    - tools/: Individual tool implementations (search, index, context)
    - services/: Core business logic (embedding, storage, parsing)
    - parsers/: Language-specific code parsers using tree-sitter
    - models/: Data models for code chunks, symbols, and call graphs
    - config.py: Configuration with auto-discovery support

For detailed documentation, see:
    - README.md: User guide
    - DEVELOPER_GUIDE.md: Developer onboarding

Author: RepoMind Team
License: Internal Use
"""

__version__ = "1.0.0"
__author__ = "Code Expert Team"
__description__ = "AI-powered code intelligence platform for any codebase"

# Public API
from repomind.constants import (
    APPLICATION_NAME,
    APPLICATION_VERSION,
    DEFAULT_EMBEDDING_MODEL,
    CHROMADB_COLLECTION_NAME,
    SupportedLanguage,
    ChunkTypeName,
)

from repomind.logging import (
    get_logger,
    setup_logging,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__description__",

    # Constants
    "APPLICATION_NAME",
    "APPLICATION_VERSION",
    "DEFAULT_EMBEDDING_MODEL",
    "CHROMADB_COLLECTION_NAME",
    "SupportedLanguage",
    "ChunkTypeName",

    # Logging
    "get_logger",
    "setup_logging",
]
