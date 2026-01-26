"""
Data models for RepoMind.

Provides the core data structures used throughout the application:

- **CodeChunk**: Represents a semantic unit of code (function, class, method)
  with support for multiple detail levels (summary/preview/full) for
  token-efficient responses.
- **ChunkType**: Enum of code construct types (function, class, method, etc.)
- **CallInfo**: Represents a function/method call relationship in the call graph.
- **ParseResult**: Result of parsing a source file into chunks.
- **DetailLevel**: Controls response verbosity (summary ~50 tokens,
  preview ~200 tokens, full ~500+ tokens).
- **InheritanceInfo**: Tracks class inheritance and interface implementation
  relationships for polymorphic resolution.
"""

from .chunk import CallInfo, ChunkType, CodeChunk, DetailLevel, InheritanceInfo, ParseResult

__all__ = [
    "CodeChunk",
    "ChunkType",
    "CallInfo",
    "ParseResult",
    "DetailLevel",
    "InheritanceInfo",
]
