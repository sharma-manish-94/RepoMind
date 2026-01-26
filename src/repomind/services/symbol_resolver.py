"""
Hybrid Symbol Resolver - LSP + Tree-sitter Integration.

This module provides a unified interface for symbol resolution that
combines the precision of LSP with the speed of tree-sitter:

1. **LSP (Language Server Protocol)**:
   - Compiler-grade accuracy
   - Resolves overloads, generics, aliased imports
   - Handles dynamic dispatch correctly
   - Slower, requires running language servers

2. **Tree-sitter**:
   - Fast static analysis
   - Works offline without language servers
   - Good for most cases
   - May miss complex type relationships

Strategy:
- Use LSP when available and precision is critical
- Fall back to tree-sitter for speed or when LSP unavailable
- Cache results to minimize LSP calls

Usage:
    resolver = HybridSymbolResolver()
    await resolver.initialize("/path/to/project")

    # Find references with automatic backend selection
    refs = await resolver.find_references("UserService.save")

    # Force LSP for precision
    refs = await resolver.find_references("UserService.save", prefer_lsp=True)

Author: RepoMind Team
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import logging

from .lsp_client import LSPClientManager, LSPLocation
from .symbol_table import SymbolTableService
from .call_graph import CallGraphService

logger = logging.getLogger(__name__)


class ResolverBackend(str, Enum):
    """Backend used for symbol resolution."""
    LSP = "lsp"
    TREE_SITTER = "tree_sitter"
    HYBRID = "hybrid"


@dataclass
class ResolvedSymbol:
    """A symbol resolved by the hybrid resolver."""
    name: str
    qualified_name: str
    file_path: str
    line: int
    character: int
    symbol_type: str
    backend: ResolverBackend
    confidence: float  # 0.0 to 1.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Reference:
    """A reference to a symbol."""
    file_path: str
    line: int
    character: int
    context: str  # The line of code
    reference_type: str  # call, type_hint, import, assignment, etc.
    backend: ResolverBackend
    confidence: float


@dataclass
class Definition:
    """The definition of a symbol."""
    symbol: ResolvedSymbol
    signature: Optional[str] = None
    docstring: Optional[str] = None
    implementation: Optional[str] = None


class HybridSymbolResolver:
    """
    Unified symbol resolver combining LSP and tree-sitter.

    Provides a single interface for symbol resolution that automatically
    selects the best backend based on availability and accuracy needs.
    """

    def __init__(
        self,
        lsp_enabled: bool = True,
        lsp_timeout: float = 5.0,
        cache_enabled: bool = True,
    ):
        """
        Initialize the hybrid resolver.

        Args:
            lsp_enabled: Whether to use LSP when available.
            lsp_timeout: Timeout for LSP operations in seconds.
            cache_enabled: Whether to cache resolution results.
        """
        self.lsp_enabled = lsp_enabled
        self.lsp_timeout = lsp_timeout
        self.cache_enabled = cache_enabled

        self._lsp_manager: Optional[LSPClientManager] = None
        self._symbol_table: Optional[SymbolTableService] = None
        self._call_graph: Optional[CallGraphService] = None
        self._workspace_root: Optional[Path] = None
        self._initialized = False

        # Simple in-memory cache
        self._cache: dict[str, Any] = {}
        self._cache_max_size = 1000

    async def initialize(self, workspace_root: str) -> bool:
        """
        Initialize the resolver for a workspace.

        Args:
            workspace_root: Root directory of the project.

        Returns:
            True if initialization succeeded.
        """
        self._workspace_root = Path(workspace_root)

        # Initialize tree-sitter based services (always available)
        self._symbol_table = SymbolTableService()
        self._call_graph = CallGraphService()

        # Initialize LSP if enabled
        if self.lsp_enabled:
            self._lsp_manager = LSPClientManager()
            try:
                await self._lsp_manager.initialize(workspace_root)
            except Exception as e:
                logger.warning(f"LSP initialization failed, using tree-sitter only: {e}")
                self._lsp_manager = None

        self._initialized = True
        return True

    async def shutdown(self):
        """Shutdown the resolver and its backends."""
        if self._lsp_manager:
            await self._lsp_manager.shutdown()
        self._cache.clear()
        self._initialized = False

    def is_lsp_available(self, file_path: str) -> bool:
        """Check if LSP is available for a file type."""
        if not self._lsp_manager:
            return False
        return self._lsp_manager.is_available(file_path)

    async def find_definition(
        self,
        symbol_name: str,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        character: Optional[int] = None,
        prefer_lsp: bool = False,
        repo_filter: Optional[str] = None,
    ) -> Optional[Definition]:
        """
        Find the definition of a symbol.

        Args:
            symbol_name: Name or qualified name of the symbol.
            file_path: Optional file path for context.
            line: Optional line number (0-indexed) for LSP lookup.
            character: Optional character position for LSP lookup.
            prefer_lsp: If True, prioritize LSP even if slower.
            repo_filter: Optional repository filter.

        Returns:
            Definition if found, None otherwise.
        """
        cache_key = f"def:{symbol_name}:{file_path}:{line}:{character}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        result = None

        # Try LSP if we have position info and LSP is available
        if (prefer_lsp or (file_path and line is not None and character is not None)):
            if file_path and self.is_lsp_available(file_path) and line is not None:
                result = await self._find_definition_lsp(
                    file_path, line, character or 0
                )

        # Fall back to tree-sitter
        if not result:
            result = self._find_definition_tree_sitter(symbol_name, repo_filter)

        if result and self.cache_enabled:
            self._cache_put(cache_key, result)

        return result

    async def find_references(
        self,
        symbol_name: str,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        character: Optional[int] = None,
        prefer_lsp: bool = False,
        repo_filter: Optional[str] = None,
        include_declaration: bool = False,
    ) -> list[Reference]:
        """
        Find all references to a symbol.

        Args:
            symbol_name: Name or qualified name of the symbol.
            file_path: Optional file path for context.
            line: Optional line number (0-indexed) for LSP lookup.
            character: Optional character position for LSP lookup.
            prefer_lsp: If True, prioritize LSP even if slower.
            repo_filter: Optional repository filter.
            include_declaration: Include the declaration in results.

        Returns:
            List of references to the symbol.
        """
        cache_key = f"refs:{symbol_name}:{file_path}:{line}:{character}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        references = []

        # Try LSP first if available and position is known
        lsp_refs = []
        if file_path and self.is_lsp_available(file_path) and line is not None:
            try:
                lsp_refs = await asyncio.wait_for(
                    self._find_references_lsp(
                        file_path, line, character or 0, include_declaration
                    ),
                    timeout=self.lsp_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"LSP reference lookup timed out for {symbol_name}")

        # Also get tree-sitter results for comparison/fallback
        ts_refs = self._find_references_tree_sitter(symbol_name, repo_filter)

        if prefer_lsp and lsp_refs:
            references = lsp_refs
        elif lsp_refs and ts_refs:
            # Merge results, preferring LSP for higher confidence
            references = self._merge_references(lsp_refs, ts_refs)
        elif lsp_refs:
            references = lsp_refs
        else:
            references = ts_refs

        if references and self.cache_enabled:
            self._cache_put(cache_key, references)

        return references

    async def find_implementations(
        self,
        interface_name: str,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        character: Optional[int] = None,
        prefer_lsp: bool = False,
        repo_filter: Optional[str] = None,
        include_indirect: bool = False,
    ) -> list[ResolvedSymbol]:
        """
        Find implementations of an interface or abstract class.

        Args:
            interface_name: Name of the interface/abstract class.
            file_path: Optional file path for context.
            line: Optional line number for LSP lookup.
            character: Optional character position for LSP lookup.
            prefer_lsp: If True, prioritize LSP.
            repo_filter: Optional repository filter.
            include_indirect: Include transitive implementations.

        Returns:
            List of implementing classes/methods.
        """
        cache_key = f"impl:{interface_name}:{file_path}:{include_indirect}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        implementations = []

        # Try LSP if position info available
        if file_path and self.is_lsp_available(file_path) and line is not None:
            try:
                lsp_impls = await asyncio.wait_for(
                    self._find_implementations_lsp(file_path, line, character or 0),
                    timeout=self.lsp_timeout,
                )
                implementations.extend(lsp_impls)
            except asyncio.TimeoutError:
                logger.warning(f"LSP implementation lookup timed out")

        # Always supplement with tree-sitter for indirect implementations
        ts_impls = self._find_implementations_tree_sitter(
            interface_name, repo_filter, include_indirect
        )

        # Merge results
        if implementations:
            # Add tree-sitter results not found by LSP
            seen_names = {i.qualified_name for i in implementations}
            for impl in ts_impls:
                if impl.qualified_name not in seen_names:
                    implementations.append(impl)
        else:
            implementations = ts_impls

        if implementations and self.cache_enabled:
            self._cache_put(cache_key, implementations)

        return implementations

    async def get_type_hierarchy(
        self,
        symbol_name: str,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        character: Optional[int] = None,
        repo_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get the complete type hierarchy for a symbol.

        Returns both supertypes (parents) and subtypes (children).

        Args:
            symbol_name: Name of the class/interface.
            file_path: Optional file path for context.
            line: Optional line number for LSP lookup.
            character: Optional character position for LSP lookup.
            repo_filter: Optional repository filter.

        Returns:
            Dictionary with 'supertypes' and 'subtypes' lists.
        """
        hierarchy = {
            "symbol": symbol_name,
            "supertypes": [],
            "subtypes": [],
            "backend": ResolverBackend.TREE_SITTER.value,
        }

        # Try LSP for precise hierarchy
        if file_path and self.is_lsp_available(file_path) and line is not None:
            try:
                lsp_hierarchy = await asyncio.wait_for(
                    self._lsp_manager.get_type_hierarchy(
                        file_path, line, character or 0
                    ),
                    timeout=self.lsp_timeout,
                )
                if lsp_hierarchy:
                    hierarchy["backend"] = ResolverBackend.LSP.value
                    if "supertypes" in lsp_hierarchy:
                        hierarchy["supertypes"] = [
                            {"name": s.get("name"), "file": s.get("uri")}
                            for s in lsp_hierarchy["supertypes"]
                        ]
                    if "subtypes" in lsp_hierarchy:
                        hierarchy["subtypes"] = [
                            {"name": s.get("name"), "file": s.get("uri")}
                            for s in lsp_hierarchy["subtypes"]
                        ]
                    return hierarchy
            except asyncio.TimeoutError:
                logger.warning("LSP type hierarchy lookup timed out")

        # Fall back to tree-sitter
        if self._symbol_table:
            # Get parents
            parents = self._symbol_table.find_parents(symbol_name, repo_filter)
            hierarchy["supertypes"] = [
                {"name": p.get("parent_name"), "relation": p.get("relation_type")}
                for p in parents
            ]

            # Get children
            children = self._symbol_table.find_implementations(symbol_name, repo_filter)
            hierarchy["subtypes"] = [
                {"name": c.get("child_name"), "relation": c.get("relation_type")}
                for c in children
            ]

        return hierarchy

    # =========================================================================
    # LSP Backend Methods
    # =========================================================================

    async def _find_definition_lsp(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> Optional[Definition]:
        """Find definition using LSP."""
        if not self._lsp_manager:
            return None

        location = await self._lsp_manager.goto_definition(file_path, line, character)
        if not location:
            return None

        # Convert URI to file path
        uri = location.uri
        if uri.startswith("file://"):
            resolved_path = uri[7:]  # Remove file:// prefix
        else:
            resolved_path = uri

        return Definition(
            symbol=ResolvedSymbol(
                name="<resolved>",  # LSP doesn't always give us the name
                qualified_name="<resolved>",
                file_path=resolved_path,
                line=location.range.start.line,
                character=location.range.start.character,
                symbol_type="unknown",
                backend=ResolverBackend.LSP,
                confidence=1.0,
            )
        )

    async def _find_references_lsp(
        self,
        file_path: str,
        line: int,
        character: int,
        include_declaration: bool,
    ) -> list[Reference]:
        """Find references using LSP."""
        if not self._lsp_manager:
            return []

        locations = await self._lsp_manager.find_references(
            file_path, line, character, include_declaration
        )

        references = []
        for loc in locations:
            uri = loc.uri
            if uri.startswith("file://"):
                resolved_path = uri[7:]
            else:
                resolved_path = uri

            references.append(
                Reference(
                    file_path=resolved_path,
                    line=loc.range.start.line,
                    character=loc.range.start.character,
                    context="",  # Would need to read file for context
                    reference_type="unknown",
                    backend=ResolverBackend.LSP,
                    confidence=1.0,
                )
            )

        return references

    async def _find_implementations_lsp(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> list[ResolvedSymbol]:
        """Find implementations using LSP."""
        if not self._lsp_manager:
            return []

        locations = await self._lsp_manager.find_implementations(
            file_path, line, character
        )

        symbols = []
        for loc in locations:
            uri = loc.uri
            if uri.startswith("file://"):
                resolved_path = uri[7:]
            else:
                resolved_path = uri

            symbols.append(
                ResolvedSymbol(
                    name="<implementation>",
                    qualified_name="<implementation>",
                    file_path=resolved_path,
                    line=loc.range.start.line,
                    character=loc.range.start.character,
                    symbol_type="class",
                    backend=ResolverBackend.LSP,
                    confidence=1.0,
                )
            )

        return symbols

    # =========================================================================
    # Tree-sitter Backend Methods
    # =========================================================================

    def _find_definition_tree_sitter(
        self,
        symbol_name: str,
        repo_filter: Optional[str] = None,
    ) -> Optional[Definition]:
        """Find definition using tree-sitter based symbol table."""
        if not self._symbol_table:
            return None

        symbols = self._symbol_table.lookup(symbol_name, repo_filter)
        if not symbols:
            return None

        sym = symbols[0]
        return Definition(
            symbol=ResolvedSymbol(
                name=sym.get("name", symbol_name),
                qualified_name=sym.get("qualified_name", symbol_name),
                file_path=sym.get("file", ""),
                line=sym.get("line", 0),
                character=0,
                symbol_type=sym.get("type", "unknown"),
                backend=ResolverBackend.TREE_SITTER,
                confidence=0.9,  # Tree-sitter is slightly less precise
            ),
            signature=sym.get("signature"),
            docstring=sym.get("docstring"),
        )

    def _find_references_tree_sitter(
        self,
        symbol_name: str,
        repo_filter: Optional[str] = None,
    ) -> list[Reference]:
        """Find references using call graph service."""
        if not self._call_graph:
            return []

        # Get callers from call graph
        callers = self._call_graph.find_callers(symbol_name, repo_filter)

        references = []
        for caller in callers:
            references.append(
                Reference(
                    file_path=caller.caller_file,
                    line=caller.caller_line,
                    character=0,
                    context="",
                    reference_type="call",
                    backend=ResolverBackend.TREE_SITTER,
                    confidence=0.85,
                )
            )

        return references

    def _find_implementations_tree_sitter(
        self,
        interface_name: str,
        repo_filter: Optional[str] = None,
        include_indirect: bool = False,
    ) -> list[ResolvedSymbol]:
        """Find implementations using symbol table."""
        if not self._symbol_table:
            return []

        impls = self._symbol_table.find_implementations(
            interface_name, repo_filter, include_indirect
        )

        symbols = []
        for impl in impls:
            symbols.append(
                ResolvedSymbol(
                    name=impl.get("child_name", ""),
                    qualified_name=impl.get("child_qualified", ""),
                    file_path=impl.get("file", ""),
                    line=impl.get("line", 0),
                    character=0,
                    symbol_type="class",
                    backend=ResolverBackend.TREE_SITTER,
                    confidence=0.9,
                )
            )

        return symbols

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _merge_references(
        self,
        lsp_refs: list[Reference],
        ts_refs: list[Reference],
    ) -> list[Reference]:
        """Merge LSP and tree-sitter references, removing duplicates."""
        # Use file:line as key for deduplication
        seen = set()
        merged = []

        # Add LSP refs first (higher confidence)
        for ref in lsp_refs:
            key = f"{ref.file_path}:{ref.line}"
            if key not in seen:
                seen.add(key)
                merged.append(ref)

        # Add tree-sitter refs not already found
        for ref in ts_refs:
            key = f"{ref.file_path}:{ref.line}"
            if key not in seen:
                seen.add(key)
                merged.append(ref)

        return merged

    def _cache_put(self, key: str, value: Any):
        """Put a value in the cache, evicting old entries if needed."""
        if len(self._cache) >= self._cache_max_size:
            # Simple eviction: remove first 10% of entries
            evict_count = self._cache_max_size // 10
            for _ in range(evict_count):
                if self._cache:
                    self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value


# Convenience function for one-off resolution
async def resolve_symbol(
    symbol_name: str,
    workspace_root: str,
    prefer_lsp: bool = False,
) -> Optional[Definition]:
    """
    One-off symbol resolution.

    Convenience function that creates a resolver, resolves the symbol,
    and cleans up. For repeated resolution, use HybridSymbolResolver directly.

    Args:
        symbol_name: Name of the symbol to resolve.
        workspace_root: Root directory of the project.
        prefer_lsp: If True, prioritize LSP backend.

    Returns:
        Definition if found, None otherwise.
    """
    resolver = HybridSymbolResolver(lsp_enabled=prefer_lsp)
    try:
        await resolver.initialize(workspace_root)
        return await resolver.find_definition(symbol_name, prefer_lsp=prefer_lsp)
    finally:
        await resolver.shutdown()
