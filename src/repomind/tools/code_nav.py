"""Code navigation tools for MCP - symbol lookup and call graph queries."""

from typing import Optional

from ..services.call_graph import CallGraphService
from ..services.symbol_table import SymbolTableService


def lookup_symbol(
    name: str,
    exact: bool = True,
    symbol_type: Optional[str] = None,
    repo_name: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """Look up a symbol by name.

    Fast O(log n) lookup using the symbol table. Use this to find
    exact definitions of functions, classes, methods, etc.

    Args:
        name: Symbol name to search for (e.g., "MyClass", "process_request")
        exact: If True, exact match. If False, prefix search.
        symbol_type: Filter by type: "class", "method", "function", "interface"
        repo_name: Filter by repository name

    Returns:
        Dictionary with matching symbols and their locations
    """
    symbol_table = SymbolTableService()

    if symbol_type and not name:
        # Type-only lookup
        symbols = symbol_table.lookup_by_type(symbol_type, repo_name, limit)
    else:
        symbols = symbol_table.lookup(name, exact=exact)

        # Apply filters
        if symbol_type:
            symbols = [s for s in symbols if s.symbol_type == symbol_type]
        if repo_name:
            symbols = [s for s in symbols if s.repo_name == repo_name]

        symbols = symbols[:limit]

    return {
        "query": name,
        "exact": exact,
        "count": len(symbols),
        "symbols": [
            {
                "name": s.name,
                "qualified_name": s.qualified_name,
                "type": s.symbol_type,
                "file": s.file_path,
                "repo": s.repo_name,
                "line": s.start_line,
                "end_line": s.end_line,
                "signature": s.signature,
                "parent": s.parent_name,
                "language": s.language,
            }
            for s in symbols
        ],
    }


def find_callers(
    symbol: str,
    repo_name: Optional[str] = None,
    max_depth: int = 1,
) -> dict:
    """Find all functions/methods that call the given symbol.

    Use this for impact analysis - "if I change this function, what else
    might be affected?"

    Args:
        symbol: Qualified name of the function/method (e.g., "MyClass.my_method")
        repo_name: Filter by repository
        max_depth: How many levels up to traverse (1 = direct callers only)

    Returns:
        Dictionary with list of callers and their locations
    """
    call_graph = CallGraphService()
    callers = call_graph.find_callers(symbol, repo_name, max_depth)

    return {
        "symbol": symbol,
        "direction": "callers",
        "max_depth": max_depth,
        "count": len(callers),
        "callers": [
            {
                "caller": c.caller,
                "file": c.caller_file,
                "line": c.caller_line,
                "repo": c.repo_name,
                "call_type": c.call_type,
            }
            for c in callers
        ],
    }


def find_callees(
    symbol: str,
    repo_name: Optional[str] = None,
    max_depth: int = 1,
) -> dict:
    """Find all functions/methods called by the given symbol.

    Use this to understand dependencies - "what does this function depend on?"

    Args:
        symbol: Qualified name of the function/method
        repo_name: Filter by repository
        max_depth: How many levels down to traverse (1 = direct callees only)

    Returns:
        Dictionary with list of callees
    """
    call_graph = CallGraphService()
    callees = call_graph.find_callees(symbol, repo_name, max_depth)

    return {
        "symbol": symbol,
        "direction": "callees",
        "max_depth": max_depth,
        "count": len(callees),
        "callees": [
            {
                "callee": c.callee,
                "caller_file": c.caller_file,
                "caller_line": c.caller_line,
                "repo": c.repo_name,
                "call_type": c.call_type,
            }
            for c in callees
        ],
    }


def find_call_path(
    from_symbol: str,
    to_symbol: str,
    repo_name: Optional[str] = None,
    max_depth: int = 10,
) -> dict:
    """Find a call path from one symbol to another.

    Use this to trace how execution can flow from one function to another.

    Args:
        from_symbol: Starting function
        to_symbol: Target function
        repo_name: Filter by repository
        max_depth: Maximum path length to search

    Returns:
        Dictionary with the path (if found) or empty path if no connection
    """
    call_graph = CallGraphService()
    path = call_graph.find_path(from_symbol, to_symbol, repo_name, max_depth)

    return {
        "from": from_symbol,
        "to": to_symbol,
        "found": path is not None,
        "path": path or [],
        "length": len(path) if path else 0,
    }


def get_symbol_context(
    symbol: str,
    repo_name: Optional[str] = None,
) -> dict:
    """Get comprehensive context for a symbol.

    Returns the symbol definition along with its callers and callees.
    Useful for understanding a function's role in the codebase.

    Args:
        symbol: Symbol name or qualified name
        repo_name: Filter by repository

    Returns:
        Dictionary with symbol info, callers, and callees
    """
    symbol_table = SymbolTableService()
    call_graph = CallGraphService()

    # Look up the symbol
    symbols = symbol_table.lookup(symbol, exact=True)
    if repo_name:
        symbols = [s for s in symbols if s.repo_name == repo_name]

    if not symbols:
        return {
            "symbol": symbol,
            "found": False,
            "message": f"Symbol '{symbol}' not found in index",
        }

    # Get the first match
    sym = symbols[0]

    # Get call node (callers and callees)
    call_node = call_graph.get_call_node(sym.qualified_name, repo_name)

    return {
        "symbol": symbol,
        "found": True,
        "definition": {
            "name": sym.name,
            "qualified_name": sym.qualified_name,
            "type": sym.symbol_type,
            "file": sym.file_path,
            "repo": sym.repo_name,
            "line": sym.start_line,
            "end_line": sym.end_line,
            "signature": sym.signature,
            "parent": sym.parent_name,
            "language": sym.language,
        },
        "callers": call_node.callers[:20],  # Limit for readability
        "callees": call_node.callees[:20],
        "caller_count": len(call_node.callers),
        "callee_count": len(call_node.callees),
    }


def get_class_members(
    class_name: str,
    repo_name: Optional[str] = None,
) -> dict:
    """Get all members (methods, properties) of a class.

    Args:
        class_name: Name of the class
        repo_name: Filter by repository

    Returns:
        Dictionary with class info and its members
    """
    symbol_table = SymbolTableService()

    # Find the class
    symbols = symbol_table.lookup(class_name, exact=True)
    classes = [s for s in symbols if s.symbol_type == "class"]
    if repo_name:
        classes = [s for s in classes if s.repo_name == repo_name]

    if not classes:
        return {
            "class_name": class_name,
            "found": False,
            "message": f"Class '{class_name}' not found",
        }

    cls = classes[0]

    # Get children (methods, properties)
    members = symbol_table.lookup_children(cls.name, repo_name)

    return {
        "class_name": class_name,
        "found": True,
        "class": {
            "qualified_name": cls.qualified_name,
            "file": cls.file_path,
            "repo": cls.repo_name,
            "line": cls.start_line,
            "language": cls.language,
        },
        "members": [
            {
                "name": m.name,
                "type": m.symbol_type,
                "line": m.start_line,
                "signature": m.signature,
            }
            for m in members
        ],
        "member_count": len(members),
    }


def get_index_stats() -> dict:
    """Get statistics about the code index.

    Returns:
        Dictionary with stats about symbols, call graph, etc.
    """
    symbol_table = SymbolTableService()
    call_graph = CallGraphService()

    symbol_stats = symbol_table.get_stats()
    call_stats = call_graph.get_stats()

    return {
        "symbol_table": {
            "total_symbols": symbol_stats.get("total_symbols", 0),
            "by_type": symbol_stats.get("by_type", {}),
            "by_repo": symbol_stats.get("by_repo", {}),
            "by_language": symbol_stats.get("by_language", {}),
        },
        "call_graph": {
            "total_edges": call_stats.get("total_edges", 0),
            "unique_callers": call_stats.get("unique_callers", 0),
            "unique_callees": call_stats.get("unique_callees", 0),
            "by_repo": call_stats.get("by_repo", {}),
            "hotspots": call_stats.get("hotspots", []),
        },
    }
