"""Get context tool for MCP - retrieve code context for symbols."""

from typing import Optional

from rich.console import Console
from rich.syntax import Syntax

from ..services.storage import StorageService

# Use wide console to avoid truncation - important for MCP/CLI output
console = Console(width=200, force_terminal=False)


def get_context(
    symbol_name: str,
    repo_filter: Optional[str] = None,
    include_related: bool = True,
) -> dict:
    """Get full context for a code symbol (function, class, method).

    This tool retrieves the complete code for a specific symbol along with
    related context like the parent class, sibling methods, or imports.

    Args:
        symbol_name: Name of the symbol to look up (e.g., "handleRequest", "UserService")
        repo_filter: Filter to a specific repository
        include_related: Whether to include related code (parent class, siblings)

    Returns:
        Dictionary containing the symbol's code and related context
    """
    if not symbol_name.strip():
        return {"error": "Symbol name cannot be empty"}

    console.print(f"[bold blue]Looking up: {symbol_name}[/bold blue]")

    storage_service = StorageService()
    chunk_map = storage_service._load_chunk_metadata()

    # Find matching chunks
    matches = []
    for chunk_id, chunk in chunk_map.items():
        if repo_filter and chunk.repo_name != repo_filter:
            continue

        # Match by name or qualified name
        if (
            chunk.name == symbol_name
            or chunk.get_qualified_name() == symbol_name
            or symbol_name in chunk.name
        ):
            matches.append(chunk)

    if not matches:
        return {
            "symbol": symbol_name,
            "found": False,
            "message": f"No symbol found matching '{symbol_name}'",
        }

    # Sort by relevance (exact matches first)
    matches.sort(key=lambda c: (c.name != symbol_name, c.get_qualified_name() != symbol_name))

    results = []

    for chunk in matches[:5]:  # Limit to top 5 matches
        result = {
            "name": chunk.name,
            "qualified_name": chunk.get_qualified_name(),
            "type": chunk.chunk_type.value,
            "repo": chunk.repo_name,
            "file": chunk.file_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "language": chunk.language,
            "code": chunk.content,
        }

        if chunk.signature:
            result["signature"] = chunk.signature

        if chunk.docstring:
            result["docstring"] = chunk.docstring

        # Get related context
        if include_related:
            related = []

            # If this is a method, get the parent class
            if chunk.parent_name:
                parent_chunks = [
                    c
                    for c in chunk_map.values()
                    if c.name == chunk.parent_name
                    and c.repo_name == chunk.repo_name
                    and c.file_path == chunk.file_path
                ]
                for parent in parent_chunks:
                    related.append(
                        {
                            "relationship": "parent_class",
                            "name": parent.name,
                            "type": parent.chunk_type.value,
                            "signature": parent.signature,
                            "docstring": parent.docstring,
                        }
                    )

                # Get sibling methods
                siblings = [
                    c
                    for c in chunk_map.values()
                    if c.parent_name == chunk.parent_name
                    and c.repo_name == chunk.repo_name
                    and c.file_path == chunk.file_path
                    and c.name != chunk.name
                ]
                for sib in siblings[:10]:  # Limit siblings
                    related.append(
                        {
                            "relationship": "sibling_method",
                            "name": sib.name,
                            "type": sib.chunk_type.value,
                            "signature": sib.signature,
                        }
                    )

            # Get other symbols in the same file
            same_file = [
                c
                for c in chunk_map.values()
                if c.repo_name == chunk.repo_name
                and c.file_path == chunk.file_path
                and c.id != chunk.id
                and c.parent_name is None  # Only top-level
            ]
            for sf in same_file[:5]:
                related.append(
                    {
                        "relationship": "same_file",
                        "name": sf.name,
                        "type": sf.chunk_type.value,
                    }
                )

            if related:
                result["related"] = related

        results.append(result)

        # Display
        console.print(f"\n[bold green]{chunk.get_qualified_name()}[/bold green]")
        console.print(f"[dim]{chunk.repo_name}/{chunk.file_path}:{chunk.start_line}[/dim]")

        if chunk.docstring:
            # Show full docstring without truncation
            console.print(f"[italic]{chunk.docstring}[/italic]")

        syntax = Syntax(
            chunk.content,
            chunk.language,
            line_numbers=True,
            start_line=chunk.start_line,
        )
        console.print(syntax)

    return {
        "symbol": symbol_name,
        "found": True,
        "match_count": len(matches),
        "results": results,
    }
