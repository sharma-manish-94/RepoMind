"""File summary tool for getting overview of symbols in a file.

Provides a quick overview of all symbols (classes, functions, methods) in a file
without needing to read the entire content. Uses the symbol table for fast lookups.
"""

from typing import Optional

from rich.console import Console
from rich.tree import Tree

from ..services.symbol_table import Symbol, SymbolTableService

console = Console(width=200, force_terminal=False)


def file_summary(
    file_path: str,
    repo_name: Optional[str] = None,
) -> dict:
    """Get overview of symbols in a file without reading entire content.

    This tool provides a structural overview of a file, showing all classes,
    functions, and methods with their signatures and line numbers. Useful for
    understanding file structure before diving into specific code.

    Args:
        file_path: Path to the file (relative to repo root)
        repo_name: Filter to a specific repository (optional if file path is unique)

    Returns:
        Dictionary containing:
        - file: The file path
        - repo: The repository name
        - symbols: Nested structure of classes/functions with their members
        - summary: Quick stats about the file

    Example:
        >>> file_summary("src/services/auth.py", "my-repo")
        {
            "file": "src/services/auth.py",
            "repo": "my-repo",
            "symbols": {
                "classes": [
                    {
                        "name": "AuthService",
                        "line": 15,
                        "signature": "class AuthService:",
                        "methods": [
                            {"name": "validate", "line": 20, "signature": "def validate(...)"}
                        ]
                    }
                ],
                "functions": [
                    {"name": "create_token", "line": 50, "signature": "def create_token(...)"}
                ]
            }
        }
    """
    if not file_path.strip():
        return {"error": "File path cannot be empty"}

    symbol_service = SymbolTableService()

    # If repo_name not provided, try to find the file across all repos
    if repo_name:
        symbols = symbol_service.lookup_by_file(repo_name, file_path)
        if not symbols:
            return {
                "error": f"No symbols found in {file_path} for repository {repo_name}",
                "file": file_path,
                "repo": repo_name,
            }
    else:
        # Search across all repos - need to find matching file
        all_symbols = _find_file_across_repos(symbol_service, file_path)
        if not all_symbols:
            return {
                "error": f"No symbols found for file matching: {file_path}",
                "suggestion": "Ensure the repository is indexed and the file path is correct",
            }
        symbols = all_symbols
        repo_name = symbols[0].repo_name if symbols else None

    # Group symbols by type and parent
    result = _organize_symbols(symbols, file_path, repo_name)

    # Display to console
    _display_file_summary(result)

    return result


def _find_file_across_repos(symbol_service: SymbolTableService, file_path: str) -> list[Symbol]:
    """Find symbols for a file path across all repositories.

    Searches for files that end with the given path to handle both full and partial paths.
    """
    conn = symbol_service._get_connection()
    try:
        # Try exact match first
        cursor = conn.execute(
            """
            SELECT DISTINCT repo_name, file_path FROM symbols
            WHERE file_path = ?
            LIMIT 1
            """,
            (file_path,),
        )
        row = cursor.fetchone()

        if not row:
            # Try partial match (file ends with the given path)
            cursor = conn.execute(
                """
                SELECT DISTINCT repo_name, file_path FROM symbols
                WHERE file_path LIKE ?
                LIMIT 1
                """,
                (f"%{file_path}",),
            )
            row = cursor.fetchone()

        if row:
            return symbol_service.lookup_by_file(row["repo_name"], row["file_path"])

        return []
    finally:
        conn.close()


def _organize_symbols(symbols: list[Symbol], file_path: str, repo_name: str) -> dict:
    """Organize symbols into a nested structure.

    Groups symbols by type and nests class members under their parent classes.
    """
    classes = []
    functions = []
    interfaces = []
    other = []

    # First pass: collect top-level symbols
    class_map = {}  # name -> class info
    for symbol in symbols:
        if symbol.parent_name:
            continue  # Handle nested symbols later

        symbol_info = {
            "name": symbol.name,
            "qualified_name": symbol.qualified_name,
            "line": symbol.start_line,
            "end_line": symbol.end_line,
            "signature": symbol.signature,
            "type": symbol.symbol_type,
        }

        if symbol.symbol_type == "class":
            symbol_info["methods"] = []
            symbol_info["properties"] = []
            classes.append(symbol_info)
            class_map[symbol.name] = symbol_info
        elif symbol.symbol_type == "interface":
            symbol_info["methods"] = []
            interfaces.append(symbol_info)
            class_map[symbol.name] = symbol_info
        elif symbol.symbol_type == "function":
            functions.append(symbol_info)
        else:
            other.append(symbol_info)

    # Second pass: nest members under their parent classes
    for symbol in symbols:
        if not symbol.parent_name:
            continue

        member_info = {
            "name": symbol.name,
            "line": symbol.start_line,
            "end_line": symbol.end_line,
            "signature": symbol.signature,
            "type": symbol.symbol_type,
        }

        if symbol.parent_name in class_map:
            parent = class_map[symbol.parent_name]
            if symbol.symbol_type in ("method", "function", "constructor"):
                parent["methods"].append(member_info)
            else:
                parent.setdefault("properties", []).append(member_info)

    # Sort everything by line number
    classes.sort(key=lambda x: x["line"])
    functions.sort(key=lambda x: x["line"])
    interfaces.sort(key=lambda x: x["line"])
    for cls in classes + interfaces:
        cls["methods"].sort(key=lambda x: x["line"])

    # Calculate summary stats
    total_methods = sum(len(c.get("methods", [])) for c in classes + interfaces)
    lines_covered = max((s.end_line for s in symbols), default=0)

    return {
        "file": file_path,
        "repo": repo_name,
        "symbols": {
            "classes": classes,
            "interfaces": interfaces,
            "functions": functions,
            "other": other if other else None,
        },
        "summary": {
            "total_classes": len(classes),
            "total_interfaces": len(interfaces),
            "total_functions": len(functions),
            "total_methods": total_methods,
            "lines": lines_covered,
        },
    }


def _display_file_summary(result: dict) -> None:
    """Display file summary using Rich tree visualization."""
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    console.print(f"\n[bold blue]File Summary: {result['file']}[/bold blue]")
    console.print(f"[dim]Repository: {result['repo']}[/dim]\n")

    tree = Tree(f"[bold]{result['file']}[/bold]")

    symbols = result["symbols"]

    # Add interfaces
    for interface in symbols.get("interfaces", []):
        branch = tree.add(f"[cyan]interface[/cyan] [bold]{interface['name']}[/bold] [dim]:{interface['line']}[/dim]")
        for method in interface.get("methods", []):
            sig = method.get("signature", method["name"])
            if sig and len(sig) > 60:
                sig = sig[:57] + "..."
            branch.add(f"[green]{method['name']}[/green] [dim]:{method['line']}[/dim]")

    # Add classes
    for cls in symbols.get("classes", []):
        branch = tree.add(f"[yellow]class[/yellow] [bold]{cls['name']}[/bold] [dim]:{cls['line']}[/dim]")
        for method in cls.get("methods", []):
            method_type = "[magenta]constructor[/magenta]" if method["type"] == "constructor" else "[green]method[/green]"
            branch.add(f"{method_type} {method['name']} [dim]:{method['line']}[/dim]")

    # Add functions
    for func in symbols.get("functions", []):
        tree.add(f"[blue]function[/blue] [bold]{func['name']}[/bold] [dim]:{func['line']}[/dim]")

    console.print(tree)

    # Print summary
    summary = result["summary"]
    console.print(
        f"\n[dim]Summary: {summary['total_classes']} classes, "
        f"{summary['total_interfaces']} interfaces, "
        f"{summary['total_functions']} functions, "
        f"{summary['total_methods']} methods[/dim]"
    )
