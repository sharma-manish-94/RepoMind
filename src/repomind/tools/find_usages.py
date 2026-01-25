"""Find usages tool for locating all references to a symbol.

Finds ALL references to a symbol, not just function calls. Includes:
- Function/method calls
- Type annotations
- Inheritance (extends, implements)
- Variable assignments
- Import statements
"""

import re
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..services.call_graph import CallGraphService
from ..services.storage import StorageService
from ..services.symbol_table import SymbolTableService

console = Console(width=200, force_terminal=False)


def find_usages(
    symbol_name: str,
    repo_filter: Optional[str] = None,
    include_definitions: bool = False,
    limit: int = 50,
) -> dict:
    """Find ALL references to a symbol across the codebase.

    Unlike find_callers which only finds call sites, this tool finds every
    reference to a symbol including:
    - Direct function/method calls
    - Type annotations (param: MyClass, return type hints)
    - Inheritance (class Child(Parent))
    - Variable assignments and type declarations
    - Import statements

    Args:
        symbol_name: Name of the symbol to find usages for
        repo_filter: Filter to a specific repository
        include_definitions: If True, also include where the symbol is defined
        limit: Maximum number of usages to return (default: 50)

    Returns:
        Dictionary containing:
        - symbol: The searched symbol name
        - total_usages: Total count of usages found
        - usages: List of usages grouped by type

    Example:
        >>> find_usages("UserService")
        {
            "symbol": "UserService",
            "total_usages": 15,
            "usages": {
                "calls": [...],
                "type_hints": [...],
                "inheritance": [...],
                "imports": [...]
            }
        }
    """
    if not symbol_name.strip():
        return {"error": "Symbol name cannot be empty"}

    console.print(f"[bold blue]Finding usages of: {symbol_name}[/bold blue]")

    call_graph = CallGraphService()
    storage_service = StorageService()
    symbol_service = SymbolTableService()

    usages = {
        "calls": [],
        "type_hints": [],
        "inheritance": [],
        "imports": [],
        "assignments": [],
        "definitions": [],
    }

    # 1. Find call usages via CallGraphService
    call_relations = call_graph.find_callers(symbol_name, repo_filter)
    for relation in call_relations[:limit]:
        usages["calls"].append({
            "file": relation.caller_file,
            "line": relation.caller_line,
            "caller": relation.caller,
            "call_type": relation.call_type,
            "repo": relation.repo_name,
        })

    # 2. Search chunk content for other usage types
    chunk_map = storage_service._load_chunk_metadata()

    for chunk_id, chunk in chunk_map.items():
        if repo_filter and chunk.repo_name != repo_filter:
            continue

        # Skip if this is the definition and we don't want definitions
        if chunk.name == symbol_name or chunk.get_qualified_name() == symbol_name:
            if include_definitions:
                usages["definitions"].append({
                    "file": chunk.file_path,
                    "line": chunk.start_line,
                    "name": chunk.name,
                    "type": chunk.chunk_type.value,
                    "repo": chunk.repo_name,
                })
            continue

        # Search for usages in the chunk content
        content = chunk.content
        if symbol_name not in content:
            continue

        # Check for type annotations
        type_hint_patterns = [
            rf":\s*{re.escape(symbol_name)}\b",  # param: SymbolName
            rf"->\s*{re.escape(symbol_name)}\b",  # -> SymbolName
            rf":\s*Optional\[{re.escape(symbol_name)}\]",  # : Optional[SymbolName]
            rf":\s*list\[{re.escape(symbol_name)}\]",  # : list[SymbolName]
            rf":\s*List\[{re.escape(symbol_name)}\]",  # : List[SymbolName]
            rf":\s*dict\[.*,\s*{re.escape(symbol_name)}\]",  # : dict[..., SymbolName]
            rf"<{re.escape(symbol_name)}>",  # Java generics: List<SymbolName>
        ]
        for pattern in type_hint_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                for match in matches:
                    line_num = chunk.start_line + content[:match.start()].count('\n')
                    if not _usage_exists(usages["type_hints"], chunk.file_path, line_num, chunk.repo_name):
                        usages["type_hints"].append({
                            "file": chunk.file_path,
                            "line": line_num,
                            "context": _get_line_context(content, match.start()),
                            "chunk": chunk.name,
                            "repo": chunk.repo_name,
                        })

        # Check for inheritance
        inheritance_patterns = [
            rf"class\s+\w+\s*\({re.escape(symbol_name)}[,)]",  # Python: class X(SymbolName)
            rf"class\s+\w+\s+extends\s+{re.escape(symbol_name)}\b",  # Java/TS: extends
            rf"class\s+\w+.*implements\s+.*{re.escape(symbol_name)}\b",  # Java: implements
        ]
        for pattern in inheritance_patterns:
            matches = list(re.finditer(pattern, content))
            if matches:
                for match in matches:
                    line_num = chunk.start_line + content[:match.start()].count('\n')
                    if not _usage_exists(usages["inheritance"], chunk.file_path, line_num, chunk.repo_name):
                        usages["inheritance"].append({
                            "file": chunk.file_path,
                            "line": line_num,
                            "context": match.group(0).strip(),
                            "repo": chunk.repo_name,
                        })

        # Check for imports
        import_patterns = [
            rf"from\s+\S+\s+import\s+.*{re.escape(symbol_name)}",  # Python from import
            rf"import\s+.*{re.escape(symbol_name)}",  # Generic import
        ]
        for pattern in import_patterns:
            matches = list(re.finditer(pattern, content))
            if matches:
                for match in matches:
                    line_num = chunk.start_line + content[:match.start()].count('\n')
                    if not _usage_exists(usages["imports"], chunk.file_path, line_num, chunk.repo_name):
                        usages["imports"].append({
                            "file": chunk.file_path,
                            "line": line_num,
                            "statement": match.group(0).strip()[:80],
                            "repo": chunk.repo_name,
                        })

        # Check for assignments/instantiation
        assignment_patterns = [
            rf"\w+\s*=\s*{re.escape(symbol_name)}\(",  # var = SymbolName()
            rf"\w+\s*:\s*{re.escape(symbol_name)}\s*=",  # var: SymbolName =
            rf"new\s+{re.escape(symbol_name)}\(",  # new SymbolName()
        ]
        for pattern in assignment_patterns:
            matches = list(re.finditer(pattern, content))
            if matches:
                for match in matches:
                    line_num = chunk.start_line + content[:match.start()].count('\n')
                    if not _usage_exists(usages["assignments"], chunk.file_path, line_num, chunk.repo_name):
                        usages["assignments"].append({
                            "file": chunk.file_path,
                            "line": line_num,
                            "context": _get_line_context(content, match.start()),
                            "repo": chunk.repo_name,
                        })

    # Apply limit to each category
    for key in usages:
        usages[key] = usages[key][:limit]

    # Calculate total
    total = sum(len(v) for v in usages.values())

    # Remove empty categories
    usages = {k: v for k, v in usages.items() if v}

    result = {
        "symbol": symbol_name,
        "total_usages": total,
        "usages": usages,
    }

    # Display results
    _display_usages(result)

    return result


def _usage_exists(usage_list: list, file_path: str, line: int, repo: str) -> bool:
    """Check if a usage already exists in the list to avoid duplicates."""
    for usage in usage_list:
        if usage.get("file") == file_path and usage.get("line") == line and usage.get("repo") == repo:
            return True
    return False


def _get_line_context(content: str, position: int, max_length: int = 80) -> str:
    """Extract the line containing the position from content."""
    start = content.rfind('\n', 0, position) + 1
    end = content.find('\n', position)
    if end == -1:
        end = len(content)

    line = content[start:end].strip()
    if len(line) > max_length:
        # Try to center around the position
        rel_pos = position - start
        half = max_length // 2
        if rel_pos < half:
            line = line[:max_length - 3] + "..."
        elif rel_pos > len(line) - half:
            line = "..." + line[-(max_length - 3):]
        else:
            line = "..." + line[rel_pos - half:rel_pos + half] + "..."

    return line


def _display_usages(result: dict) -> None:
    """Display usages in a formatted table."""
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    console.print(f"\n[bold]Found {result['total_usages']} usages of '{result['symbol']}'[/bold]\n")

    usages = result.get("usages", {})

    # Display definitions first
    if usages.get("definitions"):
        console.print("[bold cyan]Definitions:[/bold cyan]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Line", style="green", justify="right")
        table.add_column("Type", style="yellow")
        for usage in usages["definitions"]:
            table.add_row(
                f"{usage['repo']}/{usage['file']}" if usage.get('repo') else usage['file'],
                str(usage['line']),
                usage['type'],
            )
        console.print(table)
        console.print()

    # Display call usages
    if usages.get("calls"):
        console.print("[bold green]Function Calls:[/bold green]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Line", style="green", justify="right")
        table.add_column("Caller", style="yellow")
        for usage in usages["calls"][:10]:
            table.add_row(
                f"{usage['repo']}/{usage['file']}" if usage.get('repo') else usage['file'],
                str(usage['line']),
                usage['caller'],
            )
        if len(usages["calls"]) > 10:
            console.print(f"[dim]... and {len(usages['calls']) - 10} more calls[/dim]")
        console.print(table)
        console.print()

    # Display type hints
    if usages.get("type_hints"):
        console.print("[bold magenta]Type Hints:[/bold magenta]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Line", style="green", justify="right")
        table.add_column("Context", style="white")
        for usage in usages["type_hints"][:10]:
            table.add_row(
                f"{usage['repo']}/{usage['file']}" if usage.get('repo') else usage['file'],
                str(usage['line']),
                usage.get('context', '')[:50],
            )
        if len(usages["type_hints"]) > 10:
            console.print(f"[dim]... and {len(usages['type_hints']) - 10} more type hints[/dim]")
        console.print(table)
        console.print()

    # Display inheritance
    if usages.get("inheritance"):
        console.print("[bold yellow]Inheritance:[/bold yellow]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Line", style="green", justify="right")
        table.add_column("Declaration", style="white")
        for usage in usages["inheritance"]:
            table.add_row(
                f"{usage['repo']}/{usage['file']}" if usage.get('repo') else usage['file'],
                str(usage['line']),
                usage.get('context', '')[:50],
            )
        console.print(table)
        console.print()

    # Display imports
    if usages.get("imports"):
        console.print("[bold blue]Imports:[/bold blue]")
        for usage in usages["imports"][:10]:
            console.print(f"  [dim]{usage['repo']}/[/dim]{usage['file']}:{usage['line']}")
        if len(usages["imports"]) > 10:
            console.print(f"[dim]... and {len(usages['imports']) - 10} more imports[/dim]")
