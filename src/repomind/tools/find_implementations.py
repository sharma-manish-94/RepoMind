"""Find implementations tool for locating classes that extend/implement a type.

Uses the inheritance table to find all classes that implement an interface
or extend a base class, with support for transitive implementations.
"""

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from ..services.symbol_table import SymbolTableService

console = Console(width=200, force_terminal=False)


def find_implementations(
    interface_name: str,
    repo_filter: Optional[str] = None,
    include_indirect: bool = False,
) -> dict:
    """Find classes implementing an interface or extending a base class.

    This tool uses the inheritance table built during indexing to find
    all classes that implement an interface or extend a base class.

    Args:
        interface_name: Name of the interface or base class to find implementations for
        repo_filter: Filter to a specific repository
        include_indirect: If True, include transitive implementations (classes that
                         extend classes that implement the interface)

    Returns:
        Dictionary containing:
        - interface: The searched interface/class name
        - implementations: List of implementing classes with details
        - extends: List of classes that directly extend (not implement)

    Example:
        >>> find_implementations("BaseParser")
        {
            "interface": "BaseParser",
            "implementations": [
                {"name": "PythonParser", "file": "parsers/python_parser.py", "line": 10}
            ],
            "extends": [...]
        }
    """
    if not interface_name.strip():
        return {"error": "Interface/class name cannot be empty"}

    console.print(f"[bold blue]Finding implementations of: {interface_name}[/bold blue]")

    symbol_service = SymbolTableService()

    # Find all implementations
    implementations = symbol_service.find_implementations(
        parent_name=interface_name,
        repo_name=repo_filter,
        include_indirect=include_indirect,
    )

    if not implementations:
        return {
            "interface": interface_name,
            "found": False,
            "message": f"No implementations found for '{interface_name}'",
            "suggestion": "Ensure the repository is indexed and the name is correct",
        }

    # Separate by relation type
    extends_list = []
    implements_list = []

    for impl in implementations:
        entry = {
            "name": impl["child_name"],
            "qualified_name": impl["child_qualified"],
            "file": impl["file_path"],
            "line": impl["line_number"],
            "repo": impl["repo_name"],
        }

        if include_indirect and impl.get("indirect"):
            entry["indirect"] = True

        if impl["relation_type"] == "implements":
            implements_list.append(entry)
        else:  # extends
            extends_list.append(entry)

    result = {
        "interface": interface_name,
        "found": True,
        "total_count": len(implementations),
        "implements": implements_list if implements_list else None,
        "extends": extends_list if extends_list else None,
    }

    # Display results
    _display_implementations(result, include_indirect)

    return result


def find_hierarchy(
    class_name: str,
    repo_filter: Optional[str] = None,
) -> dict:
    """Find the complete type hierarchy for a class.

    Shows both what a class extends/implements (parents) and what
    extends/implements it (children).

    Args:
        class_name: Name of the class to analyze
        repo_filter: Filter to a specific repository

    Returns:
        Dictionary containing:
        - class: The analyzed class name
        - parents: Classes/interfaces this class extends/implements
        - children: Classes that extend/implement this class
    """
    if not class_name.strip():
        return {"error": "Class name cannot be empty"}

    console.print(f"[bold blue]Finding hierarchy for: {class_name}[/bold blue]")

    symbol_service = SymbolTableService()

    # Find parents (what this class extends/implements)
    parents = symbol_service.find_parents(class_name, repo_filter)

    # Find children (what extends/implements this class)
    children = symbol_service.find_implementations(class_name, repo_filter)

    parents_list = [
        {
            "name": p["parent_name"],
            "relation": p["relation_type"],
        }
        for p in parents
    ]

    children_list = [
        {
            "name": c["child_name"],
            "relation": c["relation_type"],
            "file": c["file_path"],
            "line": c["line_number"],
            "repo": c["repo_name"],
        }
        for c in children
    ]

    result = {
        "class": class_name,
        "parents": parents_list if parents_list else None,
        "children": children_list if children_list else None,
    }

    # Display results
    _display_hierarchy(result)

    return result


def _display_implementations(result: dict, include_indirect: bool) -> None:
    """Display implementation results in a formatted table."""
    if not result.get("found"):
        console.print(f"[yellow]{result.get('message', 'No results found')}[/yellow]")
        return

    console.print(f"\n[bold]Implementations of '{result['interface']}'[/bold]")
    console.print(f"[dim]Found {result['total_count']} implementation(s)[/dim]\n")

    # Display implements
    implements = result.get("implements")
    if implements:
        console.print("[bold green]Implements:[/bold green]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("Class", style="cyan")
        table.add_column("File", style="dim")
        table.add_column("Line", style="green", justify="right")
        if include_indirect:
            table.add_column("Direct", style="yellow")

        for impl in implements:
            row = [
                impl["name"],
                f"{impl['repo']}/{impl['file']}" if impl.get('repo') else impl['file'],
                str(impl["line"]),
            ]
            if include_indirect:
                row.append("No" if impl.get("indirect") else "Yes")
            table.add_row(*row)

        console.print(table)
        console.print()

    # Display extends
    extends = result.get("extends")
    if extends:
        console.print("[bold magenta]Extends:[/bold magenta]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("Class", style="cyan")
        table.add_column("File", style="dim")
        table.add_column("Line", style="green", justify="right")
        if include_indirect:
            table.add_column("Direct", style="yellow")

        for ext in extends:
            row = [
                ext["name"],
                f"{ext['repo']}/{ext['file']}" if ext.get('repo') else ext['file'],
                str(ext["line"]),
            ]
            if include_indirect:
                row.append("No" if ext.get("indirect") else "Yes")
            table.add_row(*row)

        console.print(table)


def _display_hierarchy(result: dict) -> None:
    """Display class hierarchy as a tree."""
    class_name = result["class"]

    console.print(f"\n[bold]Type Hierarchy for '{class_name}'[/bold]\n")

    # Build tree
    tree = Tree(f"[bold cyan]{class_name}[/bold cyan]")

    # Add parents (what this class extends/implements)
    parents = result.get("parents")
    if parents:
        parents_branch = tree.add("[dim]extends/implements →[/dim]")
        for parent in parents:
            icon = "🔵" if parent["relation"] == "implements" else "🟢"
            parents_branch.add(f"{icon} {parent['name']} [dim]({parent['relation']})[/dim]")

    # Add children (what extends/implements this)
    children = result.get("children")
    if children:
        children_branch = tree.add("[dim]← extended/implemented by[/dim]")
        for child in children:
            icon = "🔵" if child["relation"] == "implements" else "🟢"
            children_branch.add(
                f"{icon} {child['name']} [dim]({child['relation']})[/dim] "
                f"[dim]{child['file']}:{child['line']}[/dim]"
            )

    console.print(tree)

    if not parents and not children:
        console.print("[yellow]No inheritance relationships found for this class.[/yellow]")
