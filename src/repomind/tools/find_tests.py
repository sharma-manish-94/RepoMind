"""Find tests tool for discovering tests related to a symbol.

Uses heuristics to find test files and test methods that test a specific
symbol (function, class, method). Supports Python, Java, and TypeScript/JavaScript
test conventions.
"""

import re
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..services.storage import StorageService
from ..services.symbol_table import SymbolTableService

console = Console(width=200, force_terminal=False)


def find_tests(
    symbol_name: str,
    repo_filter: Optional[str] = None,
) -> dict:
    """Find test files and methods for a symbol using heuristics.

    This tool discovers tests related to a symbol by checking:
    1. File name patterns: test_{symbol}.py, {Symbol}Test.java, {symbol}.spec.ts
    2. Test method patterns: def test_*{symbol}*, @Test methods referencing symbol
    3. Import analysis: Test files that import the symbol
    4. Content matching: Test code that references the symbol

    Args:
        symbol_name: Name of the symbol to find tests for
        repo_filter: Filter to a specific repository

    Returns:
        Dictionary containing:
        - symbol: The searched symbol name
        - test_files: List of test files that likely test this symbol
        - test_methods: List of specific test methods
        - match_reasons: Why each result was matched

    Example:
        >>> find_tests("UserService")
        {
            "symbol": "UserService",
            "test_files": [
                {"file": "test_user_service.py", "match": "filename_pattern"}
            ],
            "test_methods": [
                {"name": "test_user_service_validate", "file": "test_auth.py", "line": 45}
            ]
        }
    """
    if not symbol_name.strip():
        return {"error": "Symbol name cannot be empty"}

    console.print(f"[bold blue]Finding tests for: {symbol_name}[/bold blue]")

    storage_service = StorageService()
    symbol_service = SymbolTableService()

    chunk_map = storage_service._load_chunk_metadata()

    test_files = []
    test_methods = []
    seen_files = set()
    seen_methods = set()

    # Normalize symbol name for pattern matching
    symbol_lower = symbol_name.lower()
    symbol_snake = _to_snake_case(symbol_name)
    symbol_parts = _extract_words(symbol_name)

    # Define test file patterns
    test_file_patterns = [
        f"test_{symbol_snake}",
        f"{symbol_snake}_test",
        f"test_{symbol_lower}",
        f"{symbol_lower}_test",
        f"{symbol_name}Test",
        f"{symbol_name}Spec",
        f"{symbol_name}.test",
        f"{symbol_name}.spec",
    ]

    # Define test method patterns (regex)
    test_method_patterns = [
        rf"test.*{re.escape(symbol_snake)}",
        rf"test.*{re.escape(symbol_lower)}",
        rf"should.*{re.escape(symbol_lower)}",
        rf"it.*{re.escape(symbol_lower)}",
    ]

    for chunk_id, chunk in chunk_map.items():
        if repo_filter and chunk.repo_name != repo_filter:
            continue

        file_path = chunk.file_path
        file_lower = file_path.lower()
        file_name = file_path.split('/')[-1].lower()

        # Check if this is a test file
        is_test_file = any(
            pattern in file_lower
            for pattern in ['test_', '_test.', 'test.', '.spec.', '/tests/', '/test/']
        )

        if not is_test_file:
            continue

        # 1. Check file name pattern match
        file_key = (chunk.repo_name, file_path)
        if file_key not in seen_files:
            for pattern in test_file_patterns:
                if pattern.lower() in file_name:
                    seen_files.add(file_key)
                    test_files.append({
                        "file": file_path,
                        "repo": chunk.repo_name,
                        "match_reason": "filename_pattern",
                        "pattern": pattern,
                    })
                    break

        # 2. Check test method names
        if chunk.chunk_type.value in ('function', 'method'):
            method_key = (chunk.repo_name, file_path, chunk.name)
            if method_key not in seen_methods:
                chunk_name_lower = chunk.name.lower()

                # Check for explicit test method patterns
                for pattern in test_method_patterns:
                    if re.search(pattern, chunk_name_lower, re.IGNORECASE):
                        seen_methods.add(method_key)
                        test_methods.append({
                            "name": chunk.name,
                            "qualified_name": chunk.get_qualified_name(),
                            "file": file_path,
                            "line": chunk.start_line,
                            "repo": chunk.repo_name,
                            "match_reason": "method_name_pattern",
                        })
                        break

        # 3. Check content for symbol references
        content = chunk.content
        content_lower = content.lower()

        # Check if content references the symbol
        if symbol_name in content or symbol_lower in content_lower:
            # Add file if not already added
            if file_key not in seen_files:
                seen_files.add(file_key)
                test_files.append({
                    "file": file_path,
                    "repo": chunk.repo_name,
                    "match_reason": "content_reference",
                })

            # Check for test methods that reference the symbol
            if chunk.chunk_type.value in ('function', 'method'):
                method_key = (chunk.repo_name, file_path, chunk.name)
                chunk_name_lower = chunk.name.lower()

                # Is this a test method?
                is_test_method = (
                    chunk_name_lower.startswith('test') or
                    chunk_name_lower.startswith('it_') or
                    chunk_name_lower.startswith('should_') or
                    'test' in chunk.parent_name.lower() if chunk.parent_name else False
                )

                if is_test_method and method_key not in seen_methods:
                    seen_methods.add(method_key)
                    test_methods.append({
                        "name": chunk.name,
                        "qualified_name": chunk.get_qualified_name(),
                        "file": file_path,
                        "line": chunk.start_line,
                        "repo": chunk.repo_name,
                        "match_reason": "content_reference",
                    })

        # 4. Check imports
        if 'import' in content_lower and symbol_name in content:
            if file_key not in seen_files:
                seen_files.add(file_key)
                test_files.append({
                    "file": file_path,
                    "repo": chunk.repo_name,
                    "match_reason": "imports_symbol",
                })

    # Sort results
    test_files.sort(key=lambda x: (x['repo'], x['file']))
    test_methods.sort(key=lambda x: (x['repo'], x['file'], x['line']))

    result = {
        "symbol": symbol_name,
        "test_files_count": len(test_files),
        "test_methods_count": len(test_methods),
        "test_files": test_files,
        "test_methods": test_methods,
    }

    # Display results
    _display_tests(result)

    return result


def _to_snake_case(name: str) -> str:
    """Convert CamelCase or PascalCase to snake_case."""
    # Insert underscore before uppercase letters and lowercase the result
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _extract_words(name: str) -> list[str]:
    """Extract words from camelCase, PascalCase, or snake_case."""
    if '_' in name:
        return [w.lower() for w in name.split('_') if w]
    else:
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
        return [w.lower() for w in words if w]


def _display_tests(result: dict) -> None:
    """Display test discovery results."""
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    symbol = result["symbol"]
    console.print(f"\n[bold]Tests for '{symbol}'[/bold]\n")

    # Display test files
    test_files = result.get("test_files", [])
    if test_files:
        console.print(f"[bold green]Test Files ({len(test_files)}):[/bold green]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Repo", style="dim")
        table.add_column("Match Reason", style="yellow")

        for tf in test_files[:20]:
            table.add_row(
                tf['file'],
                tf.get('repo', ''),
                tf.get('match_reason', ''),
            )

        console.print(table)
        if len(test_files) > 20:
            console.print(f"[dim]... and {len(test_files) - 20} more test files[/dim]")
        console.print()

    # Display test methods
    test_methods = result.get("test_methods", [])
    if test_methods:
        console.print(f"[bold magenta]Test Methods ({len(test_methods)}):[/bold magenta]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("Method", style="cyan")
        table.add_column("File", style="dim")
        table.add_column("Line", style="green", justify="right")
        table.add_column("Match Reason", style="yellow")

        for tm in test_methods[:20]:
            table.add_row(
                tm['qualified_name'],
                tm['file'].split('/')[-1],
                str(tm['line']),
                tm.get('match_reason', ''),
            )

        console.print(table)
        if len(test_methods) > 20:
            console.print(f"[dim]... and {len(test_methods) - 20} more test methods[/dim]")
        console.print()

    if not test_files and not test_methods:
        console.print("[yellow]No tests found for this symbol.[/yellow]")
        console.print("[dim]Suggestions:[/dim]")
        console.print(f"[dim]  - Create a test file named test_{_to_snake_case(symbol)}.py[/dim]")
        console.print(f"[dim]  - Add test methods like test_{_to_snake_case(symbol)}_*[/dim]")
