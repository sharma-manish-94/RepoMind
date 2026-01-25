"""Diff impact tool for analyzing the impact of recent git changes.

Analyzes git diffs to find modified symbols and their callers, helping
understand the blast radius of code changes.
"""

import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..services.call_graph import CallGraphService
from ..services.symbol_table import SymbolTableService
from .find_tests import find_tests

console = Console(width=200, force_terminal=False)


def diff_impact(
    repo_path: str,
    since: str = "HEAD~1",
    include_tests: bool = True,
) -> dict:
    """Analyze impact of recent git changes.

    This tool analyzes the git diff to find:
    1. What files have changed
    2. What symbols (functions, classes, methods) were modified
    3. What other code calls those symbols (impact analysis)
    4. What tests might be affected

    Args:
        repo_path: Path to the repository to analyze
        since: Git reference to compare against (default: HEAD~1 for last commit)
        include_tests: If True, also find affected tests

    Returns:
        Dictionary containing:
        - changed_files: List of files that changed
        - modified_symbols: Symbols that were directly modified
        - impacted_symbols: Symbols that call the modified ones
        - affected_tests: Tests that should be run

    Example:
        >>> diff_impact("/path/to/repo", since="HEAD~3")
        {
            "changed_files": ["src/auth.py", "src/utils.py"],
            "modified_symbols": [{"name": "validate", "type": "function"}],
            "impacted_symbols": [{"name": "login", "calls": "validate"}],
            "risk_level": "medium"
        }
    """
    path = Path(repo_path).resolve()

    if not path.exists():
        return {"error": f"Repository path does not exist: {repo_path}"}

    if not (path / ".git").exists():
        return {"error": f"Not a git repository: {repo_path}"}

    console.print(f"[bold blue]Analyzing impact of changes since {since}[/bold blue]")

    # Step 1: Get changed files from git
    changed_files = _get_changed_files(path, since)

    if not changed_files:
        return {
            "repo_path": str(path),
            "since": since,
            "status": "no_changes",
            "message": f"No changes found since {since}",
        }

    console.print(f"[dim]Found {len(changed_files)} changed files[/dim]")

    # Determine repo name from path
    repo_name = path.name

    # Step 2: Find modified symbols
    symbol_service = SymbolTableService()
    call_graph = CallGraphService()

    modified_symbols = []
    for file_path in changed_files:
        symbols = symbol_service.lookup_by_file(repo_name, file_path)
        for symbol in symbols:
            modified_symbols.append({
                "name": symbol.name,
                "qualified_name": symbol.qualified_name,
                "type": symbol.symbol_type,
                "file": symbol.file_path,
                "line": symbol.start_line,
            })

    console.print(f"[dim]Found {len(modified_symbols)} modified symbols[/dim]")

    # Step 3: Find impacted symbols (callers of modified symbols)
    impacted_symbols = []
    seen_callers = set()

    for symbol in modified_symbols:
        callers = call_graph.find_callers(symbol["qualified_name"], repo_name)
        for caller in callers:
            caller_key = (caller.caller, caller.caller_file, caller.caller_line)
            if caller_key not in seen_callers:
                seen_callers.add(caller_key)
                impacted_symbols.append({
                    "name": caller.caller,
                    "file": caller.caller_file,
                    "line": caller.caller_line,
                    "calls": symbol["qualified_name"],
                    "call_type": caller.call_type,
                })

    console.print(f"[dim]Found {len(impacted_symbols)} impacted callers[/dim]")

    # Step 4: Find affected tests
    affected_tests = []
    if include_tests:
        tested_symbols = set()
        for symbol in modified_symbols:
            if symbol["qualified_name"] not in tested_symbols:
                tested_symbols.add(symbol["qualified_name"])
                test_result = find_tests(symbol["name"], repo_name)
                if test_result.get("test_methods"):
                    for test in test_result["test_methods"][:5]:  # Limit to 5 per symbol
                        affected_tests.append({
                            "test": test["qualified_name"],
                            "file": test["file"],
                            "line": test["line"],
                            "tests_symbol": symbol["name"],
                        })

    # Step 5: Calculate risk level
    risk_level = _calculate_risk_level(
        len(changed_files),
        len(modified_symbols),
        len(impacted_symbols),
    )

    result = {
        "repo_path": str(path),
        "repo_name": repo_name,
        "since": since,
        "status": "analyzed",
        "risk_level": risk_level,
        "summary": {
            "changed_files": len(changed_files),
            "modified_symbols": len(modified_symbols),
            "impacted_callers": len(impacted_symbols),
            "affected_tests": len(affected_tests),
        },
        "changed_files": changed_files,
        "modified_symbols": modified_symbols[:50],  # Limit output
        "impacted_callers": impacted_symbols[:50],
        "affected_tests": affected_tests[:20] if include_tests else None,
    }

    # Display results
    _display_impact(result)

    return result


def _get_changed_files(repo_path: Path, since: str) -> list[str]:
    """Get list of changed files from git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", since],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

        # Also get staged changes
        staged_result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if staged_result.returncode == 0:
            staged_files = [f.strip() for f in staged_result.stdout.strip().split("\n") if f.strip()]
            files = list(set(files + staged_files))

        # Filter to only source files
        source_extensions = {".py", ".java", ".ts", ".tsx", ".js", ".jsx"}
        files = [f for f in files if Path(f).suffix in source_extensions]

        return sorted(files)
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        # git not available
        return []


def _calculate_risk_level(
    changed_files: int,
    modified_symbols: int,
    impacted_callers: int,
) -> str:
    """Calculate risk level based on change metrics."""
    # Simple heuristic for risk assessment
    score = 0

    # File count factor
    if changed_files > 10:
        score += 3
    elif changed_files > 5:
        score += 2
    elif changed_files > 2:
        score += 1

    # Symbol count factor
    if modified_symbols > 20:
        score += 3
    elif modified_symbols > 10:
        score += 2
    elif modified_symbols > 5:
        score += 1

    # Impact factor (callers affected)
    if impacted_callers > 50:
        score += 4
    elif impacted_callers > 20:
        score += 3
    elif impacted_callers > 10:
        score += 2
    elif impacted_callers > 5:
        score += 1

    # Determine risk level
    if score >= 8:
        return "high"
    elif score >= 4:
        return "medium"
    else:
        return "low"


def _display_impact(result: dict) -> None:
    """Display impact analysis results."""
    if result.get("status") == "no_changes":
        console.print(f"[green]{result.get('message')}[/green]")
        return

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    console.print(f"\n[bold]Impact Analysis: {result['repo_name']}[/bold]")
    console.print(f"[dim]Changes since: {result['since']}[/dim]\n")

    # Risk level with color
    risk = result["risk_level"]
    risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(risk, "white")
    console.print(f"[bold {risk_color}]Risk Level: {risk.upper()}[/bold {risk_color}]\n")

    # Summary
    summary = result["summary"]
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Changed files: {summary['changed_files']}")
    console.print(f"  Modified symbols: {summary['modified_symbols']}")
    console.print(f"  Impacted callers: {summary['impacted_callers']}")
    if summary.get("affected_tests"):
        console.print(f"  Affected tests: {summary['affected_tests']}")
    console.print()

    # Changed files
    changed_files = result.get("changed_files", [])
    if changed_files:
        console.print("[bold cyan]Changed Files:[/bold cyan]")
        for f in changed_files[:10]:
            console.print(f"  {f}")
        if len(changed_files) > 10:
            console.print(f"  [dim]... and {len(changed_files) - 10} more[/dim]")
        console.print()

    # Modified symbols
    modified = result.get("modified_symbols", [])
    if modified:
        console.print("[bold yellow]Modified Symbols:[/bold yellow]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("File", style="dim")

        for sym in modified[:15]:
            table.add_row(
                sym["qualified_name"],
                sym["type"],
                f"{sym['file']}:{sym['line']}",
            )

        console.print(table)
        if len(modified) > 15:
            console.print(f"[dim]... and {len(modified) - 15} more modified symbols[/dim]")
        console.print()

    # Impacted callers
    impacted = result.get("impacted_callers", [])
    if impacted:
        console.print("[bold magenta]Impacted Callers (may need review):[/bold magenta]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("Caller", style="cyan")
        table.add_column("Calls", style="yellow")
        table.add_column("File", style="dim")

        for caller in impacted[:15]:
            table.add_row(
                caller["name"],
                caller["calls"],
                f"{caller['file']}:{caller['line']}",
            )

        console.print(table)
        if len(impacted) > 15:
            console.print(f"[dim]... and {len(impacted) - 15} more impacted callers[/dim]")
        console.print()

    # Affected tests
    tests = result.get("affected_tests", [])
    if tests:
        console.print("[bold green]Suggested Tests to Run:[/bold green]")
        for test in tests[:10]:
            console.print(f"  {test['test']} [dim]({test['file']}:{test['line']})[/dim]")
        if len(tests) > 10:
            console.print(f"  [dim]... and {len(tests) - 10} more tests[/dim]")
