"""Architecture analysis tools for code quality and maintenance.

Provides:
- Dead code detection (unreachable code)
- Circular dependency detection
- Entry point identification
- Cross-language API linking analysis
"""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..services.call_graph import CallGraphService
from ..services.storage import StorageService
from ..services.route_linker import RouteLinker
from ..services.route_registry import RouteRegistry
from ..parsers.openapi_parser import OpenAPIParser
from ..config import get_config

console = Console(width=200, force_terminal=False)


def find_dead_code(
    repo_name: Optional[str] = None,
    include_entry_points: bool = True,
    limit: int = 100,
) -> dict:
    """Find potentially dead (unreachable) code in the codebase.

    Dead code is code that cannot be reached from any entry point.
    Entry points include:
    - API route handlers
    - Main functions
    - React page components
    - Test functions

    Args:
        repo_name: Filter to a specific repository
        include_entry_points: Show identified entry points in output
        limit: Maximum number of dead code items to return

    Returns:
        Dictionary containing:
        - dead_code: List of unreachable symbols
        - entry_points: List of identified entry points (if include_entry_points)
        - statistics: Summary statistics

    Example:
        >>> find_dead_code("my-repo")
        {
            "dead_code": [
                {"symbol": "old_handler", "file": "src/handlers.py"},
                {"symbol": "unused_util", "file": "src/utils.py"}
            ],
            "entry_points": [...],
            "statistics": {"total_symbols": 100, "dead_count": 5}
        }
    """
    console.print("[bold blue]Analyzing code for dead code...[/bold blue]")

    call_graph = CallGraphService()
    storage = StorageService()

    # Load all chunks to identify entry points
    chunk_map = storage._load_chunk_metadata()
    chunks = list(chunk_map.values())

    if repo_name:
        chunks = [c for c in chunks if c.repo_name == repo_name]

    # Identify entry points
    entry_points = call_graph.identify_entry_points(chunks, repo_name)
    console.print(f"[green]Identified {len(entry_points)} entry points[/green]")

    # Find unreachable code
    unreachable = call_graph.find_unreachable_from_entry_points(entry_points, repo_name)

    # Get file information for unreachable symbols
    symbol_to_file = {}
    for chunk in chunks:
        qualified_name = chunk.get_qualified_name()
        symbol_to_file[qualified_name] = chunk.file_path

    # Filter and format dead code
    dead_code = []
    for symbol in unreachable[:limit]:
        file_path = symbol_to_file.get(symbol, "unknown")
        dead_code.append({
            "symbol": symbol,
            "file": file_path,
        })

    # Build result
    result = {
        "dead_code": dead_code,
        "statistics": {
            "total_symbols": len(call_graph.get_all_symbols(repo_name)),
            "entry_point_count": len(entry_points),
            "dead_count": len(unreachable),
            "showing": min(len(unreachable), limit),
        },
    }

    if include_entry_points:
        result["entry_points"] = [
            {"symbol": ep, "file": symbol_to_file.get(ep, "unknown")}
            for ep in entry_points[:50]
        ]

    # Display results
    if dead_code:
        table = Table(title="Potentially Dead Code")
        table.add_column("Symbol", style="red")
        table.add_column("File", style="dim")

        for item in dead_code[:20]:
            table.add_row(item["symbol"], item["file"])

        console.print(table)
    else:
        console.print("[green]No dead code detected![/green]")

    return result


def detect_circular_dependencies(
    repo_name: Optional[str] = None,
    file_level: bool = True,
    limit: int = 50,
) -> dict:
    """Detect circular dependencies in the codebase.

    Finds cycles in the call graph where:
    - File A depends on File B, which depends on File A (file level)
    - Function A calls Function B, which calls Function A (function level)

    Args:
        repo_name: Filter to a specific repository
        file_level: If True, detect file-level cycles; otherwise function-level
        limit: Maximum number of cycles to return

    Returns:
        Dictionary containing:
        - cycles: List of detected cycles
        - statistics: Summary statistics

    Example:
        >>> detect_circular_dependencies("my-repo")
        {
            "cycles": [
                ["src/a.py", "src/b.py", "src/c.py"],
                ["src/service.py", "src/handler.py"]
            ],
            "statistics": {"total_cycles": 2}
        }
    """
    console.print("[bold blue]Detecting circular dependencies...[/bold blue]")

    call_graph = CallGraphService()

    # Detect cycles
    cycles = call_graph.detect_cycles(repo_name, group_by_file=file_level)

    # Limit output
    cycles = cycles[:limit]

    # Build result
    result = {
        "cycles": cycles,
        "statistics": {
            "total_cycles": len(cycles),
            "level": "file" if file_level else "function",
        },
    }

    # Display results
    if cycles:
        table = Table(title=f"Circular Dependencies ({'File' if file_level else 'Function'} Level)")
        table.add_column("#", style="dim")
        table.add_column("Cycle", style="red")
        table.add_column("Size", style="cyan")

        for i, cycle in enumerate(cycles[:20], 1):
            cycle_str = " → ".join(cycle[:5])
            if len(cycle) > 5:
                cycle_str += f" → ... ({len(cycle)} total)"
            table.add_row(str(i), cycle_str, str(len(cycle)))

        console.print(table)
        console.print(f"\n[yellow]Found {len(cycles)} circular dependencies[/yellow]")
    else:
        console.print("[green]No circular dependencies detected![/green]")

    return result


def analyze_architecture(
    repo_name: Optional[str] = None,
) -> dict:
    """Run a complete architecture analysis on the codebase.

    Combines multiple analyses:
    - Dead code detection
    - Circular dependency detection (both file and function level)
    - Entry point analysis

    Args:
        repo_name: Filter to a specific repository

    Returns:
        Dictionary with results from all analyses

    Example:
        >>> analyze_architecture("my-repo")
        {
            "dead_code": {...},
            "circular_dependencies_file": {...},
            "circular_dependencies_function": {...},
            "summary": {
                "health_score": 85,
                "issues": [...]
            }
        }
    """
    console.print("[bold blue]Running complete architecture analysis...[/bold blue]\n")

    # Run analyses
    dead_code_result = find_dead_code(repo_name, include_entry_points=False)
    file_cycles = detect_circular_dependencies(repo_name, file_level=True)
    function_cycles = detect_circular_dependencies(repo_name, file_level=False)

    # Calculate health score (simple heuristic)
    issues = []
    health_score = 100

    dead_count = dead_code_result["statistics"]["dead_count"]
    if dead_count > 0:
        health_score -= min(20, dead_count * 2)
        issues.append(f"{dead_count} potentially unreachable symbols")

    file_cycle_count = file_cycles["statistics"]["total_cycles"]
    if file_cycle_count > 0:
        health_score -= min(30, file_cycle_count * 5)
        issues.append(f"{file_cycle_count} file-level circular dependencies")

    func_cycle_count = function_cycles["statistics"]["total_cycles"]
    if func_cycle_count > 0:
        health_score -= min(20, func_cycle_count)
        issues.append(f"{func_cycle_count} function-level circular dependencies")

    health_score = max(0, health_score)

    # Build result
    result = {
        "dead_code": dead_code_result,
        "circular_dependencies_file": file_cycles,
        "circular_dependencies_function": function_cycles,
        "summary": {
            "health_score": health_score,
            "issues": issues,
            "recommendation": (
                "Good architecture health"
                if health_score >= 80
                else "Consider refactoring to reduce dependencies"
                if health_score >= 50
                else "Architecture needs significant refactoring"
            ),
        },
    }

    # Print summary
    console.print("\n" + "=" * 60)
    console.print(f"[bold]Architecture Health Score: {health_score}/100[/bold]")

    if issues:
        console.print("\n[yellow]Issues found:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")

    console.print(f"\n[dim]{result['summary']['recommendation']}[/dim]")

    return result


def analyze_api_links(
    repo_name: Optional[str] = None,
    show_ambiguous: bool = True,
    min_confidence: float = 0.65,
    use_openapi: bool = True,
    persist: bool = True,
) -> dict:
    """Comprehensive API linking analysis across languages and frameworks.

    Analyzes the codebase to:
    1. Find all backend API endpoints (Python, Java, TypeScript)
    2. Find all frontend API calls
    3. Match calls to endpoints using multi-factor scoring
    4. Detect ambiguous matches requiring manual review
    5. Identify unlinked calls and unused endpoints
    6. Optionally validate against OpenAPI specs

    Args:
        repo_name: Filter to a specific repository
        show_ambiguous: Include ambiguous matches in output
        min_confidence: Minimum confidence threshold for matches
        use_openapi: Use OpenAPI specs as additional endpoint source
        persist: Persist results to SQLite registry

    Returns:
        Dictionary containing:
        - linked: Successfully matched call→endpoint pairs
        - ambiguous: Matches requiring manual review
        - unlinked_calls: API calls with no matching endpoint
        - unlinked_endpoints: Endpoints never called (potential dead code)
        - conflicts: Duplicate route definitions
        - statistics: Summary statistics

    Example:
        >>> analyze_api_links("my-repo", show_ambiguous=True)
        {
            "linked": [...],
            "ambiguous": [...],
            "statistics": {
                "endpoints_found": 45,
                "api_calls_found": 38,
                "links_matched": 35,
                "links_ambiguous": 3,
                ...
            }
        }
    """
    console.print("[bold blue]Analyzing cross-language API links...[/bold blue]\n")

    config = get_config()
    linker = RouteLinker()

    # Update config threshold if different
    if min_confidence != config.route_linking.min_confidence:
        config.route_linking.min_confidence = min_confidence

    # Run linking analysis
    result = linker.link_all_routes(repo_name)

    # Optionally add OpenAPI endpoints
    openapi_endpoints = []
    if use_openapi and config.repos_dir:
        console.print("[dim]Scanning for OpenAPI specifications...[/dim]")
        openapi_parser = OpenAPIParser()

        if repo_name:
            repo_path = config.repos_dir / repo_name
            if repo_path.exists():
                spec_files = openapi_parser.detect_spec_files(repo_path)
                for spec_path in spec_files:
                    endpoints = openapi_parser.parse_spec(spec_path, repo_name)
                    openapi_endpoints.extend(endpoints)
                    console.print(f"  Found {len(endpoints)} endpoints in {spec_path.name}")

        if openapi_endpoints:
            # Re-run matching with OpenAPI endpoints included
            all_endpoints = linker.collect_endpoints(repo_name) + openapi_endpoints
            api_calls = linker.collect_api_calls(repo_name)
            additional_matches = linker.match_routes(api_calls, openapi_endpoints)

            # Add non-duplicate matches
            existing_call_ids = {m.api_call.id for m in result.linked + result.ambiguous}
            for match in additional_matches:
                if match.api_call.id not in existing_call_ids:
                    if match.is_ambiguous:
                        result.ambiguous.append(match)
                    else:
                        result.linked.append(match)

            result.statistics["openapi_endpoints"] = len(openapi_endpoints)

    # Persist to registry if requested
    if persist:
        console.print("[dim]Persisting results to route registry...[/dim]")
        registry = RouteRegistry()

        endpoints = linker.collect_endpoints(repo_name)
        api_calls = linker.collect_api_calls(repo_name)
        all_matches = result.linked + result.ambiguous

        registry.register_bulk(
            endpoints=endpoints + openapi_endpoints,
            calls=api_calls,
            links=all_matches,
        )

    # Display results
    _display_api_link_results(result, show_ambiguous)

    return result.to_dict()


def _display_api_link_results(result, show_ambiguous: bool) -> None:
    """Display API linking results in formatted tables."""

    # Linked routes table
    if result.linked:
        table = Table(title="Linked API Routes")
        table.add_column("Frontend Call", style="cyan")
        table.add_column("Backend Handler", style="green")
        table.add_column("Method", style="yellow")
        table.add_column("Confidence", style="magenta")

        for match in result.linked[:20]:
            table.add_row(
                f"{match.api_call.caller_qualified_name}",
                f"{match.endpoint.qualified_name}",
                match.api_call.method,
                f"{match.confidence:.2f}",
            )

        if len(result.linked) > 20:
            table.add_row("...", f"({len(result.linked) - 20} more)", "", "")

        console.print(table)
        console.print()

    # Ambiguous matches table
    if show_ambiguous and result.ambiguous:
        table = Table(title="⚠ Ambiguous Matches (Require Review)")
        table.add_column("API Call", style="yellow")
        table.add_column("Possible Handlers", style="dim")
        table.add_column("Confidence", style="magenta")

        # Group by call
        call_matches = {}
        for match in result.ambiguous:
            call_id = match.api_call.id
            if call_id not in call_matches:
                call_matches[call_id] = {"call": match.api_call, "matches": []}
            call_matches[call_id]["matches"].append(match)

        for call_data in list(call_matches.values())[:10]:
            call = call_data["call"]
            matches = call_data["matches"]
            handlers = ", ".join(m.endpoint.qualified_name for m in matches[:3])
            if len(matches) > 3:
                handlers += f" (+{len(matches) - 3} more)"

            table.add_row(
                f"{call.url}",
                handlers,
                f"{matches[0].confidence:.2f}",
            )

        console.print(table)
        console.print()

    # Unlinked calls
    if result.unlinked_calls:
        table = Table(title="✗ Unlinked API Calls (No Matching Endpoint)")
        table.add_column("URL", style="red")
        table.add_column("Method", style="yellow")
        table.add_column("Caller", style="dim")
        table.add_column("File", style="dim")

        for call in result.unlinked_calls[:10]:
            table.add_row(
                call.url[:50] + ("..." if len(call.url) > 50 else ""),
                call.method,
                call.caller_qualified_name,
                call.file_path.split("/")[-1],
            )

        if len(result.unlinked_calls) > 10:
            table.add_row("...", "", f"({len(result.unlinked_calls) - 10} more)", "")

        console.print(table)
        console.print()

    # Unlinked endpoints (potential dead code)
    if result.unlinked_endpoints:
        table = Table(title="Unlinked Endpoints (Never Called)")
        table.add_column("Path", style="dim")
        table.add_column("Method", style="yellow")
        table.add_column("Handler", style="cyan")
        table.add_column("Framework", style="dim")

        for endpoint in result.unlinked_endpoints[:10]:
            table.add_row(
                endpoint.path,
                endpoint.method,
                endpoint.qualified_name,
                endpoint.framework or "",
            )

        if len(result.unlinked_endpoints) > 10:
            table.add_row("...", "", f"({len(result.unlinked_endpoints) - 10} more)", "")

        console.print(table)
        console.print()

    # Conflicts
    if result.conflicts:
        table = Table(title="⚠ Route Conflicts")
        table.add_column("Route", style="red")
        table.add_column("Conflicting Handlers", style="yellow")

        for conflict in result.conflicts[:5]:
            handlers = ", ".join(e.qualified_name for e in conflict.endpoints)
            table.add_row(
                f"{conflict.endpoints[0].method} {conflict.endpoints[0].path}",
                handlers,
            )

        console.print(table)
        console.print()

    # Summary
    stats = result.statistics
    console.print("\n" + "=" * 60)
    console.print("[bold]API Linking Summary[/bold]")
    console.print(f"  Endpoints found: {stats.get('endpoints_found', 0)}")
    console.print(f"  API calls found: {stats.get('api_calls_found', 0)}")
    console.print(f"  [green]✓ Linked: {stats.get('links_confident', 0)}[/green]")

    if stats.get('links_ambiguous', 0) > 0:
        console.print(f"  [yellow]⚠ Ambiguous: {stats.get('links_ambiguous', 0)}[/yellow]")

    if stats.get('unlinked_calls', 0) > 0:
        console.print(f"  [red]✗ Unlinked calls: {stats.get('unlinked_calls', 0)}[/red]")

    if stats.get('unlinked_endpoints', 0) > 0:
        console.print(f"  [dim]Unused endpoints: {stats.get('unlinked_endpoints', 0)}[/dim]")

    if stats.get('conflicts_detected', 0) > 0:
        console.print(f"  [red]⚠ Conflicts: {stats.get('conflicts_detected', 0)}[/red]")

    # Calculate link rate
    total_calls = stats.get('api_calls_found', 0)
    linked_calls = stats.get('links_confident', 0) + stats.get('links_ambiguous', 0)
    link_rate = (linked_calls / total_calls * 100) if total_calls > 0 else 0
    console.print(f"\n  Link rate: {link_rate:.1f}%")


def validate_routes_against_spec(
    repo_name: str,
    spec_path: Optional[str] = None,
) -> dict:
    """Validate code-extracted routes against an OpenAPI specification.

    Compares routes extracted from code with those defined in an OpenAPI spec
    to find discrepancies.

    Args:
        repo_name: Repository name
        spec_path: Optional path to OpenAPI spec (auto-detected if not provided)

    Returns:
        Validation report with matches and mismatches
    """
    console.print("[bold blue]Validating routes against OpenAPI spec...[/bold blue]\n")

    config = get_config()
    linker = RouteLinker()
    openapi_parser = OpenAPIParser()

    # Get code-extracted endpoints
    code_endpoints = linker.collect_endpoints(repo_name)

    # Find spec file
    if spec_path:
        spec_file = Path(spec_path)
    else:
        if not config.repos_dir:
            return {"error": "repos_dir not configured and no spec_path provided"}

        repo_path = config.repos_dir / repo_name
        spec_files = openapi_parser.detect_spec_files(repo_path)

        if not spec_files:
            return {"error": f"No OpenAPI spec found in {repo_path}"}

        spec_file = spec_files[0]
        console.print(f"Using spec: {spec_file}")

    # Validate
    result = openapi_parser.validate_against_spec(code_endpoints, spec_file)

    # Display results
    console.print("\n[bold]Validation Results[/bold]")
    console.print(f"  Code endpoints: {result['summary']['total_code_endpoints']}")
    console.print(f"  Spec endpoints: {result['summary']['total_spec_endpoints']}")
    console.print(f"  [green]Matching: {result['summary']['matching_count']}[/green]")
    console.print(f"  Coverage: {result['summary']['coverage_percent']}%")

    if result['missing_from_spec']:
        console.print(f"\n[yellow]Routes in code but not in spec: {len(result['missing_from_spec'])}[/yellow]")
        for item in result['missing_from_spec'][:5]:
            console.print(f"    {item['method']} {item['path']}")

    if result['missing_from_code']:
        console.print(f"\n[red]Routes in spec but not in code: {len(result['missing_from_code'])}[/red]")
        for item in result['missing_from_code'][:5]:
            console.print(f"    {item['method']} {item['path']}")

    return result
