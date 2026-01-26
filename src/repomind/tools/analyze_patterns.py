"""
Analyze Patterns Tool - Pattern and Convention Analysis for AI Assistants.

This tool helps AI assistants understand and match team conventions by analyzing:
- Library and framework usage patterns
- Code patterns (DI, error handling, async, etc.)
- Testing conventions
- Exemplary "golden" files

Example:
    # Get pattern recommendations
    result = analyze_patterns(category="dependency_injection")
    # Returns: {"recommended": "constructor_injection", "usage": "78%"}

    # Get full convention summary
    result = analyze_patterns(full_summary=True)

Author: RepoMind Team
"""

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..services.pattern_analyzer import (
    PatternAnalyzer,
    PatternCategory,
)

console = Console(width=200, force_terminal=False)


def analyze_patterns(
    category: Optional[str] = None,
    repo_filter: Optional[str] = None,
    full_summary: bool = False,
    include_golden_files: bool = False,
    include_testing: bool = False,
    top_libraries: int = 5,
) -> dict:
    """
    Analyze code patterns and conventions in the codebase.

    This tool helps AI understand what patterns and libraries the team uses
    so it can generate code that matches the existing style.

    Args:
        category: Pattern category to analyze. Options:
            - dependency_injection: DI patterns (constructor, field, setter)
            - error_handling: Error handling patterns
            - async_patterns: Async/await, promises, callbacks
            - http_client: HTTP client libraries used
            - testing: Testing frameworks and patterns
        repo_filter: Filter to a specific repository
        full_summary: Get comprehensive summary of all conventions
        include_golden_files: Include exemplary files in results
        include_testing: Include testing convention analysis
        top_libraries: Number of top libraries to show per category

    Returns:
        Dictionary with pattern analysis results:
        - If category specified: Detailed analysis of that category
        - If full_summary: Complete convention summary
        - Always includes: recommended patterns, usage percentages

    Example:
        >>> analyze_patterns(category="http_client")
        {
            "category": "http_client",
            "libraries": {
                "axios": {"imports": 45, "calls": 120},
                "fetch": {"imports": 12, "calls": 30}
            },
            "recommendation": "Use axios for HTTP requests (78% of codebase)"
        }
    """
    analyzer = PatternAnalyzer()

    # Full summary mode
    if full_summary:
        summary = analyzer.get_convention_summary(repo_filter)
        _display_convention_summary(summary)
        return summary

    # Category-specific analysis
    if category:
        try:
            pattern_category = PatternCategory(category.lower())
        except ValueError:
            valid_categories = [c.value for c in PatternCategory]
            return {
                "error": f"Invalid category: {category}",
                "valid_categories": valid_categories,
            }

        # Get pattern recommendations
        recommendations = analyzer.get_pattern_recommendations(pattern_category, repo_filter)

        # For HTTP client, also get library usage
        if pattern_category == PatternCategory.HTTP_CLIENT:
            libs = analyzer.analyze_library_usage(repo_filter, "http_client")
            recommendations["libraries"] = {
                name: {
                    "imports": lib.import_count,
                    "calls": lib.call_count,
                    "files": lib.file_count,
                }
                for name, lib in sorted(
                    libs.items(),
                    key=lambda x: x[1].import_count + x[1].call_count,
                    reverse=True
                )[:top_libraries]
            }

        # For testing, include conventions
        if pattern_category == PatternCategory.TESTING:
            testing = analyzer.detect_testing_conventions(repo_filter)
            recommendations["frameworks"] = [
                {
                    "name": t.framework,
                    "file_patterns": t.file_patterns,
                    "assertions": t.common_assertions,
                    "mock_library": t.mock_library,
                    "usage_count": t.usage_count,
                }
                for t in testing
            ]

        _display_pattern_recommendation(recommendations)
        return recommendations

    # Default: library usage summary
    result = {
        "library_usage": {},
        "recommendations": [],
    }

    # Get library usage by category
    for lib_category in ["http_client", "database", "testing", "logging", "framework"]:
        libs = analyzer.analyze_library_usage(repo_filter, lib_category)
        if libs:
            top_lib = max(libs.values(), key=lambda l: l.import_count + l.call_count)
            result["library_usage"][lib_category] = {
                name: {
                    "imports": lib.import_count,
                    "calls": lib.call_count,
                }
                for name, lib in sorted(
                    libs.items(),
                    key=lambda x: x[1].import_count + x[1].call_count,
                    reverse=True
                )[:top_libraries]
            }
            result["recommendations"].append(
                f"Use {top_lib.name} for {lib_category} ({top_lib.import_count} imports)"
            )

    # Add testing conventions if requested
    if include_testing:
        testing = analyzer.detect_testing_conventions(repo_filter)
        result["testing_conventions"] = [
            {
                "framework": t.framework,
                "file_patterns": t.file_patterns,
                "usage_count": t.usage_count,
            }
            for t in testing[:3]
        ]

    # Add golden files if requested
    if include_golden_files:
        golden = analyzer.find_golden_files(repo_filter, top_n=5)
        result["golden_files"] = [
            {
                "file": g.file_path,
                "score": g.score,
                "reasons": g.reasons,
            }
            for g in golden
        ]

    _display_library_summary(result)
    return result


def get_coding_conventions(
    repo_filter: Optional[str] = None,
) -> dict:
    """
    Get coding conventions to follow when generating code.

    This is a shortcut for analyze_patterns(full_summary=True) that
    formats the output specifically for AI code generation guidance.

    Args:
        repo_filter: Filter to a specific repository

    Returns:
        Dictionary with coding conventions to follow:
        - preferred_libraries: Which libraries to use for each purpose
        - patterns: Which patterns to prefer (e.g., constructor injection)
        - testing: How to write tests
        - examples: Example files to reference

    Example:
        >>> get_coding_conventions()
        {
            "preferred_libraries": {
                "http": "axios",
                "testing": "jest",
                "validation": "zod"
            },
            "patterns": {
                "dependency_injection": "constructor_injection",
                "error_handling": "try_catch with custom exceptions"
            },
            "testing": {
                "framework": "jest",
                "file_pattern": "*.spec.ts",
                "example": "src/services/__tests__/user.spec.ts"
            },
            "reference_files": [
                "src/services/auth.service.ts (score: 85)"
            ]
        }
    """
    analyzer = PatternAnalyzer()
    summary = analyzer.get_convention_summary(repo_filter)

    # Format for code generation guidance
    conventions = {
        "preferred_libraries": {},
        "patterns": {},
        "testing": {},
        "reference_files": [],
    }

    # Extract preferred libraries
    for category, libs in summary.get("library_usage", {}).items():
        if libs:
            top_lib = list(libs.keys())[0]
            conventions["preferred_libraries"][category] = top_lib

    # Extract patterns
    for pattern_name, pattern_info in summary.get("patterns", {}).items():
        conventions["patterns"][pattern_name] = pattern_info.get("recommended", "unknown")

    # Extract testing conventions
    testing = summary.get("testing", [])
    if testing:
        t = testing[0]
        conventions["testing"] = {
            "framework": t.get("framework"),
            "file_patterns": t.get("file_patterns", []),
            "usage_count": t.get("usage_count", 0),
        }

    # Extract reference files
    for g in summary.get("golden_files", []):
        conventions["reference_files"].append(
            f"{g['file']} (score: {g['score']})"
        )

    console.print(Panel(
        "[bold]Coding Conventions Summary[/bold]\n\n"
        f"[cyan]Libraries:[/cyan] {conventions['preferred_libraries']}\n"
        f"[yellow]Patterns:[/yellow] {conventions['patterns']}\n"
        f"[green]Testing:[/green] {conventions['testing'].get('framework', 'unknown')}"
    ))

    return conventions


# =============================================================================
# Display Functions
# =============================================================================


def _display_convention_summary(summary: dict) -> None:
    """Display convention summary in console."""
    console.print(Panel("[bold]Codebase Convention Summary[/bold]"))

    # Library usage
    if summary.get("library_usage"):
        console.print("\n[cyan]Library Usage by Category:[/cyan]")
        for category, libs in summary["library_usage"].items():
            lib_str = ", ".join(f"{name}({info['imports']})" for name, info in list(libs.items())[:3])
            console.print(f"  {category}: {lib_str}")

    # Patterns
    if summary.get("patterns"):
        console.print("\n[yellow]Recommended Patterns:[/yellow]")
        for pattern, info in summary["patterns"].items():
            console.print(f"  {pattern}: {info['recommended']} ({info['usage']})")

    # Testing
    if summary.get("testing"):
        console.print("\n[green]Testing Conventions:[/green]")
        for t in summary["testing"]:
            console.print(f"  {t['framework']}: {t['usage_count']} files")

    # Golden files
    if summary.get("golden_files"):
        console.print("\n[magenta]Reference Files:[/magenta]")
        for g in summary["golden_files"]:
            console.print(f"  {g['file']} (score: {g['score']})")


def _display_pattern_recommendation(rec: dict) -> None:
    """Display pattern recommendation in console."""
    console.print(Panel(f"[bold]Pattern Analysis: {rec.get('category', 'unknown')}[/bold]"))

    if rec.get("recommended"):
        console.print(f"[green]Recommended:[/green] {rec['recommended']}")
        console.print(f"[cyan]Usage:[/cyan] {rec.get('usage_percentage', 0)}%")
        console.print(f"[yellow]Momentum:[/yellow] {rec.get('momentum', 'stable')}")

        if rec.get("example_locations"):
            console.print("\n[dim]Examples:[/dim]")
            for loc in rec["example_locations"]:
                console.print(f"  - {loc}")

    if rec.get("libraries"):
        console.print("\n[cyan]Libraries:[/cyan]")
        for name, info in rec["libraries"].items():
            console.print(f"  {name}: {info['imports']} imports, {info['calls']} calls")


def _display_library_summary(result: dict) -> None:
    """Display library summary in console."""
    table = Table(title="Library Usage Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Top Libraries", style="green")

    for category, libs in result.get("library_usage", {}).items():
        lib_str = ", ".join(f"{name}({info['imports']})" for name, info in list(libs.items())[:3])
        table.add_row(category, lib_str)

    console.print(table)

    if result.get("recommendations"):
        console.print("\n[yellow]Recommendations:[/yellow]")
        for rec in result["recommendations"]:
            console.print(f"  - {rec}")
