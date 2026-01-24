"""
Command Line Interface for RepoMind.

This module provides the CLI for managing code indexing and search
operations. It supports both individual repository indexing and
bulk operations across multiple repositories.

Commands:
    - index: Index a single repository
    - index-all: Discover and index all repositories
    - discover: List repositories without indexing
    - search: Semantic code search
    - context: Get code context for a symbol
    - stats: Show index statistics
    - clear: Clear index for a repository
    - clear-all: Clear entire index
    - serve: Run as MCP server
    - config-show: Show current configuration

Example Usage:
    $ code-expert --repos-dir ~/projects index-all
    $ code-expert search "authentication middleware"
    $ code-expert context "UserService.validate"

Author: RepoMind Team
"""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import Config, get_config, set_config
from .services.storage import StorageService
from .tools.get_context import get_context
from .tools.index_repo import (
    discover_repositories,
    index_all_repositories,
    index_repo,
)
from .tools.semantic_grep import semantic_grep

console = Console()


@click.group()
@click.option(
    "--repos-dir",
    type=click.Path(exists=True),
    help="Root directory containing repositories to index"
)
@click.pass_context
def main(ctx, repos_dir):
    """RepoMind - AI-Powered Code Intelligence Platform.

    Semantic code search and analysis for any codebase.
    Supports Python, Java, TypeScript, and JavaScript.

    \b
    Quick Start:
        # Index all repos in a directory
        code-expert --repos-dir ~/projects index-all

        # Search your codebase
        code-expert search "authentication middleware"

        # Get context for a symbol
        code-expert context "UserService.validate"

    \b
    Features:
        - Auto-discovers repositories in a directory
        - Semantic search by meaning, not just keywords
        - Symbol lookup with full context
        - Call graph analysis
        - MCP integration for AI assistants
    """
    ctx.ensure_object(dict)

    if repos_dir:
        config = Config(repos_dir=Path(repos_dir))
        set_config(config)


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Custom name for the repository")
@click.option("--mock", is_flag=True, help="Use mock embeddings (no API calls)")
@click.option("--incremental", "-i", is_flag=True, help="Only index changed files since last index")
def index(repo_path, name, mock, incremental):
    """Index a single repository.

    REPO_PATH: Path to the repository to index

    Use --incremental to only re-index files that have changed since
    the last index. This uses git to detect changes when available.
    """
    result = index_repo(repo_path, repo_name=name, use_mock_embeddings=mock, incremental=incremental)
    console.print_json(json.dumps(result, default=str))


@main.command("index-all")
@click.option("--mock", is_flag=True, help="Use mock embeddings (no API calls)")
@click.option("--incremental", "-i", is_flag=True, help="Only index changed files since last index")
@click.option(
    "--pattern", "-p",
    multiple=True,
    help="Include only repos matching pattern (e.g., -p 'service-*' -p 'lib-*')"
)
@click.option(
    "--exclude", "-e",
    multiple=True,
    help="Exclude repos matching pattern (e.g., -e 'legacy-*' -e 'test-*')"
)
def index_all(mock, incremental, pattern, exclude):
    """Discover and index all repositories.

    Auto-discovers repositories in the configured directory and indexes
    each one. Use patterns to filter which repositories to include.

    \b
    Examples:
        # Index all repositories
        code-expert --repos-dir ~/projects index-all

        # Index only repos matching patterns
        code-expert --repos-dir ~/projects index-all -p "service-*" -p "lib-*"

        # Exclude certain repos
        code-expert --repos-dir ~/projects index-all -e "legacy-*" -e "deprecated-*"

        # Incremental update (only changed files)
        code-expert --repos-dir ~/projects index-all --incremental
    """
    config = get_config()

    if not config.repos_dir or not config.repos_dir.exists():
        console.print("[red]Error: repos_dir not configured or doesn't exist[/red]")
        console.print("Use --repos-dir option to specify the directory containing your repositories")
        console.print("\nExample:")
        console.print("  code-expert --repos-dir ~/projects index-all")
        return

    # Build pattern lists
    include_patterns = list(pattern) if pattern else None
    exclude_patterns = list(exclude) if exclude else None

    result = index_all_repositories(
        repos_dir=config.repos_dir,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        use_mock_embeddings=mock,
        incremental=incremental
    )

    # Don't print full JSON for index-all, just show summary
    if "_summary" in result:
        summary = result["_summary"]
        if summary["failed"] > 0 or summary["skipped"] > 0:
            # Show details for failed/skipped
            for repo, res in result.items():
                if repo == "_summary":
                    continue
                if res.get("status") == "error":
                    console.print(f"[red]  {repo}: {res.get('error', 'Unknown error')}[/red]")
                elif res.get("status") == "skipped":
                    console.print(f"[yellow]  {repo}: {res.get('reason', 'Unknown reason')}[/yellow]")


@main.command()
@click.argument("query")
@click.option("--results", "-n", default=10, help="Number of results")
@click.option("--repo", "-r", help="Filter by repository")
@click.option("--type", "-t", "type_filter", help="Filter by type (function, class, method)")
@click.option("--lang", "-l", help="Filter by language (python, java, typescript)")
@click.option("--threshold", default=None, type=float, help="Minimum similarity score (0-1)")
@click.option("--mock", is_flag=True, help="Use mock embeddings")
def search(query, results, repo, type_filter, lang, threshold, mock):
    """Semantic code search.

    QUERY: Natural language description of what you're looking for

    Uses hybrid search combining semantic similarity with keyword matching.
    Results are filtered by similarity threshold (default: 0.35).
    """
    result = semantic_grep(
        query=query,
        n_results=results,
        repo_filter=repo,
        type_filter=type_filter,
        language_filter=lang,
        use_mock_embeddings=mock,
        similarity_threshold=threshold,
    )

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")


@main.command()
@click.argument("symbol")
@click.option("--repo", "-r", help="Filter by repository")
@click.option("--no-related", is_flag=True, help="Don't include related context")
def context(symbol, repo, no_related):
    """Get code context for a symbol.

    SYMBOL: Name of the function, class, or method to look up
    """
    result = get_context(
        symbol_name=symbol,
        repo_filter=repo,
        include_related=not no_related,
    )

    if not result.get("found"):
        console.print(f"[yellow]{result.get('message', 'Symbol not found')}[/yellow]")


@main.command()
@click.option(
    "--pattern", "-p",
    multiple=True,
    help="Include only repos matching pattern"
)
@click.option(
    "--exclude", "-e",
    multiple=True,
    help="Exclude repos matching pattern"
)
def discover(pattern, exclude):
    """Discover repositories without indexing.

    Lists all repositories that would be indexed based on the current
    configuration and patterns. Useful for previewing what will be indexed.

    \b
    Examples:
        # Discover all repositories
        code-expert --repos-dir ~/projects discover

        # Discover only service repos
        code-expert --repos-dir ~/projects discover -p "service-*"

        # Exclude test repos
        code-expert --repos-dir ~/projects discover -e "test-*"
    """
    config = get_config()

    if not config.repos_dir or not config.repos_dir.exists():
        console.print("[red]Error: repos_dir not configured or doesn't exist[/red]")
        console.print("Use --repos-dir option to specify the directory")
        return

    include_patterns = list(pattern) if pattern else None
    exclude_patterns = list(exclude) if exclude else None

    repos = discover_repositories(
        repos_dir=config.repos_dir,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    if not repos:
        console.print("[yellow]No repositories found matching the criteria[/yellow]")
        return

    console.print(f"\n[bold]Discovered {len(repos)} Repositories[/bold]\n")

    table = Table()
    table.add_column("#", style="dim", justify="right")
    table.add_column("Repository", style="cyan")
    table.add_column("Path", style="dim")

    for idx, (name, path) in enumerate(sorted(repos.items()), 1):
        table.add_row(str(idx), name, str(path))

    console.print(table)
    console.print(f"\n[dim]Use 'index-all' to index these repositories[/dim]")


@main.command()
def stats():
    """Show index statistics."""
    storage = StorageService()
    stats = storage.get_stats()

    console.print("\n[bold]RepoMind Index Statistics[/bold]\n")

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", str(stats["total_chunks"]))
    table.add_row("Repositories", ", ".join(stats["repositories"]) or "None")

    console.print(table)

    if stats["languages"]:
        console.print("\n[bold]Languages:[/bold]")
        lang_table = Table()
        lang_table.add_column("Language")
        lang_table.add_column("Chunks", justify="right")
        for lang, count in sorted(stats["languages"].items(), key=lambda x: -x[1]):
            lang_table.add_row(lang, str(count))
        console.print(lang_table)

    if stats["chunk_types"]:
        console.print("\n[bold]Chunk Types:[/bold]")
        type_table = Table()
        type_table.add_column("Type")
        type_table.add_column("Count", justify="right")
        for ctype, count in sorted(stats["chunk_types"].items(), key=lambda x: -x[1]):
            type_table.add_row(ctype, str(count))
        console.print(type_table)


@main.command()
@click.argument("repo_name")
def clear(repo_name):
    """Clear index for a specific repository.

    REPO_NAME: Name of the repository to clear from index
    """
    storage = StorageService()
    removed = storage.clear_repo(repo_name)
    console.print(f"[green]Removed {removed} chunks for {repo_name}[/green]")


@main.command("clear-all")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def clear_all(force):
    """Clear the entire index (all repositories).

    WARNING: This deletes all indexed data!
    """
    if not force:
        console.print("[yellow]⚠️  WARNING: This will delete ALL indexed data![/yellow]")
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != "yes":
            console.print("[dim]Cancelled.[/dim]")
            return

    storage = StorageService()
    removed = storage.clear_all()
    console.print(f"[green]✓ Cleared entire index. Removed {removed} chunks.[/green]")
    console.print("[dim]You can now re-index repositories with the new embedding model.[/dim]")


@main.command()
def serve():
    """Run as MCP server for AI assistant integration.

    Starts an MCP (Model Context Protocol) server that enables
    AI assistants like Claude to use code search and analysis tools.

    \b
    Integration:
        - Claude Desktop: Add to ~/.claude/claude_desktop_config.json
        - VS Code Copilot: Add to ~/.vscode/mcp-servers.json
    """
    from .server import run_server

    console.print("[bold]Starting RepoMind MCP Server...[/bold]")
    run_server()


@main.command()
def config_show():
    """Show current configuration."""
    config = get_config()

    console.print("\n[bold]RepoMind Configuration[/bold]\n")

    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repos Directory", str(config.repos_dir) if config.repos_dir else "[dim]Not set[/dim]")
    table.add_row("Data Directory", str(config.index.data_dir))
    table.add_row("ChromaDB Directory", str(config.index.chroma_dir))
    table.add_row("Embedding Model", config.embedding.model)

    # Repository discovery settings
    table.add_row("Auto-discover Repos", "✓ Yes" if config.repository.auto_discover else "✗ No")
    table.add_row("Include Patterns", ", ".join(config.repository.include_patterns))
    table.add_row("Exclude Patterns", ", ".join(config.repository.exclude_patterns[:5]) +
                  ("..." if len(config.repository.exclude_patterns) > 5 else ""))

    # Explicit repos list if set
    if config.repos_list:
        table.add_row("Explicit Repos List", ", ".join(config.repos_list[:5]) +
                      ("..." if len(config.repos_list) > 5 else ""))

    console.print(table)

    console.print("\n[bold]Search Configuration[/bold]\n")

    search_table = Table()
    search_table.add_column("Setting", style="cyan")
    search_table.add_column("Value", style="green")

    search_table.add_row("Similarity Threshold", f"{config.search.similarity_threshold:.2f}")
    search_table.add_row("High Confidence Threshold", f"{config.search.high_confidence_threshold:.2f}")
    search_table.add_row("Hybrid Search", "✓ Enabled" if config.search.enable_hybrid_search else "✗ Disabled")
    search_table.add_row("Keyword Boost Factor", f"{config.search.keyword_boost_factor:.2f}")
    search_table.add_row("Query Expansion", "✓ Enabled" if config.search.enable_query_expansion else "✗ Disabled")

    console.print(search_table)

    # Show discovered repos if repos_dir is set
    if config.repos_dir and config.repos_dir.exists():
        console.print("\n[bold]Discovered Repositories[/bold]\n")
        try:
            repos = discover_repositories(config.repos_dir)
            if repos:
                repos_table = Table()
                repos_table.add_column("Repository", style="cyan")
                repos_table.add_column("Path", style="dim")
                for name, path in sorted(repos.items()):
                    repos_table.add_row(name, str(path))
                console.print(repos_table)
            else:
                console.print("[yellow]No repositories discovered in repos_dir[/yellow]")
        except Exception as e:
            console.print(f"[red]Error discovering repos: {e}[/red]")


if __name__ == "__main__":
    main()
