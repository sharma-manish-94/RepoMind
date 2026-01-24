"""
Repository Indexing Tools for Code Expert.

This module provides tools for indexing code repositories to enable
semantic search and code intelligence features.

Key Functions:
    - index_repo: Index a single repository
    - index_all_repositories: Discover and index all repositories
    - discover_repositories: Find repositories in a directory

The indexing process:
    1. Discover/validate repository path
    2. Extract code chunks (functions, classes, methods)
    3. Build symbol table for fast lookups
    4. Build call graph for relationship analysis
    5. Generate embeddings for semantic search
    6. Store in vector database

Author: Code Expert Team
"""

from pathlib import Path
from typing import Optional
import warnings

from rich.console import Console

from ..config import get_config
from ..models.chunk import CallInfo, CodeChunk
from ..services.call_graph import CallGraphService, CallRelation
from ..services.chunking import ChunkingService
from ..services.embedding import EmbeddingService, MockEmbeddingService
from ..services.manifest import FileStatus, ManifestService
from ..services.storage import StorageService
from ..services.symbol_table import SymbolTableService

console = Console()


def index_repo(
    repo_path: str,
    repo_name: Optional[str] = None,
    use_mock_embeddings: bool = False,
    incremental: bool = False,
) -> dict:
    """Index a repository for semantic search.

    This tool parses all source files in a repository, extracts semantic
    code chunks (functions, classes, methods), generates embeddings,
    and stores them for later search.

    Args:
        repo_path: Path to the repository to index
        repo_name: Optional name for the repository (defaults to directory name)
        use_mock_embeddings: Use mock embeddings for testing (no API calls)
        incremental: Only index changed files since last index

    Returns:
        Dictionary with indexing statistics
    """
    path = Path(repo_path).resolve()

    if not path.exists():
        return {"error": f"Repository path does not exist: {repo_path}"}

    if not path.is_dir():
        return {"error": f"Path is not a directory: {repo_path}"}

    name = repo_name or path.name

    if incremental:
        return _index_incremental(path, name, use_mock_embeddings)
    else:
        return _index_full(path, name, use_mock_embeddings)


def _index_full(
    path: Path,
    name: str,
    use_mock_embeddings: bool,
) -> dict:
    """Perform a full repository index."""
    console.print(f"[bold blue]Indexing repository: {name}[/bold blue]")

    # Initialize services
    chunking_service = ChunkingService()
    storage_service = StorageService()
    manifest_service = ManifestService()
    symbol_table = SymbolTableService()
    call_graph = CallGraphService()

    if use_mock_embeddings:
        embedding_service = MockEmbeddingService()
    else:
        embedding_service = EmbeddingService()

    # Step 1: Extract chunks and call information
    console.print("[dim]Step 1/5: Extracting code chunks and call relationships...[/dim]")
    chunking_result = chunking_service.chunk_repository_full(path, name)
    chunks = chunking_result.chunks
    calls = chunking_result.calls

    if not chunks:
        return {
            "repo_name": name,
            "status": "no_chunks",
            "message": "No code chunks found in repository",
        }

    # Step 2: Populate symbol table
    console.print(f"[dim]Step 2/5: Building symbol table ({len(chunks)} symbols)...[/dim]")
    symbol_table.delete_repo(name)  # Clear old symbols for this repo
    symbols_added = symbol_table.add_symbols_from_chunks(chunks)

    # Step 3: Populate call graph
    console.print(f"[dim]Step 3/5: Building call graph ({len(calls)} relations)...[/dim]")
    call_graph.delete_repo(name)  # Clear old call relations for this repo
    call_relations = [
        CallRelation(
            caller=call.caller_qualified_name,
            callee=call.callee_name,
            caller_file=call.caller_file,
            caller_line=call.caller_line,
            repo_name=name,
            call_type=call.call_type,
        )
        for call in calls
    ]
    calls_added = call_graph.add_calls_bulk(call_relations)

    # Step 4: Generate embeddings
    console.print(f"[dim]Step 4/5: Generating embeddings for {len(chunks)} chunks...[/dim]")
    embeddings = embedding_service.embed_chunks(chunks)

    # Step 5: Store in vector database
    console.print("[dim]Step 5/5: Storing chunks and embeddings...[/dim]")
    stored = storage_service.store_chunks(chunks, embeddings)

    # Update manifest with all indexed files
    _update_manifest_from_chunks(manifest_service, name, path, chunks)

    # Save the current git commit for future incremental indexing
    manifest_service.save_last_indexed_commit(name, path)

    # Get stats
    vector_stats = storage_service.get_stats()
    symbol_stats = symbol_table.get_stats()
    call_stats = call_graph.get_stats()

    result = {
        "repo_name": name,
        "status": "success",
        "mode": "full",
        "chunks_extracted": len(chunks),
        "chunks_stored": stored,
        "symbols_indexed": symbols_added,
        "call_relations_indexed": calls_added,
        "total_chunks_in_index": vector_stats["total_chunks"],
        "languages": vector_stats["languages"],
        "chunk_types": vector_stats["chunk_types"],
    }

    console.print(f"[bold green]Successfully indexed {name}[/bold green]")
    console.print(f"  Chunks: {stored}")
    console.print(f"  Symbols: {symbols_added}")
    console.print(f"  Call relations: {calls_added}")
    console.print(f"  Languages: {vector_stats['languages']}")

    return result


def _index_incremental(
    path: Path,
    name: str,
    use_mock_embeddings: bool,
) -> dict:
    """Perform an incremental repository index."""
    console.print(f"[bold blue]Incremental indexing: {name}[/bold blue]")

    # Initialize services
    chunking_service = ChunkingService()
    storage_service = StorageService()
    manifest_service = ManifestService()
    symbol_table = SymbolTableService()
    call_graph = CallGraphService()

    if use_mock_embeddings:
        embedding_service = MockEmbeddingService()
    else:
        embedding_service = EmbeddingService()

    # Step 1: Detect changes
    console.print("[dim]Step 1/4: Detecting changes...[/dim]")

    # Get list of current source files
    source_files = chunking_service._find_source_files(path)
    changes = manifest_service.get_file_changes(path, name, source_files, use_git=True)

    # Categorize changes
    new_files = [c for c in changes if c.status == FileStatus.NEW]
    modified_files = [c for c in changes if c.status == FileStatus.MODIFIED]
    deleted_files = [c for c in changes if c.status == FileStatus.DELETED]

    console.print(f"  New: {len(new_files)}, Modified: {len(modified_files)}, Deleted: {len(deleted_files)}")

    if not changes:
        console.print("[green]No changes detected. Index is up to date.[/green]")
        return {
            "repo_name": name,
            "status": "up_to_date",
            "mode": "incremental",
            "message": "No changes detected",
        }

    # Step 2: Handle deletions
    chunks_deleted = 0
    if deleted_files:
        console.print(f"[dim]Step 2/4: Removing {len(deleted_files)} deleted files...[/dim]")
        for change in deleted_files:
            if change.old_chunk_ids:
                storage_service.delete_chunks(change.old_chunk_ids)
                chunks_deleted += len(change.old_chunk_ids)
            # Also remove from symbol table and call graph
            symbol_table.delete_file(name, change.relative_path)
            call_graph.delete_file(name, change.relative_path)
            manifest_service.remove_manifest_entry(name, change.relative_path)
    else:
        console.print("[dim]Step 2/4: No deletions to process[/dim]")

    # Step 3: Handle modifications (delete old chunks first)
    if modified_files:
        console.print(f"[dim]Step 3/4: Processing {len(modified_files)} modified files...[/dim]")
        for change in modified_files:
            if change.old_chunk_ids:
                storage_service.delete_chunks(change.old_chunk_ids)
                chunks_deleted += len(change.old_chunk_ids)
            # Also remove from symbol table and call graph
            symbol_table.delete_file(name, change.relative_path)
            call_graph.delete_file(name, change.relative_path)
    else:
        console.print("[dim]Step 3/4: No modifications to process[/dim]")

    # Step 4: Parse and index new/modified files
    files_to_index = new_files + modified_files
    stored = 0
    symbols_added = 0
    calls_added = 0

    if files_to_index:
        console.print(f"[dim]Step 4/4: Indexing {len(files_to_index)} files...[/dim]")

        all_new_chunks: list[CodeChunk] = []
        all_new_calls: list[CallInfo] = []

        for change in files_to_index:
            if not change.path.exists():
                continue

            parse_result = chunking_service.chunk_file_full(change.path, name)

            # Update file paths to be relative
            for chunk in parse_result.chunks:
                chunk.file_path = change.relative_path
            for call in parse_result.calls:
                call.caller_file = change.relative_path

            all_new_chunks.extend(parse_result.chunks)
            all_new_calls.extend(parse_result.calls)

            # Update manifest for this file
            chunk_ids = [c.id for c in parse_result.chunks]
            manifest_service.update_manifest_entry(name, change.relative_path, change.path, chunk_ids)

        if all_new_chunks:
            # Add to symbol table
            symbols_added = symbol_table.add_symbols_from_chunks(all_new_chunks)

            # Add to call graph
            call_relations = [
                CallRelation(
                    caller=call.caller_qualified_name,
                    callee=call.callee_name,
                    caller_file=call.caller_file,
                    caller_line=call.caller_line,
                    repo_name=name,
                    call_type=call.call_type,
                )
                for call in all_new_calls
            ]
            calls_added = call_graph.add_calls_bulk(call_relations)

            # Generate embeddings
            embeddings = embedding_service.embed_chunks(all_new_chunks)

            # Store
            stored = storage_service.store_chunks(all_new_chunks, embeddings)
    else:
        console.print("[dim]Step 4/4: No files to index[/dim]")

    # Save the current git commit
    manifest_service.save_last_indexed_commit(name, path)

    # Get stats
    stats = storage_service.get_stats()

    result = {
        "repo_name": name,
        "status": "success",
        "mode": "incremental",
        "files_new": len(new_files),
        "files_modified": len(modified_files),
        "files_deleted": len(deleted_files),
        "chunks_added": stored,
        "chunks_removed": chunks_deleted,
        "symbols_added": symbols_added,
        "calls_added": calls_added,
        "total_chunks_in_index": stats["total_chunks"],
    }

    console.print(f"[bold green]Incremental index complete for {name}[/bold green]")
    console.print(f"  Chunks: +{stored}/-{chunks_deleted}")
    console.print(f"  Symbols: +{symbols_added}")
    console.print(f"  Call relations: +{calls_added}")

    return result


def _update_manifest_from_chunks(
    manifest_service: ManifestService,
    repo_name: str,
    repo_path: Path,
    chunks: list[CodeChunk],
) -> None:
    """Update manifest with all chunks from a full index."""
    # Group chunks by file
    chunks_by_file: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.file_path not in chunks_by_file:
            chunks_by_file[chunk.file_path] = []
        chunks_by_file[chunk.file_path].append(chunk.id)

    # Update manifest for each file
    for rel_path, chunk_ids in chunks_by_file.items():
        abs_path = repo_path / rel_path
        if abs_path.exists():
            manifest_service.update_manifest_entry(repo_name, rel_path, abs_path, chunk_ids)


def discover_repositories(
    repos_dir: Optional[Path] = None,
    include_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> dict[str, Path]:
    """
    Discover repositories in a directory.

    Scans a directory for subdirectories that appear to be code repositories,
    filtering based on include/exclude patterns.

    Args:
        repos_dir: Directory to scan for repositories.
                  Uses config repos_dir if not provided.
        include_patterns: Glob patterns for repos to include (default: ["*"])
        exclude_patterns: Glob patterns for repos to exclude

    Returns:
        Dictionary mapping repository names to their paths

    Raises:
        ValueError: If repos_dir is not set and not provided

    Example:
        >>> repos = discover_repositories(
        ...     Path("/path/to/projects"),
        ...     include_patterns=["service-*", "lib-*"],
        ...     exclude_patterns=["legacy-*"]
        ... )
        >>> for name, path in repos.items():
        ...     print(f"{name}: {path}")
    """
    config = get_config()

    # Determine repos_dir
    scan_dir = repos_dir or config.repos_dir
    if scan_dir is None:
        raise ValueError(
            "repos_dir must be provided or configured via set_config()"
        )

    scan_dir = Path(scan_dir).resolve()

    if not scan_dir.exists():
        console.print(f"[yellow]Warning: Directory does not exist: {scan_dir}[/yellow]")
        return {}

    if not scan_dir.is_dir():
        console.print(f"[yellow]Warning: Path is not a directory: {scan_dir}[/yellow]")
        return {}

    # Determine patterns
    repo_config = config.repository
    patterns_include = include_patterns or repo_config.include_patterns
    patterns_exclude = exclude_patterns or repo_config.exclude_patterns

    discovered: dict[str, Path] = {}

    console.print(f"[dim]Scanning for repositories in: {scan_dir}[/dim]")

    for item in sorted(scan_dir.iterdir()):
        if not item.is_dir():
            continue

        repo_name = item.name

        # Check exclude patterns
        should_exclude = any(
            _matches_pattern(repo_name, pattern)
            for pattern in patterns_exclude
        )
        if should_exclude:
            continue

        # Check include patterns
        should_include = any(
            _matches_pattern(repo_name, pattern)
            for pattern in patterns_include
        )
        if not should_include:
            continue

        # Validate as repository (has source files)
        if config._is_valid_repository(item):
            discovered[repo_name] = item

    console.print(f"[dim]Discovered {len(discovered)} repositories[/dim]")

    return discovered


def _matches_pattern(name: str, pattern: str) -> bool:
    """
    Check if a name matches a glob pattern.

    Args:
        name: String to check
        pattern: Glob pattern to match against

    Returns:
        True if name matches pattern
    """
    import fnmatch
    return fnmatch.fnmatch(name, pattern)


def index_all_repositories(
    repos_dir: Optional[Path] = None,
    include_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
    use_mock_embeddings: bool = False,
    incremental: bool = False,
) -> dict:
    """
    Discover and index all repositories in a directory.

    This is the main entry point for indexing multiple repositories.
    It discovers repositories based on patterns and indexes each one.

    Args:
        repos_dir: Directory containing repositories.
                  Uses config repos_dir if not provided.
        include_patterns: Glob patterns for repos to include
        exclude_patterns: Glob patterns for repos to exclude
        use_mock_embeddings: Use mock embeddings for testing
        incremental: Only index changed files

    Returns:
        Dictionary with results for each repository:
        {
            "repo_name": {
                "status": "success" | "error" | "skipped",
                "chunks_stored": int,
                ...
            },
            ...
            "_summary": {
                "total_repos": int,
                "successful": int,
                "failed": int,
                "skipped": int
            }
        }

    Example:
        >>> results = index_all_repositories(
        ...     Path("/path/to/projects"),
        ...     include_patterns=["service-*"],
        ...     incremental=True
        ... )
        >>> print(f"Indexed {results['_summary']['successful']} repos")
    """
    config = get_config()

    # Determine repos_dir
    scan_dir = repos_dir or config.repos_dir
    if scan_dir is None:
        return {
            "error": "repos_dir must be provided or configured",
            "_summary": {"total_repos": 0, "successful": 0, "failed": 1, "skipped": 0}
        }

    scan_dir = Path(scan_dir).resolve()

    # Check if explicit repos_list is provided (overrides discovery)
    if config.repos_list is not None:
        discovered = {
            name: scan_dir / name
            for name in config.repos_list
        }
        console.print(f"[bold]Using explicit repos_list ({len(discovered)} repositories)[/bold]")
    else:
        # Discover repositories
        console.print("[bold]Discovering repositories...[/bold]")
        discovered = discover_repositories(
            repos_dir=scan_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

    if not discovered:
        console.print("[yellow]No repositories found to index[/yellow]")
        return {
            "error": "No repositories found",
            "_summary": {"total_repos": 0, "successful": 0, "failed": 0, "skipped": 0}
        }

    console.print(f"\n[bold]Indexing {len(discovered)} repositories...[/bold]\n")

    results: dict = {}
    successful = 0
    failed = 0
    skipped = 0

    for repo_name, repo_path in discovered.items():
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Processing: {repo_name}[/bold]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")

        if not repo_path.exists():
            results[repo_name] = {
                "status": "skipped",
                "reason": "directory not found"
            }
            skipped += 1
            console.print(f"[yellow]Skipping {repo_name}: directory not found[/yellow]")
            continue

        try:
            result = index_repo(
                str(repo_path),
                repo_name,
                use_mock_embeddings=use_mock_embeddings,
                incremental=incremental,
            )
            results[repo_name] = result

            if result.get("status") in ("success", "up_to_date"):
                successful += 1
            elif result.get("status") == "error":
                failed += 1
            else:
                skipped += 1

        except Exception as e:
            results[repo_name] = {
                "status": "error",
                "error": str(e)
            }
            failed += 1
            console.print(f"[red]Error indexing {repo_name}: {e}[/red]")

    # Add summary
    results["_summary"] = {
        "total_repos": len(discovered),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
    }

    # Print summary
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold]INDEXING COMPLETE[/bold]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"  Total repositories: {len(discovered)}")
    console.print(f"  [green]Successful: {successful}[/green]")
    if failed > 0:
        console.print(f"  [red]Failed: {failed}[/red]")
    if skipped > 0:
        console.print(f"  [yellow]Skipped: {skipped}[/yellow]")

    return results


