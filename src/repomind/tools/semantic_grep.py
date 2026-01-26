"""Semantic grep tool for MCP - search code by meaning."""

from typing import Optional

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..config import get_config
from ..models.chunk import ChunkType, DetailLevel
from ..services.embedding import EmbeddingService, MockEmbeddingService
from ..services.storage import StorageService
from ..services.response_formatter import ResponseFormatter, format_search_results

# Use wide console to avoid truncation - important for MCP/CLI output
console = Console(width=200, force_terminal=False)


def semantic_grep(
    query: str,
    n_results: int = 10,
    repo_filter: Optional[str] = None,
    type_filter: Optional[str] = None,
    language_filter: Optional[str] = None,
    show_code: bool = True,
    use_mock_embeddings: bool = False,
    similarity_threshold: Optional[float] = None,
    detail_level: Optional[str] = None,
    token_budget: Optional[int] = None,
) -> dict:
    """Search code semantically by meaning rather than exact text match.

    This tool finds code that is semantically similar to your query,
    even if it doesn't contain the exact words you searched for.

    Uses hybrid search combining:
    - Semantic similarity (vector embeddings)
    - Keyword matching (name, docstring, content)

    Args:
        query: Natural language description of what you're looking for
            Examples:
            - "function that validates user input"
            - "error handling for API requests"
            - "authentication middleware"
        n_results: Maximum number of results to return (default: 10)
        repo_filter: Filter results to a specific repository
        type_filter: Filter by chunk type (function, class, method, etc.)
        language_filter: Filter by programming language (python, java, typescript)
        show_code: Whether to include code snippets in results (deprecated, use detail_level)
        use_mock_embeddings: Use mock embeddings for testing
        similarity_threshold: Minimum similarity score (0-1) to include (default: from config)
        detail_level: Level of detail in results: "summary" (~50 tokens/result),
            "preview" (~200 tokens/result), or "full" (~500+ tokens/result).
            Default: "full" for backward compatibility.
        token_budget: Optional token budget. If set, automatically adjusts
            detail_level and result count to fit within budget.

    Returns:
        Dictionary containing search results with code chunks and relevance scores.
        Format varies by detail_level:
        - summary: name, signature, location only
        - preview: + docstring and first 10 lines
        - full: complete code content
    """
    if not query.strip():
        return {"error": "Query cannot be empty"}

    console.print(f"[bold blue]Searching for: {query}[/bold blue]")

    # Get configuration
    config = get_config()

    # Use configured threshold if not explicitly provided
    if similarity_threshold is None:
        similarity_threshold = config.search.similarity_threshold

    # Initialize services
    storage_service = StorageService()

    if use_mock_embeddings:
        embedding_service = MockEmbeddingService()
    else:
        embedding_service = EmbeddingService()

    # Parse type filter
    chunk_type = None
    if type_filter:
        try:
            chunk_type = ChunkType(type_filter.lower())
        except ValueError:
            valid_types = [t.value for t in ChunkType]
            return {"error": f"Invalid type_filter. Valid options: {valid_types}"}

    # Generate query embedding
    query_embedding = embedding_service.embed_query(query)

    # Search with hybrid matching
    results = storage_service.search(
        query_embedding=query_embedding,
        n_results=n_results,
        repo_filter=repo_filter,
        type_filter=chunk_type,
        language_filter=language_filter,
        query_text=query,  # Pass original query for keyword boosting
        similarity_threshold=similarity_threshold,
        enable_hybrid_search=config.search.enable_hybrid_search,
    )

    if not results:
        return {
            "query": query,
            "results": [],
            "message": f"No matching code found (threshold: {similarity_threshold:.2f})",
        }

    # Parse detail level
    parsed_detail_level = DetailLevel.FULL  # Default for backward compatibility
    if detail_level:
        try:
            parsed_detail_level = DetailLevel(detail_level.lower())
        except ValueError:
            valid_levels = [d.value for d in DetailLevel]
            return {"error": f"Invalid detail_level. Valid options: {valid_levels}"}

    # If show_code is explicitly False and no detail_level set, use SUMMARY
    if not show_code and detail_level is None:
        parsed_detail_level = DetailLevel.SUMMARY

    # Extract chunks and scores
    chunks = [chunk for chunk, _ in results]
    scores = [score for _, score in results]

    # Use ResponseFormatter if token_budget is set or detail_level is not FULL
    if token_budget or parsed_detail_level != DetailLevel.FULL:
        formatter = ResponseFormatter()
        formatted_response = formatter.format_results(
            chunks=chunks,
            detail_level=parsed_detail_level,
            token_budget=token_budget,
            max_results=n_results,
            include_similarity=True,
            similarities=scores,
        )

        # Display table for console output
        _display_results_table(results, config, query)

        return {
            "query": query,
            "detail_level": parsed_detail_level.value,
            **formatted_response,
        }

    # Legacy format for backward compatibility (FULL detail level, no token budget)
    formatted_results = []
    table = Table(title=f"Search Results for: {query}", show_lines=False)
    table.add_column("Score", style="cyan", width=10, no_wrap=True)
    table.add_column("Type", style="magenta", width=12, no_wrap=True)
    table.add_column("Name", style="green", overflow="fold")
    table.add_column("Location", style="dim", overflow="fold")

    for chunk, score in results:
        # Score is already a similarity score (0-1) from the updated search method
        result_entry = {
            "score": round(score, 3),
            "chunk_type": chunk.chunk_type.value,
            "name": chunk.get_qualified_name(),
            "repo": chunk.repo_name,
            "file": chunk.file_path,
            "line": chunk.start_line,
            "language": chunk.language,
        }

        if chunk.docstring:
            result_entry["docstring"] = chunk.docstring

        if chunk.signature:
            result_entry["signature"] = chunk.signature

        if show_code:
            result_entry["code"] = chunk.content

        formatted_results.append(result_entry)

        # Add to display table with confidence indicator
        confidence = "🟢" if score >= config.search.high_confidence_threshold else "🟡" if score >= 0.5 else "🔴"
        location = f"{chunk.repo_name}/{chunk.file_path}:{chunk.start_line}"
        table.add_row(
            f"{confidence} {score:.2f}",
            chunk.chunk_type.value,
            chunk.get_qualified_name(),
            location,
        )

    console.print(table)

    # Show top result's code
    if show_code and formatted_results:
        top_result = formatted_results[0]
        console.print(f"\n[bold]Top match: {top_result['name']}[/bold]")
        console.print(f"[dim]{top_result['repo']}/{top_result['file']}:{top_result['line']}[/dim]")
        if "code" in top_result:
            syntax = Syntax(
                top_result["code"],
                top_result["language"],
                line_numbers=True,
                start_line=top_result["line"],
            )
            console.print(syntax)

    return {
        "query": query,
        "total_results": len(formatted_results),
        "results": formatted_results,
    }


def _display_results_table(
    results: list,
    config,
    query: str,
) -> None:
    """Display search results in a console table."""
    table = Table(title=f"Search Results for: {query}", show_lines=False)
    table.add_column("Score", style="cyan", width=10, no_wrap=True)
    table.add_column("Type", style="magenta", width=12, no_wrap=True)
    table.add_column("Name", style="green", overflow="fold")
    table.add_column("Location", style="dim", overflow="fold")

    for chunk, score in results:
        confidence = "🟢" if score >= config.search.high_confidence_threshold else "🟡" if score >= 0.5 else "🔴"
        location = f"{chunk.repo_name}/{chunk.file_path}:{chunk.start_line}"
        table.add_row(
            f"{confidence} {score:.2f}",
            chunk.chunk_type.value,
            chunk.get_qualified_name(),
            location,
        )

    console.print(table)
