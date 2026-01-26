"""
Response Formatter Service - Token-Efficient Response Generation.

This module provides intelligent formatting of code search results
to optimize token usage when communicating with AI models.

Key Features:
1. Detail levels (summary/preview/full) for progressive disclosure
2. Token budgeting to stay within context limits
3. Result grouping by file, type, or repository
4. Automatic truncation with smart summarization

Token Efficiency Goals:
- SUMMARY mode: ~50 tokens per result (90% reduction)
- PREVIEW mode: ~200 tokens per result (60% reduction)
- FULL mode: Complete data (no reduction)

Example:
    formatter = ResponseFormatter()

    # Format for listing many results
    results = formatter.format_results(chunks, DetailLevel.SUMMARY)

    # Format with token budget
    results = formatter.format_results(
        chunks,
        DetailLevel.PREVIEW,
        token_budget=2000
    )

Author: RepoMind Team
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..models.chunk import CodeChunk, DetailLevel


class GroupBy(str, Enum):
    """Grouping options for search results."""
    NONE = "none"
    FILE = "file"
    TYPE = "type"
    REPO = "repo"


@dataclass
class FormatterConfig:
    """Configuration for response formatting."""
    default_detail_level: DetailLevel = DetailLevel.PREVIEW
    max_results: int = 50
    token_budget: Optional[int] = None  # None = unlimited
    group_by: GroupBy = GroupBy.NONE
    include_stats: bool = True
    preview_lines: int = 10


class ResponseFormatter:
    """
    Intelligent formatter for code search results.

    Handles conversion of CodeChunk objects to API responses
    with configurable detail levels and token budgeting.
    """

    # Approximate token counts per detail level
    TOKENS_PER_SUMMARY = 50
    TOKENS_PER_PREVIEW = 200
    TOKENS_PER_FULL = 500

    def __init__(self, config: Optional[FormatterConfig] = None):
        """
        Initialize the formatter with optional configuration.

        Args:
            config: Formatter configuration. Uses defaults if not provided.
        """
        self.config = config or FormatterConfig()

    def format_results(
        self,
        chunks: list[CodeChunk],
        detail_level: Optional[DetailLevel] = None,
        token_budget: Optional[int] = None,
        group_by: Optional[GroupBy] = None,
        max_results: Optional[int] = None,
        include_similarity: bool = True,
        similarities: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """
        Format search results at the specified detail level.

        Automatically manages token budget by adjusting detail level
        or truncating results as needed.

        Args:
            chunks: List of CodeChunk objects to format.
            detail_level: Detail level for formatting. Uses config default if None.
            token_budget: Maximum tokens to use. Uses config default if None.
            group_by: How to group results. Uses config default if None.
            max_results: Maximum results to return. Uses config default if None.
            include_similarity: Whether to include similarity scores.
            similarities: Optional list of similarity scores (parallel to chunks).

        Returns:
            Formatted response dictionary with results and metadata.

        Example:
            {
                "results": [...],
                "stats": {
                    "total_results": 25,
                    "returned_results": 10,
                    "detail_level": "preview",
                    "estimated_tokens": 2000
                }
            }
        """
        detail_level = detail_level or self.config.default_detail_level
        token_budget = token_budget or self.config.token_budget
        group_by = group_by or self.config.group_by
        max_results = max_results or self.config.max_results

        # Determine how many results we can fit in the token budget
        if token_budget:
            chunks, detail_level = self._fit_to_budget(
                chunks, detail_level, token_budget, max_results
            )
        elif max_results and len(chunks) > max_results:
            chunks = chunks[:max_results]

        # Format each chunk
        formatted = []
        for i, chunk in enumerate(chunks):
            result = chunk.to_response(detail_level)
            if include_similarity and similarities and i < len(similarities):
                result["similarity"] = round(similarities[i], 4)
            formatted.append(result)

        # Group if requested
        if group_by != GroupBy.NONE:
            formatted = self._group_results(formatted, group_by)

        # Build response
        response: dict[str, Any] = {"results": formatted}

        if self.config.include_stats:
            response["stats"] = {
                "total_found": len(chunks) if not token_budget else "unknown",
                "returned": len(formatted) if group_by == GroupBy.NONE else self._count_grouped(formatted),
                "detail_level": detail_level.value,
                "estimated_tokens": self._estimate_tokens(formatted, detail_level),
            }

        return response

    def format_single(
        self,
        chunk: CodeChunk,
        detail_level: Optional[DetailLevel] = None,
    ) -> dict[str, Any]:
        """
        Format a single chunk at the specified detail level.

        Args:
            chunk: The CodeChunk to format.
            detail_level: Detail level for formatting.

        Returns:
            Formatted chunk dictionary.
        """
        detail_level = detail_level or self.config.default_detail_level
        return chunk.to_response(detail_level)

    def _fit_to_budget(
        self,
        chunks: list[CodeChunk],
        detail_level: DetailLevel,
        token_budget: int,
        max_results: int,
    ) -> tuple[list[CodeChunk], DetailLevel]:
        """
        Adjust results and detail level to fit within token budget.

        Strategy:
        1. First try at requested detail level
        2. If over budget, reduce detail level
        3. If still over budget, truncate results

        Returns:
            Tuple of (adjusted chunks, adjusted detail level).
        """
        tokens_per_item = self._tokens_for_level(detail_level)
        max_by_budget = token_budget // tokens_per_item

        # If we can fit all results, return as-is
        if len(chunks) <= max_by_budget and len(chunks) <= max_results:
            return chunks, detail_level

        # Try reducing detail level
        if detail_level == DetailLevel.FULL:
            return self._fit_to_budget(chunks, DetailLevel.PREVIEW, token_budget, max_results)
        elif detail_level == DetailLevel.PREVIEW:
            # Check if SUMMARY would help
            summary_max = token_budget // self.TOKENS_PER_SUMMARY
            if summary_max >= len(chunks) or summary_max > max_by_budget:
                return self._fit_to_budget(chunks, DetailLevel.SUMMARY, token_budget, max_results)

        # Truncate to fit
        final_count = min(max_by_budget, max_results, len(chunks))
        return chunks[:final_count], detail_level

    def _tokens_for_level(self, detail_level: DetailLevel) -> int:
        """Get estimated tokens per result for a detail level."""
        if detail_level == DetailLevel.SUMMARY:
            return self.TOKENS_PER_SUMMARY
        elif detail_level == DetailLevel.PREVIEW:
            return self.TOKENS_PER_PREVIEW
        else:
            return self.TOKENS_PER_FULL

    def _estimate_tokens(
        self,
        results: list[dict[str, Any]],
        detail_level: DetailLevel,
    ) -> int:
        """Estimate total tokens for formatted results."""
        if not results:
            return 0

        # For grouped results, count nested items
        if isinstance(results, list) and results and "items" in results[0]:
            total = sum(len(group.get("items", [])) for group in results)
        else:
            total = len(results)

        return total * self._tokens_for_level(detail_level)

    def _group_results(
        self,
        results: list[dict[str, Any]],
        group_by: GroupBy,
    ) -> list[dict[str, Any]]:
        """
        Group results by the specified field.

        Args:
            results: Formatted results to group.
            group_by: Grouping criteria.

        Returns:
            List of grouped result dictionaries.
        """
        if group_by == GroupBy.NONE:
            return results

        # Determine grouping key
        key_map = {
            GroupBy.FILE: lambda r: r.get("location", "").split(":")[0],
            GroupBy.TYPE: lambda r: r.get("type", "unknown"),
            GroupBy.REPO: lambda r: r.get("repo", "unknown"),
        }
        key_fn = key_map.get(group_by, lambda r: "unknown")

        # Group results
        groups: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            key = key_fn(result)
            if key not in groups:
                groups[key] = []
            groups[key].append(result)

        # Convert to list format
        return [
            {
                "group": key,
                "count": len(items),
                "items": items,
            }
            for key, items in sorted(groups.items())
        ]

    def _count_grouped(self, grouped: list[dict[str, Any]]) -> int:
        """Count total items in grouped results."""
        return sum(group.get("count", 0) for group in grouped)


# Convenience functions for common formatting patterns

def format_search_results(
    chunks: list[CodeChunk],
    similarities: Optional[list[float]] = None,
    detail_level: DetailLevel = DetailLevel.PREVIEW,
    max_results: int = 20,
) -> dict[str, Any]:
    """
    Format semantic search results with similarity scores.

    Convenience function for the common case of formatting search results.

    Args:
        chunks: Search result chunks.
        similarities: Similarity scores for each chunk.
        detail_level: Level of detail to include.
        max_results: Maximum results to return.

    Returns:
        Formatted response with results and stats.
    """
    formatter = ResponseFormatter()
    return formatter.format_results(
        chunks=chunks,
        detail_level=detail_level,
        max_results=max_results,
        include_similarity=True,
        similarities=similarities,
    )


def format_code_context(
    chunks: list[CodeChunk],
    token_budget: int = 4000,
) -> dict[str, Any]:
    """
    Format code context for AI analysis within token budget.

    Automatically adjusts detail level to fit within budget
    while maximizing useful information.

    Args:
        chunks: Chunks to include in context.
        token_budget: Maximum tokens to use.

    Returns:
        Formatted context optimized for token budget.
    """
    config = FormatterConfig(
        default_detail_level=DetailLevel.FULL,
        token_budget=token_budget,
        include_stats=True,
    )
    formatter = ResponseFormatter(config)
    return formatter.format_results(chunks, token_budget=token_budget)


def chunks_to_summary_list(chunks: list[CodeChunk]) -> list[dict[str, Any]]:
    """
    Convert chunks to a minimal summary list.

    Useful for listing many results without detail.

    Args:
        chunks: Chunks to summarize.

    Returns:
        List of summary dictionaries.
    """
    return [chunk.to_summary() for chunk in chunks]
