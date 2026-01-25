"""
MCP Server for Code Expert.

This module implements the Model Context Protocol (MCP) server that exposes
code intelligence tools to AI assistants like Claude, VS Code Copilot, and others.

The MCP Protocol:
    MCP is a standard protocol for AI assistant integrations that allows
    tools to be discovered and invoked by AI systems. This server:

    1. Registers tools via @app.list_tools()
    2. Handles tool invocations via @app.call_tool()
    3. Communicates via stdio (standard input/output)

Available Tools:
    - index_repo: Index a single repository for semantic search
    - index_all_repositories: Discover and index all repositories
    - semantic_grep: Search code by meaning, not just text
    - get_context: Get full code context for a symbol
    - get_index_stats: View index statistics

Transport:
    Uses stdio transport, which means:
    - Communication happens via stdin/stdout pipes
    - No network ports are required
    - The calling process (Claude CLI, VS Code) spawns this as a subprocess

Usage:
    # Start the server directly
    python -m repomind.server

    # Or via the CLI
    code-expert serve

    # Or configured in MCP client (Claude Desktop, etc.)
    {
      "command": "python",
      "args": ["-m", "repomind.server"]
    }

Author: Code Expert Team
"""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .constants import APPLICATION_NAME, MCPToolName
from .logging import get_logger, log_operation_start, log_operation_end
from .tools.diff_impact import diff_impact
from .tools.file_summary import file_summary
from .tools.find_implementations import find_implementations
from .tools.find_tests import find_tests
from .tools.find_usages import find_usages
from .tools.get_context import get_context
from .tools.index_repo import index_all_repositories, index_repo
from .tools.semantic_grep import semantic_grep


# Module logger
logger = get_logger(__name__)

# Create MCP server instance
app = Server(APPLICATION_NAME.lower().replace(" ", "-"))


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="index_repo",
            description="""Index a repository for semantic code search.

Parses all source files, extracts semantic code chunks (functions, classes, methods),
generates embeddings, and stores them for later search.

Use this tool when you need to:
- Index a new repository for the first time
- Re-index a repository after code changes
- Add a repository to the code knowledge base""",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository to index",
                    },
                    "repo_name": {
                        "type": "string",
                        "description": "Optional custom name for the repository",
                    },
                },
                "required": ["repo_path"],
            },
        ),
        Tool(
            name="index_all_repositories",
            description="""Discover and index all repositories in the configured directory.

This auto-discovers all subdirectories that appear to be code repositories
and indexes each one. Supports pattern filtering for include/exclude.

Use this for:
- Initial setup of a new project workspace
- Full re-indexing after major changes
- Bulk indexing of multiple repositories""",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns for repos to include (e.g., ['service-*'])",
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns for repos to exclude (e.g., ['legacy-*'])",
                    },
                },
            },
        ),
        Tool(
            name="semantic_grep",
            description="""Search code semantically by meaning, not just text.

Unlike regular grep, this finds code that matches the MEANING of your query,
even if it doesn't contain the exact words.

Examples:
- "function that validates user input" → finds validation functions
- "error handling for API requests" → finds try/catch blocks, error handlers
- "authentication middleware" → finds auth-related middleware
- "database connection setup" → finds DB initialization code

Use this when you need to:
- Find code related to a concept or feature
- Understand how something is implemented
- Find similar code patterns across repositories""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what you're looking for",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 10)",
                        "default": 10,
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to specific repository",
                    },
                    "type_filter": {
                        "type": "string",
                        "description": "Filter by type: function, class, method, interface",
                    },
                    "language_filter": {
                        "type": "string",
                        "description": "Filter by language: python, java, typescript",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_context",
            description="""Get complete code context for a symbol (function, class, method).

Retrieves the full source code for a specific symbol along with related context
like parent class, sibling methods, and other symbols in the same file.

Use this when you:
- Need to see the full implementation of a function
- Want to understand a class and its methods
- Need context about how a symbol is used""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol (e.g., 'handleRequest', 'UserService.validate')",
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to specific repository",
                    },
                    "include_related": {
                        "type": "boolean",
                        "description": "Include related code (parent, siblings)",
                        "default": True,
                    },
                },
                "required": ["symbol_name"],
            },
        ),
        Tool(
            name="get_index_stats",
            description="""Get statistics about the indexed codebase.

Shows total chunks, repositories, languages, and chunk type breakdown.""",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="file_summary",
            description="""Get overview of symbols in a file without reading entire content.

Shows all classes, functions, and methods with their signatures and line numbers.
Useful for understanding file structure before diving into specific code.

Use this when you:
- Want to see what's in a file without reading all the code
- Need to understand file structure quickly
- Want to find specific functions by line number""",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (relative to repo root)",
                    },
                    "repo_name": {
                        "type": "string",
                        "description": "Filter to a specific repository",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="find_usages",
            description="""Find ALL references to a symbol across the codebase.

Unlike find_callers which only finds call sites, this finds every reference including:
- Direct function/method calls
- Type annotations and type hints
- Inheritance (extends, implements)
- Variable assignments and declarations
- Import statements

Use this for comprehensive symbol usage analysis.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol to find usages for",
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to a specific repository",
                    },
                    "include_definitions": {
                        "type": "boolean",
                        "description": "Include where the symbol is defined",
                        "default": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum usages to return",
                        "default": 50,
                    },
                },
                "required": ["symbol_name"],
            },
        ),
        Tool(
            name="find_implementations",
            description="""Find classes implementing an interface or extending a base class.

Uses the inheritance table to find all concrete implementations.
Supports finding both direct and transitive (indirect) implementations.

Use this when you:
- Need to find all implementations of an interface
- Want to see what classes extend a base class
- Are doing refactoring that affects a base type""",
            inputSchema={
                "type": "object",
                "properties": {
                    "interface_name": {
                        "type": "string",
                        "description": "Name of the interface or base class",
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to a specific repository",
                    },
                    "include_indirect": {
                        "type": "boolean",
                        "description": "Include transitive implementations",
                        "default": False,
                    },
                },
                "required": ["interface_name"],
            },
        ),
        Tool(
            name="find_tests",
            description="""Find test files and methods for a symbol using heuristics.

Discovers tests by checking:
- File name patterns (test_*.py, *Test.java, *.spec.ts)
- Test method names (test_*, should_*, it_*)
- Import analysis (test files importing the symbol)
- Content matching (references in test code)

Use this to find what tests cover a particular symbol.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol to find tests for",
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to a specific repository",
                    },
                },
                "required": ["symbol_name"],
            },
        ),
        Tool(
            name="diff_impact",
            description="""Analyze impact of recent git changes.

Examines git diff to find:
- What files have changed
- What symbols were modified
- What other code calls those symbols (blast radius)
- What tests might be affected

Use this before code review or to understand change impact.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository to analyze",
                    },
                    "since": {
                        "type": "string",
                        "description": "Git reference to compare against (default: HEAD~1)",
                        "default": "HEAD~1",
                    },
                    "include_tests": {
                        "type": "boolean",
                        "description": "Also find affected tests",
                        "default": True,
                    },
                },
                "required": ["repo_path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle incoming tool invocations from MCP clients.

    This is the main entry point for all tool calls. It:
    1. Logs the incoming request
    2. Routes to the appropriate tool handler
    3. Formats the response as JSON
    4. Handles and logs any errors

    Args:
        name: Name of the tool being invoked.
        arguments: Dictionary of arguments passed to the tool.

    Returns:
        List containing a single TextContent with JSON-formatted results.
    """
    start_time = log_operation_start(
        logger, f"tool_call:{name}",
        tool_name=name,
        arguments=arguments
    )

    try:
        result = await _execute_tool(name, arguments)

        log_operation_end(
            logger, f"tool_call:{name}", start_time,
            success=True,
            result_type=type(result).__name__
        )

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]

    except Exception as error:
        log_operation_end(
            logger, f"tool_call:{name}", start_time,
            success=False,
            error=str(error)
        )

        error_response = {
            "error": str(error),
            "tool": name,
            "arguments": arguments,
        }

        return [TextContent(
            type="text",
            text=json.dumps(error_response, indent=2)
        )]


async def _execute_tool(name: str, arguments: dict[str, Any]) -> dict:
    """
    Execute a specific tool by name.

    This internal function routes tool calls to their implementations.
    It's separated from call_tool for cleaner error handling.

    Args:
        name: Name of the tool to execute.
        arguments: Arguments for the tool.

    Returns:
        Dictionary with tool results.

    Raises:
        ValueError: If tool name is unknown.
    """
    import asyncio

    if name == MCPToolName.INDEX_REPO.value:
        # Run sync function in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: index_repo(
                repo_path=arguments["repo_path"],
                repo_name=arguments.get("repo_name"),
            )
        )

    elif name == MCPToolName.INDEX_ALL.value:
        # Run sync function in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        include_patterns = arguments.get("include_patterns")
        exclude_patterns = arguments.get("exclude_patterns")
        return await loop.run_in_executor(
            None,
            lambda: index_all_repositories(
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
        )

    elif name == MCPToolName.SEMANTIC_GREP.value:
        return semantic_grep(
            query=arguments["query"],
            n_results=arguments.get("n_results", 10),
            repo_filter=arguments.get("repo_filter"),
            type_filter=arguments.get("type_filter"),
            language_filter=arguments.get("language_filter"),
        )

    elif name == MCPToolName.GET_CONTEXT.value:
        return get_context(
            symbol_name=arguments["symbol_name"],
            repo_filter=arguments.get("repo_filter"),
            include_related=arguments.get("include_related", True),
        )

    elif name == MCPToolName.GET_INDEX_STATS.value:
        from .services.storage import StorageService
        storage = StorageService()
        return storage.get_stats()

    elif name == MCPToolName.FILE_SUMMARY.value:
        return file_summary(
            file_path=arguments["file_path"],
            repo_name=arguments.get("repo_name"),
        )

    elif name == MCPToolName.FIND_USAGES.value:
        return find_usages(
            symbol_name=arguments["symbol_name"],
            repo_filter=arguments.get("repo_filter"),
            include_definitions=arguments.get("include_definitions", False),
            limit=arguments.get("limit", 50),
        )

    elif name == MCPToolName.FIND_IMPLEMENTATIONS.value:
        return find_implementations(
            interface_name=arguments["interface_name"],
            repo_filter=arguments.get("repo_filter"),
            include_indirect=arguments.get("include_indirect", False),
        )

    elif name == MCPToolName.FIND_TESTS.value:
        return find_tests(
            symbol_name=arguments["symbol_name"],
            repo_filter=arguments.get("repo_filter"),
        )

    elif name == MCPToolName.DIFF_IMPACT.value:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: diff_impact(
                repo_path=arguments["repo_path"],
                since=arguments.get("since", "HEAD~1"),
                include_tests=arguments.get("include_tests", True),
            )
        )

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """
    Run the MCP server with stdio transport.

    This is the main entry point that:
    1. Sets up the stdio transport (stdin/stdout)
    2. Starts the MCP server
    3. Handles graceful shutdown
    """
    logger.info("Starting RepoMind MCP server")

    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server ready, waiting for connections")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

    logger.info("MCP server shut down")


def run_server():
    """Entry point for running the server."""
    asyncio.run(main())


if __name__ == "__main__":
    run_server()
