"""
MCP Server for RepoMind.

This module implements the Model Context Protocol (MCP) server that exposes
code intelligence tools to AI assistants like Claude, VS Code Copilot, and others.

The MCP Protocol:
    MCP is a standard protocol for AI assistant integrations that allows
    tools to be discovered and invoked by AI systems. This server:

    1. Registers tools via @app.list_tools()
    2. Handles tool invocations via @app.call_tool()
    3. Communicates via stdio (standard input/output)

Available Tools (21 total):
    Indexing:
    - index_repo: Index a single repository for semantic search
    - index_all_repositories: Discover and index all repositories

    Search & Navigation:
    - semantic_grep: Search code by meaning (supports detail_level and token_budget)
    - get_context: Get full code context for a symbol
    - get_index_stats: View index statistics
    - file_summary: Get file structure overview

    Code Analysis:
    - find_usages: Find all references to a symbol
    - find_implementations: Find interface/base class implementations
    - find_tests: Find tests for a symbol
    - diff_impact: Analyze impact of git changes

    Compound Operations (Token-Efficient):
    - explore: Comprehensive symbol exploration in one call (60-75% token savings)
    - understand: Deep understanding of code behavior and dependencies
    - prepare_change: Impact analysis with risk assessment before modifications

    Pattern Analysis:
    - analyze_patterns: Analyze code patterns, library usage, and conventions
    - get_coding_conventions: Get conventions summary for AI code generation

    Production Features:
    - analyze_ownership: CODEOWNERS + git blame analysis, reviewer suggestions
    - scan_secrets: Detect 26 types of hardcoded secrets and credentials
    - get_metrics: Cyclomatic/cognitive complexity, maintainability index

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

Author: RepoMind Team
"""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .constants import APPLICATION_NAME, MCPToolName
from .logging import get_logger, log_operation_start, log_operation_end
from .services.metrics import MetricsService
from .services.ownership import OwnershipService
from .services.security_scanner import SecurityScanner
from .tools.analyze_patterns import analyze_patterns, get_coding_conventions
from .tools.compound_ops import explore, understand, prepare_change
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

Supports detail levels for token efficiency:
- "summary": ~50 tokens/result (name, signature, location only)
- "preview": ~200 tokens/result (+ docstring and first 10 lines)
- "full": ~500+ tokens/result (complete code content)

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
                    "detail_level": {
                        "type": "string",
                        "description": "Level of detail: 'summary' (~50 tokens), 'preview' (~200 tokens), 'full' (~500+ tokens)",
                        "enum": ["summary", "preview", "full"],
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Optional token budget - automatically adjusts detail level and result count",
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
        # ====================================================================
        # Compound Operations (Token-Efficient)
        # ====================================================================
        Tool(
            name="explore",
            description="""Comprehensive exploration of a symbol in one operation.

Combines multiple queries into a single, token-efficient response:
- Symbol definition with signature and docstring
- Direct callers (who uses this?)
- Direct callees (what does this use?)
- Related tests
- Impact radius (change blast area)

Use this when you want a complete overview of a symbol without
multiple round-trips. Saves 60-75% tokens compared to separate calls.

Example: explore("UserService.save") returns definition, all callers,
all callees, and tests in a single response.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol (e.g., 'UserService.save', 'handleRequest')",
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to a specific repository",
                    },
                    "depth": {
                        "type": "string",
                        "description": "Exploration depth: 'shallow', 'normal', or 'deep'",
                        "default": "normal",
                        "enum": ["shallow", "normal", "deep"],
                    },
                    "max_callers": {
                        "type": "integer",
                        "description": "Maximum callers to return",
                        "default": 10,
                    },
                    "max_callees": {
                        "type": "integer",
                        "description": "Maximum callees to return",
                        "default": 10,
                    },
                    "detail_level": {
                        "type": "string",
                        "description": "Detail level: 'summary', 'preview', or 'full'",
                        "default": "summary",
                        "enum": ["summary", "preview", "full"],
                    },
                },
                "required": ["symbol_name"],
            },
        ),
        Tool(
            name="understand",
            description="""Deep understanding of a symbol's behavior and dependencies.

Goes beyond explore() to provide:
- Full implementation code
- Type hierarchy (implements/extends)
- Dependency chain (what it needs to work)
- Data flow (input → processing → output)
- Usage summary across the codebase

Use this when you need to truly understand HOW code works,
not just find WHERE it is.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol to understand",
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to a specific repository",
                    },
                    "include_implementation": {
                        "type": "boolean",
                        "description": "Include full code implementation",
                        "default": True,
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth for dependency tracking",
                        "default": 2,
                    },
                },
                "required": ["symbol_name"],
            },
        ),
        Tool(
            name="prepare_change",
            description="""Impact analysis to prepare for modifying a symbol.

Analyzes what would be affected by changing a symbol:
- Direct dependents (will definitely be affected)
- Transitive dependents (might be affected)
- Potential breaking changes
- Tests that need updating
- Files that need review

Use BEFORE making changes to understand the blast radius and risk level.

Returns a risk assessment: low/medium/high/critical.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol you plan to change",
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to a specific repository",
                    },
                    "change_type": {
                        "type": "string",
                        "description": "Type of change: 'modify', 'rename', 'delete', or 'signature'",
                        "default": "modify",
                        "enum": ["modify", "rename", "delete", "signature"],
                    },
                },
                "required": ["symbol_name"],
            },
        ),
        # ====================================================================
        # Pattern Analysis Tools
        # ====================================================================
        Tool(
            name="analyze_patterns",
            description="""Analyze code patterns and conventions used in the codebase.

Helps understand what patterns, libraries, and conventions the team uses
so you can generate code that matches the existing style.

Analyzes:
- Library usage (axios vs fetch, pytest vs unittest, etc.)
- Code patterns (constructor injection vs field injection)
- Testing conventions (file patterns, assertions, mock libraries)
- Exemplary "golden files" that demonstrate best practices

Use this when you need to:
- Understand team conventions before writing code
- Choose between libraries or patterns
- Find example implementations to follow""",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Pattern category to analyze",
                        "enum": [
                            "dependency_injection",
                            "error_handling",
                            "async_patterns",
                            "http_client",
                            "testing",
                        ],
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to a specific repository",
                    },
                    "full_summary": {
                        "type": "boolean",
                        "description": "Get comprehensive summary of all conventions",
                        "default": False,
                    },
                    "include_golden_files": {
                        "type": "boolean",
                        "description": "Include exemplary files in results",
                        "default": False,
                    },
                    "include_testing": {
                        "type": "boolean",
                        "description": "Include testing convention analysis",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="get_coding_conventions",
            description="""Get coding conventions to follow when generating code.

Returns a concise summary of:
- Preferred libraries for each purpose (HTTP, testing, etc.)
- Recommended patterns (DI, error handling, etc.)
- Testing framework and conventions
- Reference files to use as examples

This is the quick way to learn "how to write code like this team".""",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to a specific repository",
                    },
                },
            },
        ),
        # ====================================================================
        # Production Features (Phase 4)
        # ====================================================================
        Tool(
            name="analyze_ownership",
            description="""Analyze code ownership and suggest reviewers.

Uses CODEOWNERS files and git blame data to determine:
- Who owns specific files or directories
- Who recently contributed to affected code
- Who should review changes
- Contributor statistics

Use this when:
- Planning code reviews
- Understanding who to ask about specific code
- Analyzing team contributions""",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the git repository",
                    },
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific files to analyze ownership for",
                    },
                    "suggest_reviewers": {
                        "type": "boolean",
                        "description": "Suggest reviewers for the given files",
                        "default": False,
                    },
                    "exclude_author": {
                        "type": "string",
                        "description": "Author to exclude from reviewer suggestions (e.g., PR author)",
                    },
                },
                "required": ["repo_path"],
            },
        ),
        Tool(
            name="scan_secrets",
            description="""Scan code for hardcoded secrets and credentials.

Detects 26 types of secrets including:
- AWS access keys and secret keys
- GitHub/GitLab tokens
- Slack tokens and webhooks
- Private keys (RSA, SSH)
- Database connection strings with passwords
- API keys, bearer tokens, passwords
- Stripe, SendGrid, Twilio keys

Returns findings with severity levels (critical/high/medium/low)
and remediation recommendations.

Use this before committing code or during security reviews.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository to scan",
                    },
                    "repo_filter": {
                        "type": "string",
                        "description": "Filter to a specific indexed repository (scans chunks instead of files)",
                    },
                    "min_severity": {
                        "type": "string",
                        "description": "Minimum severity to report",
                        "default": "low",
                        "enum": ["info", "low", "medium", "high", "critical"],
                    },
                },
            },
        ),
        Tool(
            name="get_metrics",
            description="""Get code complexity and quality metrics.

Calculates:
- Cyclomatic complexity (decision paths through code)
- Cognitive complexity (human readability difficulty)
- Lines of code metrics (SLOC, comment ratio)
- Maintainability index (0-100 scale)
- Complexity hotspots (most complex functions)

Use this to:
- Identify complex code that needs refactoring
- Monitor code quality trends
- Find functions with too many decision branches""",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_filter": {
                        "type": "string",
                        "description": "Repository to analyze metrics for",
                    },
                    "symbol_name": {
                        "type": "string",
                        "description": "Analyze a specific function/method",
                    },
                },
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
            detail_level=arguments.get("detail_level"),
            token_budget=arguments.get("token_budget"),
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

    # ========================================================================
    # Compound Operations (Token-Efficient)
    # ========================================================================

    elif name == MCPToolName.EXPLORE.value:
        return explore(
            symbol_name=arguments["symbol_name"],
            repo_filter=arguments.get("repo_filter"),
            depth=arguments.get("depth", "normal"),
            max_callers=arguments.get("max_callers", 10),
            max_callees=arguments.get("max_callees", 10),
            detail_level=arguments.get("detail_level", "summary"),
        )

    elif name == MCPToolName.UNDERSTAND.value:
        return understand(
            symbol_name=arguments["symbol_name"],
            repo_filter=arguments.get("repo_filter"),
            include_implementation=arguments.get("include_implementation", True),
            max_depth=arguments.get("max_depth", 2),
        )

    elif name == MCPToolName.PREPARE_CHANGE.value:
        return prepare_change(
            symbol_name=arguments["symbol_name"],
            repo_filter=arguments.get("repo_filter"),
            change_type=arguments.get("change_type", "modify"),
        )

    # ========================================================================
    # Pattern Analysis Tools
    # ========================================================================

    elif name == MCPToolName.ANALYZE_PATTERNS.value:
        return analyze_patterns(
            category=arguments.get("category"),
            repo_filter=arguments.get("repo_filter"),
            full_summary=arguments.get("full_summary", False),
            include_golden_files=arguments.get("include_golden_files", False),
            include_testing=arguments.get("include_testing", False),
        )

    elif name == MCPToolName.GET_CODING_CONVENTIONS.value:
        return get_coding_conventions(
            repo_filter=arguments.get("repo_filter"),
        )

    # ========================================================================
    # Production Features (Phase 4)
    # ========================================================================

    elif name == MCPToolName.ANALYZE_OWNERSHIP.value:
        repo_path = arguments["repo_path"]
        service = OwnershipService(repo_path)

        file_paths = arguments.get("file_paths")
        suggest = arguments.get("suggest_reviewers", False)
        exclude = arguments.get("exclude_author")

        result = service.get_ownership_summary(file_paths)

        if suggest and file_paths:
            reviewers = service.suggest_reviewers(
                file_paths,
                exclude_author=exclude,
            )
            result["suggested_reviewers"] = [
                {
                    "name": r.name,
                    "reason": r.reason,
                    "confidence": round(r.confidence, 2),
                }
                for r in reviewers
            ]

        return result

    elif name == MCPToolName.SCAN_SECRETS.value:
        from .services.security_scanner import Severity as ScanSeverity

        min_sev_str = arguments.get("min_severity", "low")
        try:
            min_sev = ScanSeverity(min_sev_str)
        except ValueError:
            min_sev = ScanSeverity.LOW

        scanner = SecurityScanner(min_severity=min_sev)

        repo_path = arguments.get("repo_path")
        repo_filter = arguments.get("repo_filter")

        if repo_path:
            scan_result = scanner.scan_repository(repo_path)
        elif repo_filter:
            scan_result = scanner.scan_chunks(repo_filter)
        else:
            scan_result = scanner.scan_chunks()

        return {
            "total_files_scanned": scan_result.total_files_scanned,
            "total_findings": scan_result.total_findings,
            "findings_by_severity": scan_result.findings_by_severity,
            "scan_duration_ms": scan_result.scan_duration_ms,
            "findings": [
                {
                    "type": f.secret_type.value,
                    "severity": f.severity.value,
                    "file": f.file_path,
                    "line": f.line_number,
                    "description": f.description,
                    "matched": f.matched_text,
                    "recommendation": f.recommendation,
                    "confidence": round(f.confidence, 2),
                }
                for f in scan_result.findings[:50]  # Limit output
            ],
        }

    elif name == MCPToolName.GET_METRICS.value:
        metrics_service = MetricsService()

        symbol_name = arguments.get("symbol_name")
        repo_filter = arguments.get("repo_filter")

        if symbol_name:
            # Analyze a specific function
            from .services.storage import StorageService
            from .models.chunk import ChunkType

            storage = StorageService()
            chunk_map = storage._load_chunk_metadata()

            for chunk_id, chunk in chunk_map.items():
                if repo_filter and chunk.repo_name != repo_filter:
                    continue
                if (chunk.get_qualified_name() == symbol_name or chunk.name == symbol_name):
                    if chunk.chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.CONSTRUCTOR):
                        fm = metrics_service.analyze_function(
                            code=chunk.content,
                            language=chunk.language,
                            name=chunk.name,
                            qualified_name=chunk.get_qualified_name(),
                            file_path=chunk.file_path,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                        )
                        return {
                            "name": fm.qualified_name,
                            "file": f"{fm.file_path}:{fm.start_line}-{fm.end_line}",
                            "cyclomatic_complexity": fm.cyclomatic.complexity,
                            "cyclomatic_rating": fm.cyclomatic.rating.value,
                            "cognitive_complexity": fm.cognitive.complexity,
                            "cognitive_rating": fm.cognitive.rating.value,
                            "source_lines": fm.loc.source_lines,
                            "parameter_count": fm.parameter_count,
                            "return_count": fm.return_count,
                            "is_complex": fm.is_complex,
                        }
            return {"error": f"Function not found: {symbol_name}"}

        else:
            # Analyze entire repository
            repo_metrics = metrics_service.analyze_repository(repo_filter)
            return {
                "repo": repo_metrics.repo_name,
                "total_files": repo_metrics.total_files,
                "total_functions": repo_metrics.total_functions,
                "total_classes": repo_metrics.total_classes,
                "lines_of_code": {
                    "total": repo_metrics.loc.total_lines,
                    "source": repo_metrics.loc.source_lines,
                    "comment": repo_metrics.loc.comment_lines,
                    "comment_ratio": repo_metrics.loc.comment_ratio,
                },
                "complexity": {
                    "avg_cyclomatic": repo_metrics.avg_cyclomatic,
                    "avg_cognitive": repo_metrics.avg_cognitive,
                    "complex_functions": repo_metrics.complex_function_count,
                    "complex_percentage": repo_metrics.complex_function_percentage,
                },
                "maintainability_index": repo_metrics.maintainability_index,
                "hotspots": repo_metrics.hotspots[:10],
                "language_breakdown": repo_metrics.language_breakdown,
            }

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
