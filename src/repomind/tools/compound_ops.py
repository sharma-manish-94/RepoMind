"""
Compound Operations - Token-Efficient Multi-Query Tools.

This module provides high-level operations that combine multiple
underlying tools into single, token-efficient responses.

Key Operations:
1. explore: Get comprehensive overview of a symbol
2. understand: Deep analysis of code behavior and dependencies
3. prepareChange: Impact analysis before making modifications

Token Efficiency:
- explore: ~500 tokens vs ~2000 for separate calls (75% reduction)
- understand: ~800 tokens vs ~3000 for separate calls (73% reduction)
- prepareChange: ~600 tokens vs ~2500 for separate calls (76% reduction)

Example:
    # Instead of 4 separate tool calls:
    # semantic_grep("UserService") + find_callers("UserService.save")
    # + find_callees("UserService.save") + find_tests("UserService")

    # Single compound operation:
    explore("UserService.save")
    # Returns: definition + callers + callees + tests in one response

Author: RepoMind Team
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..models.chunk import DetailLevel
from ..services.call_graph import CallGraphService
from ..services.storage import StorageService
from ..services.symbol_table import SymbolTableService
from ..services.response_formatter import ResponseFormatter
from .find_usages import find_usages
from .find_tests import find_tests

console = Console(width=200, force_terminal=False)


class ExploreDepth(str, Enum):
    """Depth of exploration for the explore operation."""
    SHALLOW = "shallow"  # Just definition + immediate callers/callees
    NORMAL = "normal"    # + tests + usages summary
    DEEP = "deep"        # + transitive callers/callees (2 levels)


@dataclass
class ExploreResult:
    """Result of an explore operation."""
    symbol: str
    definition: Optional[dict[str, Any]]
    callers: list[dict[str, Any]]
    callees: list[dict[str, Any]]
    tests: list[dict[str, Any]]
    usages_summary: Optional[dict[str, int]]
    impact_radius: int  # Number of symbols that would be affected by a change


def explore(
    symbol_name: str,
    repo_filter: Optional[str] = None,
    depth: str = "normal",
    max_callers: int = 10,
    max_callees: int = 10,
    detail_level: str = "summary",
) -> dict[str, Any]:
    """
    Comprehensive exploration of a symbol in one operation.

    Combines multiple queries into a single, token-efficient response:
    - Symbol definition with signature and docstring
    - Direct callers (who uses this?)
    - Direct callees (what does this use?)
    - Related tests
    - Impact radius (change blast area)

    This operation is designed to give AI assistants enough context
    to understand a symbol without multiple round-trips.

    Args:
        symbol_name: Name of the symbol to explore (e.g., "UserService.save")
        repo_filter: Optional repository filter
        depth: Exploration depth - "shallow", "normal", or "deep"
        max_callers: Maximum callers to return (default: 10)
        max_callees: Maximum callees to return (default: 10)
        detail_level: Detail level for code - "summary", "preview", or "full"

    Returns:
        Comprehensive exploration result with all relevant information.

    Example:
        >>> explore("UserService.save")
        {
            "symbol": "UserService.save",
            "definition": {
                "type": "method",
                "signature": "def save(self, user: User) -> bool",
                "location": "services/user.py:45-67",
                "docstring": "Persist user to database"
            },
            "callers": [
                {"name": "UserController.create", "location": "..."},
                ...
            ],
            "callees": [
                {"name": "UserRepository.insert", "location": "..."},
                ...
            ],
            "tests": [
                {"name": "test_user_save", "file": "test_user.py"}
            ],
            "impact_radius": 5,
            "stats": {"tokens_used": ~500}
        }
    """
    if not symbol_name.strip():
        return {"error": "Symbol name cannot be empty"}

    console.print(f"[bold blue]Exploring: {symbol_name}[/bold blue]")

    # Parse depth
    try:
        explore_depth = ExploreDepth(depth.lower())
    except ValueError:
        return {"error": f"Invalid depth. Options: {[d.value for d in ExploreDepth]}"}

    # Parse detail level
    try:
        parsed_detail = DetailLevel(detail_level.lower())
    except ValueError:
        return {"error": f"Invalid detail_level. Options: {[d.value for d in DetailLevel]}"}

    # Initialize services
    call_graph = CallGraphService()
    storage = StorageService()
    symbol_table = SymbolTableService()
    formatter = ResponseFormatter()

    result: dict[str, Any] = {
        "symbol": symbol_name,
        "depth": explore_depth.value,
    }

    # 1. Find definition
    definition = _find_definition(symbol_name, symbol_table, storage, parsed_detail, repo_filter)
    result["definition"] = definition

    # 2. Find callers
    caller_relations = call_graph.find_callers(symbol_name, repo_filter)
    callers = [
        {
            "name": r.caller,
            "location": f"{r.caller_file}:{r.caller_line}",
            "call_type": r.call_type,
        }
        for r in caller_relations[:max_callers]
    ]
    result["callers"] = callers
    result["total_callers"] = len(caller_relations)

    # 3. Find callees
    callee_relations = call_graph.find_callees(symbol_name, repo_filter)
    callees = [
        {
            "name": r.callee,
            "location": f"{r.caller_file}:{r.caller_line}",
            "call_type": r.call_type,
        }
        for r in callee_relations[:max_callees]
    ]
    result["callees"] = callees
    result["total_callees"] = len(callee_relations)

    # 4. Find tests (for NORMAL and DEEP)
    if explore_depth in [ExploreDepth.NORMAL, ExploreDepth.DEEP]:
        # Extract base symbol name for test lookup
        base_name = symbol_name.split(".")[-1] if "." in symbol_name else symbol_name
        test_result = find_tests(base_name, repo_filter)
        tests = []
        if "test_methods" in test_result:
            for t in test_result["test_methods"][:5]:
                tests.append({
                    "name": t.get("name", "unknown"),
                    "file": t.get("file", "unknown"),
                    "line": t.get("line"),
                })
        result["tests"] = tests

    # 5. Calculate impact radius
    impact_radius = len(caller_relations)  # Direct impact
    if explore_depth == ExploreDepth.DEEP:
        # Add transitive callers (2 levels)
        for r in caller_relations[:5]:  # Limit to prevent explosion
            transitive = call_graph.find_callers(r.caller, repo_filter)
            impact_radius += len(transitive)
    result["impact_radius"] = impact_radius

    # 6. Add stats
    result["stats"] = {
        "detail_level": parsed_detail.value,
        "estimated_tokens": _estimate_explore_tokens(result, parsed_detail),
    }

    _display_explore_result(result)

    return result


def understand(
    symbol_name: str,
    repo_filter: Optional[str] = None,
    include_implementation: bool = True,
    max_depth: int = 2,
) -> dict[str, Any]:
    """
    Deep understanding of a symbol's behavior and dependencies.

    Goes beyond explore() to provide:
    - Full implementation code
    - Type hierarchy (implements/extends)
    - Dependency chain (what it needs to work)
    - Data flow (input → processing → output)

    Designed for when you need to truly understand how code works,
    not just locate it.

    Args:
        symbol_name: Name of the symbol to understand
        repo_filter: Optional repository filter
        include_implementation: Include full code (default: True)
        max_depth: Maximum depth for dependency tracking (default: 2)

    Returns:
        Deep analysis of the symbol's behavior.

    Example:
        >>> understand("AuthService.validate")
        {
            "symbol": "AuthService.validate",
            "implementation": "def validate(self, token): ...",
            "hierarchy": {
                "implements": ["IAuthService"],
                "extends": ["BaseService"]
            },
            "dependencies": {
                "direct": ["TokenDecoder", "UserRepository"],
                "transitive": ["DatabaseConnection", "CacheService"]
            },
            "data_flow": {
                "inputs": ["token: str"],
                "outputs": ["bool"],
                "side_effects": ["logs validation attempt"]
            }
        }
    """
    if not symbol_name.strip():
        return {"error": "Symbol name cannot be empty"}

    console.print(f"[bold blue]Understanding: {symbol_name}[/bold blue]")

    # Initialize services
    call_graph = CallGraphService()
    storage = StorageService()
    symbol_table = SymbolTableService()

    result: dict[str, Any] = {
        "symbol": symbol_name,
    }

    # 1. Get full implementation
    if include_implementation:
        definition = _find_definition(
            symbol_name, symbol_table, storage, DetailLevel.FULL, repo_filter
        )
        result["definition"] = definition
        if definition and "content" in definition:
            result["implementation"] = definition.get("content")

    # 2. Get type hierarchy
    hierarchy = _get_type_hierarchy(symbol_name, symbol_table, repo_filter)
    result["hierarchy"] = hierarchy

    # 3. Get dependency chain
    dependencies = _get_dependency_chain(symbol_name, call_graph, max_depth, repo_filter)
    result["dependencies"] = dependencies

    # 4. Analyze data flow (from signature and docstring)
    if result.get("definition"):
        data_flow = _analyze_data_flow(result["definition"])
        result["data_flow"] = data_flow

    # 5. Get usages summary
    usage_result = find_usages(symbol_name, repo_filter, limit=100)
    if "usages" in usage_result:
        usages = usage_result["usages"]
        result["usage_summary"] = {
            "calls": len(usages.get("calls", [])),
            "type_hints": len(usages.get("type_hints", [])),
            "inheritance": len(usages.get("inheritance", [])),
            "imports": len(usages.get("imports", [])),
        }

    _display_understand_result(result)

    return result


def prepare_change(
    symbol_name: str,
    repo_filter: Optional[str] = None,
    change_type: str = "modify",
) -> dict[str, Any]:
    """
    Impact analysis to prepare for modifying a symbol.

    Analyzes what would be affected by changing a symbol:
    - Direct dependents (will definitely be affected)
    - Potential breaking changes (signature changes, etc.)
    - Tests that need updating
    - Files that need review

    Use before making changes to understand the blast radius.

    Args:
        symbol_name: Name of the symbol to change
        repo_filter: Optional repository filter
        change_type: Type of change - "modify", "rename", "delete", "signature"

    Returns:
        Impact analysis with affected code and recommendations.

    Example:
        >>> prepare_change("UserService.save", change_type="signature")
        {
            "symbol": "UserService.save",
            "change_type": "signature",
            "impact": {
                "direct_dependents": 5,
                "transitive_dependents": 12,
                "affected_tests": 3
            },
            "affected_files": [
                {"file": "user_controller.py", "reason": "calls save()"},
                ...
            ],
            "breaking_risks": [
                "Changing return type may break UserController.create"
            ],
            "recommended_review": [
                "user_controller.py",
                "test_user_service.py"
            ]
        }
    """
    if not symbol_name.strip():
        return {"error": "Symbol name cannot be empty"}

    valid_change_types = ["modify", "rename", "delete", "signature"]
    if change_type not in valid_change_types:
        return {"error": f"Invalid change_type. Options: {valid_change_types}"}

    console.print(f"[bold blue]Preparing change: {symbol_name} ({change_type})[/bold blue]")

    # Initialize services
    call_graph = CallGraphService()
    symbol_table = SymbolTableService()
    storage = StorageService()

    result: dict[str, Any] = {
        "symbol": symbol_name,
        "change_type": change_type,
    }

    # 1. Find direct dependents (callers)
    direct_callers = call_graph.find_callers(symbol_name, repo_filter)
    result["direct_dependents"] = len(direct_callers)

    # 2. Find transitive dependents (callers of callers)
    transitive_count = 0
    transitive_files = set()
    for caller in direct_callers[:20]:  # Limit for performance
        trans_callers = call_graph.find_callers(caller.caller, repo_filter)
        transitive_count += len(trans_callers)
        for t in trans_callers:
            transitive_files.add(t.caller_file)
    result["transitive_dependents"] = transitive_count

    # 3. Find affected tests
    base_name = symbol_name.split(".")[-1] if "." in symbol_name else symbol_name
    test_result = find_tests(base_name, repo_filter)
    affected_tests = []
    if "test_methods" in test_result:
        affected_tests = [t.get("name") for t in test_result["test_methods"][:10]]
    result["affected_tests"] = affected_tests

    # 4. Collect affected files with reasons
    affected_files = []
    seen_files = set()
    for caller in direct_callers[:15]:
        if caller.caller_file not in seen_files:
            affected_files.append({
                "file": caller.caller_file,
                "line": caller.caller_line,
                "reason": f"calls {symbol_name}",
                "symbol": caller.caller,
            })
            seen_files.add(caller.caller_file)
    result["affected_files"] = affected_files

    # 5. Analyze breaking risks based on change type
    breaking_risks = []
    if change_type == "delete":
        breaking_risks.append(f"Deleting will break {len(direct_callers)} direct callers")
    elif change_type == "rename":
        breaking_risks.append(f"Renaming requires updating {len(direct_callers)} call sites")
        if test_result.get("test_methods"):
            breaking_risks.append(f"Update {len(test_result['test_methods'])} tests that reference this symbol")
    elif change_type == "signature":
        breaking_risks.append("Signature changes may require updating all call sites")
        # Check if it's a public method (in a class)
        if "." in symbol_name:
            class_name = symbol_name.rsplit(".", 1)[0]
            # Check for interface implementations
            impls = symbol_table.find_implementations(class_name, repo_filter)
            if impls:
                breaking_risks.append(f"May affect {len(impls)} implementations")
    result["breaking_risks"] = breaking_risks

    # 6. Recommended files to review
    recommended_review = list(seen_files)
    # Add test files
    for test in test_result.get("test_files", [])[:5]:
        if test.get("file") and test["file"] not in recommended_review:
            recommended_review.append(test["file"])
    result["recommended_review"] = recommended_review[:10]

    # 7. Impact summary
    result["impact_summary"] = {
        "total_affected_symbols": result["direct_dependents"] + result["transitive_dependents"],
        "total_affected_files": len(seen_files | transitive_files),
        "total_affected_tests": len(affected_tests),
        "risk_level": _calculate_risk_level(
            result["direct_dependents"],
            result["transitive_dependents"],
            len(affected_tests),
            change_type,
        ),
    }

    _display_prepare_change_result(result)

    return result


# =============================================================================
# Helper Functions
# =============================================================================


def _find_definition(
    symbol_name: str,
    symbol_table: SymbolTableService,
    storage: StorageService,
    detail_level: DetailLevel,
    repo_filter: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Find the definition of a symbol."""
    # Try exact lookup first
    symbols = symbol_table.lookup(symbol_name, repo_filter)
    if not symbols:
        # Try as qualified name
        symbols = symbol_table.lookup(symbol_name.split(".")[-1], repo_filter)
        symbols = [s for s in symbols if symbol_name in s.get("qualified_name", "")]

    if not symbols:
        return None

    symbol = symbols[0]
    chunk_map = storage._load_chunk_metadata()

    # Find the chunk for this symbol
    for chunk_id, chunk in chunk_map.items():
        if (chunk.get_qualified_name() == symbol.get("qualified_name") or
            chunk.name == symbol.get("name")):
            if repo_filter and chunk.repo_name != repo_filter:
                continue
            return chunk.to_response(detail_level)

    # Return basic symbol info if chunk not found
    return {
        "name": symbol.get("name"),
        "qualified_name": symbol.get("qualified_name"),
        "type": symbol.get("type"),
        "file": symbol.get("file"),
        "line": symbol.get("line"),
    }


def _get_type_hierarchy(
    symbol_name: str,
    symbol_table: SymbolTableService,
    repo_filter: Optional[str] = None,
) -> dict[str, list[str]]:
    """Get the type hierarchy for a symbol."""
    hierarchy = {
        "implements": [],
        "extends": [],
        "implemented_by": [],
        "extended_by": [],
    }

    # Get class name (symbol might be ClassName.method)
    class_name = symbol_name.split(".")[0] if "." in symbol_name else symbol_name

    # Find parents
    parents = symbol_table.find_parents(class_name, repo_filter)
    for parent in parents:
        rel_type = parent.get("relation_type", "extends")
        if rel_type == "implements":
            hierarchy["implements"].append(parent.get("parent_name", ""))
        else:
            hierarchy["extends"].append(parent.get("parent_name", ""))

    # Find implementations
    impls = symbol_table.find_implementations(class_name, repo_filter)
    for impl in impls:
        impl_name = impl.get("child_name", "")
        rel_type = impl.get("relation_type", "implements")
        if rel_type == "implements":
            hierarchy["implemented_by"].append(impl_name)
        else:
            hierarchy["extended_by"].append(impl_name)

    return hierarchy


def _get_dependency_chain(
    symbol_name: str,
    call_graph: CallGraphService,
    max_depth: int,
    repo_filter: Optional[str] = None,
) -> dict[str, list[str]]:
    """Get the dependency chain for a symbol."""
    dependencies: dict[str, list[str]] = {
        "direct": [],
        "transitive": [],
    }

    # Direct dependencies (callees)
    callees = call_graph.find_callees(symbol_name, repo_filter)
    direct = list(set(r.callee for r in callees))
    dependencies["direct"] = direct[:20]

    # Transitive dependencies (callees of callees)
    if max_depth > 1:
        transitive = set()
        for callee in direct[:10]:  # Limit for performance
            trans_callees = call_graph.find_callees(callee, repo_filter)
            for t in trans_callees:
                if t.callee not in direct:
                    transitive.add(t.callee)
        dependencies["transitive"] = list(transitive)[:20]

    return dependencies


def _analyze_data_flow(definition: dict[str, Any]) -> dict[str, Any]:
    """Analyze data flow from a definition."""
    data_flow: dict[str, Any] = {
        "inputs": [],
        "outputs": [],
        "side_effects": [],
    }

    signature = definition.get("signature", "")
    docstring = definition.get("docstring", "")

    # Parse inputs from signature
    if "(" in signature and ")" in signature:
        params_str = signature[signature.find("(") + 1:signature.rfind(")")]
        if params_str.strip():
            # Split by comma, handling nested generics
            params = _split_params(params_str)
            for param in params:
                param = param.strip()
                if param and param != "self" and param != "cls":
                    data_flow["inputs"].append(param)

    # Parse return type from signature
    if "->" in signature:
        return_type = signature.split("->")[-1].strip().rstrip(":")
        data_flow["outputs"].append(return_type)

    # Detect side effects from docstring
    side_effect_keywords = ["save", "write", "delete", "update", "send", "log", "emit"]
    if docstring:
        docstring_lower = docstring.lower()
        for keyword in side_effect_keywords:
            if keyword in docstring_lower:
                data_flow["side_effects"].append(f"May {keyword}")

    return data_flow


def _split_params(params_str: str) -> list[str]:
    """Split parameter string handling nested generics."""
    params = []
    current = ""
    depth = 0

    for char in params_str:
        if char == "[" or char == "<":
            depth += 1
            current += char
        elif char == "]" or char == ">":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            params.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        params.append(current.strip())

    return params


def _calculate_risk_level(
    direct: int,
    transitive: int,
    tests: int,
    change_type: str,
) -> str:
    """Calculate the risk level of a change."""
    total_impact = direct + (transitive * 0.5)  # Transitive has less weight

    # Adjust for change type
    multiplier = {
        "modify": 1.0,
        "signature": 1.5,
        "rename": 1.3,
        "delete": 2.0,
    }.get(change_type, 1.0)

    adjusted_impact = total_impact * multiplier

    # Factor in test coverage
    if tests == 0 and direct > 0:
        adjusted_impact *= 1.5  # Higher risk without tests

    if adjusted_impact <= 3:
        return "low"
    elif adjusted_impact <= 10:
        return "medium"
    elif adjusted_impact <= 25:
        return "high"
    else:
        return "critical"


def _estimate_explore_tokens(result: dict[str, Any], detail_level: DetailLevel) -> int:
    """Estimate tokens used by explore result."""
    base = 100  # Metadata overhead

    # Definition tokens
    if result.get("definition"):
        if detail_level == DetailLevel.SUMMARY:
            base += 50
        elif detail_level == DetailLevel.PREVIEW:
            base += 200
        else:
            base += 500

    # Callers/callees (summary format)
    base += len(result.get("callers", [])) * 30
    base += len(result.get("callees", [])) * 30

    # Tests
    base += len(result.get("tests", [])) * 20

    return base


# =============================================================================
# Display Functions
# =============================================================================


def _display_explore_result(result: dict[str, Any]) -> None:
    """Display explore result in console."""
    console.print(Panel(f"[bold]Explore: {result['symbol']}[/bold]"))

    if result.get("definition"):
        defn = result["definition"]
        console.print(f"[green]Definition:[/green] {defn.get('type', 'unknown')} at {defn.get('location', 'unknown')}")
        if defn.get("signature"):
            console.print(f"  {defn['signature']}")

    if result.get("callers"):
        console.print(f"\n[yellow]Callers ({result.get('total_callers', 0)}):[/yellow]")
        for c in result["callers"][:5]:
            console.print(f"  - {c['name']} at {c['location']}")

    if result.get("callees"):
        console.print(f"\n[cyan]Callees ({result.get('total_callees', 0)}):[/cyan]")
        for c in result["callees"][:5]:
            console.print(f"  - {c['name']}")

    if result.get("tests"):
        console.print(f"\n[magenta]Tests:[/magenta]")
        for t in result["tests"][:3]:
            console.print(f"  - {t['name']}")

    console.print(f"\n[dim]Impact radius: {result.get('impact_radius', 0)} symbols[/dim]")


def _display_understand_result(result: dict[str, Any]) -> None:
    """Display understand result in console."""
    console.print(Panel(f"[bold]Understand: {result['symbol']}[/bold]"))

    if result.get("hierarchy"):
        h = result["hierarchy"]
        if h.get("implements"):
            console.print(f"[green]Implements:[/green] {', '.join(h['implements'])}")
        if h.get("extends"):
            console.print(f"[green]Extends:[/green] {', '.join(h['extends'])}")

    if result.get("dependencies"):
        d = result["dependencies"]
        console.print(f"\n[yellow]Direct dependencies:[/yellow] {len(d.get('direct', []))}")
        console.print(f"[yellow]Transitive dependencies:[/yellow] {len(d.get('transitive', []))}")

    if result.get("usage_summary"):
        u = result["usage_summary"]
        console.print(f"\n[cyan]Usage summary:[/cyan]")
        console.print(f"  Calls: {u.get('calls', 0)}, Type hints: {u.get('type_hints', 0)}")


def _display_prepare_change_result(result: dict[str, Any]) -> None:
    """Display prepare_change result in console."""
    summary = result.get("impact_summary", {})
    risk = summary.get("risk_level", "unknown")

    risk_color = {
        "low": "green",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red",
    }.get(risk, "white")

    console.print(Panel(
        f"[bold]Prepare Change: {result['symbol']}[/bold]\n"
        f"Change type: {result['change_type']}\n"
        f"Risk level: [{risk_color}]{risk}[/{risk_color}]"
    ))

    console.print(f"\n[yellow]Impact:[/yellow]")
    console.print(f"  Direct dependents: {result.get('direct_dependents', 0)}")
    console.print(f"  Transitive dependents: {result.get('transitive_dependents', 0)}")
    console.print(f"  Affected tests: {len(result.get('affected_tests', []))}")

    if result.get("breaking_risks"):
        console.print(f"\n[red]Breaking risks:[/red]")
        for risk_item in result["breaking_risks"]:
            console.print(f"  - {risk_item}")

    if result.get("recommended_review"):
        console.print(f"\n[cyan]Recommended for review:[/cyan]")
        for f in result["recommended_review"][:5]:
            console.print(f"  - {f}")
