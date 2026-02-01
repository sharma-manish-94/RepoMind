"""
MCP tools for RepoMind.

Provides all MCP-exposed tools organized by category:

**Indexing**: index_repo
**Semantic Search**: semantic_grep (with detail_level/token_budget), get_context
**Code Navigation**: lookup_symbol, find_callers, find_callees, find_call_path,
    get_symbol_context, get_class_members, get_index_stats
**Code Analysis**: file_summary, find_usages, find_tests, find_implementations,
    find_hierarchy, diff_impact
**Compound Operations** (token-efficient): explore, understand, prepare_change
**Pattern Analysis**: analyze_patterns, get_coding_conventions
**Diagram Generation**: generate_diagram, generate_architecture_diagram,
    generate_dataflow_diagram, generate_callflow_diagram, generate_class_diagram,
    generate_sequence_diagram, generate_dependency_diagram
"""

from .code_nav import (
    find_call_path,
    find_callers,
    find_callees,
    get_class_members,
    get_index_stats,
    get_symbol_context,
    lookup_symbol,
)
from .compound_ops import explore, prepare_change, understand
from .diff_impact import diff_impact
from .file_summary import file_summary
from .find_implementations import find_hierarchy, find_implementations
from .find_tests import find_tests
from .find_usages import find_usages
from .get_context import get_context
from .index_repo import index_repo
from .semantic_grep import semantic_grep
from .analyze_patterns import analyze_patterns, get_coding_conventions
from .generate_diagrams import (
    generate_diagram,
    generate_architecture_diagram,
    generate_dataflow_diagram,
    generate_callflow_diagram,
    generate_class_diagram,
    generate_sequence_diagram,
    generate_dependency_diagram,
)

__all__ = [
    # Indexing
    "index_repo",
    # Semantic search
    "semantic_grep",
    "get_context",
    # Code navigation
    "lookup_symbol",
    "find_callers",
    "find_callees",
    "find_call_path",
    "get_symbol_context",
    "get_class_members",
    "get_index_stats",
    # Code analysis tools
    "file_summary",
    "find_usages",
    "find_tests",
    "find_implementations",
    "find_hierarchy",
    "diff_impact",
    # Compound operations (token-efficient)
    "explore",
    "understand",
    "prepare_change",
    # Pattern analysis
    "analyze_patterns",
    "get_coding_conventions",
    # Diagram generation
    "generate_diagram",
    "generate_architecture_diagram",
    "generate_dataflow_diagram",
    "generate_callflow_diagram",
    "generate_class_diagram",
    "generate_sequence_diagram",
    "generate_dependency_diagram",
]
