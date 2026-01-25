"""Call Graph service for tracking function/method relationships.

Provides:
- Who calls this function? (find callers / impact analysis)
- What does this function call? (find callees / dependencies)
- Trace execution paths
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import get_config
from .symbol_table import SymbolTableService


@dataclass
class CallRelation:
    """A call relationship between two symbols."""

    caller: str  # Qualified name of the caller
    callee: str  # Qualified name of the callee
    caller_file: str
    caller_line: int
    repo_name: str
    call_type: str = "direct"  # direct, virtual, callback, etc.


@dataclass
class CallNode:
    """A node in the call graph with its relationships."""

    name: str
    callers: list[str]  # Who calls this
    callees: list[str]  # What this calls
    file_path: Optional[str] = None
    repo_name: Optional[str] = None


class CallGraphService:
    """SQLite-based call graph for relationship queries.

    Stores directed edges: caller -> callee
    Supports:
    - Find all callers of a function (impact analysis)
    - Find all callees of a function (dependencies)
    - Trace paths between functions
    - Detect cycles
    """

    def __init__(self):
        config = get_config()
        self.db_path = config.index.metadata_dir / "callgraph.db"
        config.index.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    caller TEXT NOT NULL,
                    callee TEXT NOT NULL,
                    caller_file TEXT NOT NULL,
                    caller_line INTEGER NOT NULL,
                    repo_name TEXT NOT NULL,
                    call_type TEXT DEFAULT 'direct',
                    UNIQUE(caller, callee, caller_file, caller_line)
                );

                -- Indices for fast lookups in both directions
                CREATE INDEX IF NOT EXISTS idx_caller ON calls(caller);
                CREATE INDEX IF NOT EXISTS idx_callee ON calls(callee);
                CREATE INDEX IF NOT EXISTS idx_repo ON calls(repo_name);
                CREATE INDEX IF NOT EXISTS idx_file ON calls(caller_file);

                -- Table for Spring DI and class dependency mappings
                CREATE TABLE IF NOT EXISTS class_dependencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_name TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    field_type TEXT NOT NULL,
                    injection_type TEXT DEFAULT 'field',
                    repo_name TEXT NOT NULL,
                    UNIQUE(class_name, field_name, repo_name)
                );

                CREATE INDEX IF NOT EXISTS idx_class_deps ON class_dependencies(class_name, repo_name);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add_call(self, relation: CallRelation) -> int:
        """Add a call relationship.

        Returns:
            The row ID of the inserted relation
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO calls
                (caller, callee, caller_file, caller_line, repo_name, call_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    relation.caller,
                    relation.callee,
                    relation.caller_file,
                    relation.caller_line,
                    relation.repo_name,
                    relation.call_type,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def add_calls_bulk(self, relations: list[CallRelation]) -> int:
        """Bulk add call relationships.

        Returns:
            Number of relations added
        """
        if not relations:
            return 0

        conn = self._get_connection()
        try:
            data = [
                (r.caller, r.callee, r.caller_file, r.caller_line, r.repo_name, r.call_type)
                for r in relations
            ]
            conn.executemany(
                """
                INSERT OR IGNORE INTO calls
                (caller, callee, caller_file, caller_line, repo_name, call_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                data,
            )
            conn.commit()
            return len(data)
        finally:
            conn.close()

    def find_callers(
        self,
        symbol: str,
        repo_name: Optional[str] = None,
        max_depth: int = 1,
    ) -> list[CallRelation]:
        """Find all functions that call the given symbol.

        Args:
            symbol: Qualified name of the function
            repo_name: Optional repository filter
            max_depth: How many levels up to traverse (1 = direct callers only)

        Returns:
            List of call relations where this symbol is the callee
        """
        conn = self._get_connection()
        try:
            if max_depth == 1:
                # Simple direct callers query
                if repo_name:
                    cursor = conn.execute(
                        """
                        SELECT * FROM calls
                        WHERE callee = ? AND repo_name = ?
                        ORDER BY caller
                        """,
                        (symbol, repo_name),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM calls
                        WHERE callee = ?
                        ORDER BY caller
                        """,
                        (symbol,),
                    )
                return [self._row_to_relation(row) for row in cursor.fetchall()]
            else:
                # Recursive query for transitive callers
                return self._find_transitive_callers(conn, symbol, repo_name, max_depth)
        finally:
            conn.close()

    def find_callees(
        self,
        symbol: str,
        repo_name: Optional[str] = None,
        max_depth: int = 1,
    ) -> list[CallRelation]:
        """Find all functions called by the given symbol.

        Args:
            symbol: Qualified name of the function
            repo_name: Optional repository filter
            max_depth: How many levels down to traverse (1 = direct callees only)

        Returns:
            List of call relations where this symbol is the caller
        """
        conn = self._get_connection()
        try:
            if max_depth == 1:
                if repo_name:
                    cursor = conn.execute(
                        """
                        SELECT * FROM calls
                        WHERE caller = ? AND repo_name = ?
                        ORDER BY callee
                        """,
                        (symbol, repo_name),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM calls
                        WHERE caller = ?
                        ORDER BY callee
                        """,
                        (symbol,),
                    )
                return [self._row_to_relation(row) for row in cursor.fetchall()]
            else:
                return self._find_transitive_callees(conn, symbol, repo_name, max_depth)
        finally:
            conn.close()

    def _find_transitive_callers(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        repo_name: Optional[str],
        max_depth: int,
    ) -> list[CallRelation]:
        """Find transitive callers using BFS."""
        visited = set()
        result = []
        queue = [(symbol, 0)]  # (symbol, depth)

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            if repo_name:
                cursor = conn.execute(
                    "SELECT * FROM calls WHERE callee = ? AND repo_name = ?",
                    (current, repo_name),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM calls WHERE callee = ?",
                    (current,),
                )

            for row in cursor.fetchall():
                relation = self._row_to_relation(row)
                if relation.caller not in visited:
                    visited.add(relation.caller)
                    result.append(relation)
                    queue.append((relation.caller, depth + 1))

        return result

    def _find_transitive_callees(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        repo_name: Optional[str],
        max_depth: int,
    ) -> list[CallRelation]:
        """Find transitive callees using BFS."""
        visited = set()
        result = []
        queue = [(symbol, 0)]

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            if repo_name:
                cursor = conn.execute(
                    "SELECT * FROM calls WHERE caller = ? AND repo_name = ?",
                    (current, repo_name),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM calls WHERE caller = ?",
                    (current,),
                )

            for row in cursor.fetchall():
                relation = self._row_to_relation(row)
                if relation.callee not in visited:
                    visited.add(relation.callee)
                    result.append(relation)
                    queue.append((relation.callee, depth + 1))

        return result

    def get_call_node(self, symbol: str, repo_name: Optional[str] = None) -> CallNode:
        """Get complete call information for a symbol.

        Returns:
            CallNode with both callers and callees
        """
        callers = self.find_callers(symbol, repo_name)
        callees = self.find_callees(symbol, repo_name)

        file_path = None
        if callers:
            # Use the first caller's file as reference
            file_path = callers[0].caller_file
        elif callees:
            file_path = callees[0].caller_file

        return CallNode(
            name=symbol,
            callers=[r.caller for r in callers],
            callees=[r.callee for r in callees],
            file_path=file_path,
            repo_name=repo_name,
        )

    def find_path(
        self,
        from_symbol: str,
        to_symbol: str,
        repo_name: Optional[str] = None,
        max_depth: int = 10,
    ) -> Optional[list[str]]:
        """Find a call path from one symbol to another.

        Args:
            from_symbol: Starting function
            to_symbol: Target function
            repo_name: Optional repository filter
            max_depth: Maximum path length

        Returns:
            List of symbols in the path, or None if no path exists
        """
        conn = self._get_connection()
        try:
            visited = {from_symbol}
            queue = [(from_symbol, [from_symbol])]

            while queue:
                current, path = queue.pop(0)

                if len(path) > max_depth:
                    continue

                if repo_name:
                    cursor = conn.execute(
                        "SELECT DISTINCT callee FROM calls WHERE caller = ? AND repo_name = ?",
                        (current, repo_name),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT DISTINCT callee FROM calls WHERE caller = ?",
                        (current,),
                    )

                for row in cursor.fetchall():
                    callee = row["callee"]
                    if callee == to_symbol:
                        return path + [callee]

                    if callee not in visited:
                        visited.add(callee)
                        queue.append((callee, path + [callee]))

            return None
        finally:
            conn.close()

    def delete_repo(self, repo_name: str) -> int:
        """Delete all call relations for a repository.

        Returns:
            Number of relations deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM calls WHERE repo_name = ?",
                (repo_name,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def delete_file(self, repo_name: str, file_path: str) -> int:
        """Delete all call relations from a specific file.

        Returns:
            Number of relations deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM calls WHERE repo_name = ? AND caller_file = ?",
                (repo_name, file_path),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Get statistics about the call graph."""
        conn = self._get_connection()
        try:
            stats = {}

            # Total edges
            cursor = conn.execute("SELECT COUNT(*) FROM calls")
            stats["total_edges"] = cursor.fetchone()[0]

            # Unique callers
            cursor = conn.execute("SELECT COUNT(DISTINCT caller) FROM calls")
            stats["unique_callers"] = cursor.fetchone()[0]

            # Unique callees
            cursor = conn.execute("SELECT COUNT(DISTINCT callee) FROM calls")
            stats["unique_callees"] = cursor.fetchone()[0]

            # By repository
            cursor = conn.execute(
                """
                SELECT repo_name, COUNT(*) as count
                FROM calls
                GROUP BY repo_name
                ORDER BY count DESC
                """
            )
            stats["by_repo"] = {row["repo_name"]: row["count"] for row in cursor.fetchall()}

            # Most called (hotspots)
            cursor = conn.execute(
                """
                SELECT callee, COUNT(*) as call_count
                FROM calls
                GROUP BY callee
                ORDER BY call_count DESC
                LIMIT 10
                """
            )
            stats["hotspots"] = [(row["callee"], row["call_count"]) for row in cursor.fetchall()]

            return stats
        finally:
            conn.close()

    def _row_to_relation(self, row: sqlite3.Row) -> CallRelation:
        """Convert a database row to a CallRelation object."""
        return CallRelation(
            caller=row["caller"],
            callee=row["callee"],
            caller_file=row["caller_file"],
            caller_line=row["caller_line"],
            repo_name=row["repo_name"],
            call_type=row["call_type"],
        )

    # =========================================================================
    # Dependency Injection Support
    # =========================================================================

    def add_class_dependencies(
        self, class_name: str, dependencies: list[dict], repo_name: str
    ) -> int:
        """Store class dependency mappings for DI resolution.

        Args:
            class_name: The name of the class (e.g., "UserController")
            dependencies: List of dicts with 'name', 'type', 'injection_type' keys
            repo_name: Repository name

        Returns:
            Number of dependencies added
        """
        if not dependencies:
            return 0

        conn = self._get_connection()
        try:
            data = [
                (class_name, dep["name"], dep["type"], dep.get("injection_type", "field"), repo_name)
                for dep in dependencies
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO class_dependencies
                (class_name, field_name, field_type, injection_type, repo_name)
                VALUES (?, ?, ?, ?, ?)
                """,
                data,
            )
            conn.commit()
            return len(data)
        finally:
            conn.close()

    def get_class_dependencies(self, class_name: str, repo_name: Optional[str] = None) -> list[dict]:
        """Get dependency mappings for a class.

        Args:
            class_name: The class name to look up
            repo_name: Optional repository filter

        Returns:
            List of dependency dicts with 'field_name', 'field_type', 'injection_type'
        """
        conn = self._get_connection()
        try:
            if repo_name:
                cursor = conn.execute(
                    """
                    SELECT field_name, field_type, injection_type
                    FROM class_dependencies
                    WHERE class_name = ? AND repo_name = ?
                    """,
                    (class_name, repo_name),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT field_name, field_type, injection_type
                    FROM class_dependencies
                    WHERE class_name = ?
                    """,
                    (class_name,),
                )
            return [
                {
                    "field_name": row["field_name"],
                    "field_type": row["field_type"],
                    "injection_type": row["injection_type"],
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def resolve_callee(
        self, caller_class: str, callee: str, repo_name: Optional[str] = None
    ) -> str:
        """Resolve a callee through dependency injection.

        If the callee is in the form "fieldName.method", and fieldName is a known
        dependency of the caller class, resolve it to "FieldType.method".

        Args:
            caller_class: The class making the call
            callee: The callee expression (e.g., "userService.save")
            repo_name: Optional repository filter

        Returns:
            Resolved callee name (e.g., "UserService.save") or original if not resolvable
        """
        if "." not in callee:
            return callee

        parts = callee.split(".", 1)
        field_name = parts[0]
        method_name = parts[1] if len(parts) > 1 else ""

        # Skip common non-dependency prefixes
        if field_name in ("this", "self", "super"):
            return callee

        # Look up the field in class dependencies
        dependencies = self.get_class_dependencies(caller_class, repo_name)
        for dep in dependencies:
            if dep["field_name"] == field_name:
                resolved = f"{dep['field_type']}.{method_name}" if method_name else dep["field_type"]
                return resolved

        return callee

    def find_callees_resolved(
        self,
        symbol: str,
        repo_name: Optional[str] = None,
        max_depth: int = 1,
    ) -> list[CallRelation]:
        """Find callees with dependency injection resolution.

        Like find_callees, but attempts to resolve calls through injected dependencies.

        Args:
            symbol: Qualified name of the function (e.g., "UserController.getUser")
            repo_name: Optional repository filter
            max_depth: How many levels down to traverse

        Returns:
            List of call relations with resolved callees
        """
        relations = self.find_callees(symbol, repo_name, max_depth)

        # Extract class name from symbol
        caller_class = symbol.split(".")[0] if "." in symbol else symbol

        # Resolve each callee
        resolved_relations = []
        for rel in relations:
            resolved_callee = self.resolve_callee(caller_class, rel.callee, repo_name)
            resolved_relations.append(
                CallRelation(
                    caller=rel.caller,
                    callee=resolved_callee,
                    caller_file=rel.caller_file,
                    caller_line=rel.caller_line,
                    repo_name=rel.repo_name,
                    call_type="resolved" if resolved_callee != rel.callee else rel.call_type,
                )
            )

        return resolved_relations

    # =========================================================================
    # Polymorphic Call Resolution (Epic 3.1)
    # =========================================================================

    def find_callees_polymorphic(
        self,
        symbol: str,
        repo_name: Optional[str] = None,
        max_depth: int = 1,
        symbol_table: Optional[SymbolTableService] = None,
    ) -> list[CallRelation]:
        """Find callees with polymorphic/interface resolution.

        When a call is made to an interface method, this also returns
        the concrete implementations of that method.

        Args:
            symbol: Qualified name of the function (e.g., "UserController.getUser")
            repo_name: Optional repository filter
            max_depth: How many levels down to traverse
            symbol_table: SymbolTableService for looking up implementations

        Returns:
            List of call relations including virtual/polymorphic calls
        """
        # First get resolved calls (handles DI)
        relations = self.find_callees_resolved(symbol, repo_name, max_depth)

        if symbol_table is None:
            return relations

        # For each call, check if it's to an interface and find implementations
        polymorphic_relations = []
        for rel in relations:
            polymorphic_relations.append(rel)

            # Check if callee is an interface method (e.g., "Repository.save")
            if "." in rel.callee:
                class_name, method_name = rel.callee.rsplit(".", 1)

                # Check if class_name is an interface
                symbols = symbol_table.lookup(class_name, exact=True)
                is_interface = any(s.symbol_type == "interface" for s in symbols)

                if is_interface:
                    # Find all implementations
                    implementations = symbol_table.find_implementations(class_name, repo_name)

                    for impl in implementations:
                        impl_callee = f"{impl['child_name']}.{method_name}"
                        polymorphic_relations.append(
                            CallRelation(
                                caller=rel.caller,
                                callee=impl_callee,
                                caller_file=rel.caller_file,
                                caller_line=rel.caller_line,
                                repo_name=rel.repo_name,
                                call_type="virtual",
                            )
                        )

        return polymorphic_relations

    def find_callers_polymorphic(
        self,
        symbol: str,
        repo_name: Optional[str] = None,
        max_depth: int = 1,
        symbol_table: Optional[SymbolTableService] = None,
    ) -> list[CallRelation]:
        """Find callers including those calling through interfaces.

        If the symbol is a method on a class that implements an interface,
        this also returns callers that call the interface method.

        Args:
            symbol: Qualified name of the function (e.g., "UserServiceImpl.save")
            repo_name: Optional repository filter
            max_depth: How many levels up to traverse
            symbol_table: SymbolTableService for looking up parent interfaces

        Returns:
            List of call relations including virtual/polymorphic calls
        """
        # First get direct callers
        relations = self.find_callers(symbol, repo_name, max_depth)

        if symbol_table is None:
            return relations

        # Check if the symbol's class implements any interfaces
        if "." in symbol:
            class_name, method_name = symbol.rsplit(".", 1)

            # Find parent interfaces
            parents = symbol_table.find_parents(class_name, repo_name)

            for parent in parents:
                if parent["relation_type"] == "implements":
                    # Find callers of the interface method
                    interface_method = f"{parent['parent_name']}.{method_name}"
                    interface_callers = self.find_callers(interface_method, repo_name, max_depth)

                    for caller in interface_callers:
                        relations.append(
                            CallRelation(
                                caller=caller.caller,
                                callee=symbol,  # Point to the concrete implementation
                                caller_file=caller.caller_file,
                                caller_line=caller.caller_line,
                                repo_name=caller.repo_name,
                                call_type="virtual",
                            )
                        )

        return relations

    # =========================================================================
    # Dead Code Detection (Epic 4.1)
    # =========================================================================

    def get_all_symbols(self, repo_name: Optional[str] = None) -> set[str]:
        """Get all unique symbols (callers and callees) from the call graph.

        Args:
            repo_name: Optional repository filter

        Returns:
            Set of all symbol names in the call graph
        """
        conn = self._get_connection()
        try:
            if repo_name:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT caller FROM calls WHERE repo_name = ?
                    UNION
                    SELECT DISTINCT callee FROM calls WHERE repo_name = ?
                    """,
                    (repo_name, repo_name),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT caller FROM calls
                    UNION
                    SELECT DISTINCT callee FROM calls
                    """
                )
            return {row[0] for row in cursor.fetchall()}
        finally:
            conn.close()

    def find_symbols_with_no_callers(
        self, repo_name: Optional[str] = None
    ) -> list[str]:
        """Find symbols that are never called by anything.

        Args:
            repo_name: Optional repository filter

        Returns:
            List of symbol names that have no incoming calls
        """
        conn = self._get_connection()
        try:
            if repo_name:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT caller FROM calls WHERE repo_name = ?
                    EXCEPT
                    SELECT DISTINCT callee FROM calls WHERE repo_name = ?
                    """,
                    (repo_name, repo_name),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT caller FROM calls
                    EXCEPT
                    SELECT DISTINCT callee FROM calls
                    """
                )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def find_unreachable_from_entry_points(
        self,
        entry_points: list[str],
        repo_name: Optional[str] = None,
    ) -> list[str]:
        """Find all symbols not reachable from the given entry points.

        Args:
            entry_points: List of entry point symbol names (e.g., main, API routes)
            repo_name: Optional repository filter

        Returns:
            List of symbol names that cannot be reached from any entry point
        """
        # Get all symbols
        all_symbols = self.get_all_symbols(repo_name)

        # BFS from all entry points to find reachable symbols
        reachable = set()
        queue = list(entry_points)

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)

            # Find all callees
            callees = self.find_callees(current, repo_name, max_depth=1)
            for callee in callees:
                if callee.callee not in reachable:
                    queue.append(callee.callee)

        # Return symbols not reachable from entry points
        unreachable = all_symbols - reachable
        return list(unreachable)

    def identify_entry_points(
        self,
        chunks: list,  # list[CodeChunk] - using list to avoid circular import
        repo_name: Optional[str] = None,
    ) -> list[str]:
        """Identify entry points from chunks based on metadata and patterns.

        Entry points are:
        - Functions with route metadata (API endpoints)
        - Functions named 'main' or '__main__'
        - React page components (in pages/ directory)
        - Test functions (test_*)
        - Functions with no callers but called by external code

        Args:
            chunks: List of CodeChunk objects
            repo_name: Optional repository filter

        Returns:
            List of entry point symbol names
        """
        entry_points = []

        for chunk in chunks:
            if repo_name and chunk.repo_name != repo_name:
                continue

            qualified_name = chunk.get_qualified_name()

            # Check for route metadata (API endpoints)
            if chunk.metadata.get("route"):
                entry_points.append(qualified_name)
                continue

            # Check for main functions
            if chunk.name.lower() in ("main", "__main__"):
                entry_points.append(qualified_name)
                continue

            # Check for React page components (in pages/ or app/ directory)
            if chunk.metadata.get("is_react_component"):
                if "/pages/" in chunk.file_path or "/app/" in chunk.file_path:
                    entry_points.append(qualified_name)
                    continue

            # Check for test functions
            if chunk.name.startswith("test_") or chunk.name.startswith("Test"):
                entry_points.append(qualified_name)
                continue

        return entry_points

    # =========================================================================
    # Circular Dependency Detection (Epic 4.2)
    # =========================================================================

    def detect_cycles(
        self, repo_name: Optional[str] = None, group_by_file: bool = True
    ) -> list[list[str]]:
        """Detect circular dependencies in the call graph.

        Uses Tarjan's algorithm to find strongly connected components (SCCs)
        with more than one node, indicating cycles.

        Args:
            repo_name: Optional repository filter
            group_by_file: If True, detect file-level cycles; otherwise function-level

        Returns:
            List of cycles, each cycle is a list of symbol/file names
        """
        conn = self._get_connection()
        try:
            # Build adjacency list
            if repo_name:
                cursor = conn.execute(
                    """
                    SELECT caller, callee, caller_file FROM calls
                    WHERE repo_name = ?
                    """,
                    (repo_name,),
                )
            else:
                cursor = conn.execute("SELECT caller, callee, caller_file FROM calls")

            rows = cursor.fetchall()

            if group_by_file:
                # Group by file
                graph = {}
                file_map = {}  # symbol -> file

                for row in rows:
                    caller, callee, caller_file = row
                    file_map[caller] = caller_file

                    # Get callee file (approximate from caller file if not available)
                    callee_file = file_map.get(callee, caller_file)

                    if caller_file != callee_file:
                        if caller_file not in graph:
                            graph[caller_file] = set()
                        graph[caller_file].add(callee_file)
            else:
                # Function-level graph
                graph = {}
                for row in rows:
                    caller, callee, _ = row
                    if caller not in graph:
                        graph[caller] = set()
                    graph[caller].add(callee)

            return self._tarjan_scc(graph)
        finally:
            conn.close()

    def _tarjan_scc(self, graph: dict[str, set[str]]) -> list[list[str]]:
        """Find strongly connected components using Tarjan's algorithm.

        Args:
            graph: Adjacency list representation of the graph

        Returns:
            List of SCCs with more than one node (cycles)
        """
        index_counter = [0]
        stack = []
        lowlink = {}
        index = {}
        on_stack = {}
        sccs = []

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlink[node] = index_counter[0]
            index_counter[0] += 1
            on_stack[node] = True
            stack.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlink[node] = min(lowlink[node], lowlink[neighbor])
                elif on_stack.get(neighbor, False):
                    lowlink[node] = min(lowlink[node], index[neighbor])

            if lowlink[node] == index[node]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == node:
                        break
                # Only keep cycles (SCCs with more than one node)
                if len(scc) > 1:
                    sccs.append(scc)

        for node in graph:
            if node not in index:
                strongconnect(node)

        return sccs
