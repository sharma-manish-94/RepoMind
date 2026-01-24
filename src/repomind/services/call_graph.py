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
