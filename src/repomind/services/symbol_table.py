"""Symbol Table service for fast exact lookups.

Provides O(log n) lookup for symbols by name, type, or file.
Uses SQLite with B-tree indices for efficient queries.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import get_config
from ..models.chunk import ChunkType, CodeChunk


@dataclass
class Symbol:
    """A code symbol entry in the symbol table."""

    name: str
    qualified_name: str  # e.g., "ClassName.methodName"
    symbol_type: str  # class, method, function, interface
    file_path: str
    repo_name: str
    start_line: int
    end_line: int
    signature: Optional[str] = None
    parent_name: Optional[str] = None
    language: str = "unknown"
    chunk_id: Optional[str] = None  # Link to vector store


class SymbolTableService:
    """SQLite-based symbol table for fast exact lookups.

    Provides:
    - O(log n) lookup by symbol name
    - Prefix search for autocomplete
    - Filtering by type, file, or repository
    - Parent-child relationships
    """

    def __init__(self):
        config = get_config()
        self.db_path = config.index.metadata_dir / "symbols.db"
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
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    repo_name TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    signature TEXT,
                    parent_name TEXT,
                    language TEXT NOT NULL,
                    chunk_id TEXT,
                    UNIQUE(repo_name, file_path, name, symbol_type, start_line)
                );

                -- Indices for fast lookups
                CREATE INDEX IF NOT EXISTS idx_name ON symbols(name);
                CREATE INDEX IF NOT EXISTS idx_qualified_name ON symbols(qualified_name);
                CREATE INDEX IF NOT EXISTS idx_type ON symbols(symbol_type);
                CREATE INDEX IF NOT EXISTS idx_file ON symbols(repo_name, file_path);
                CREATE INDEX IF NOT EXISTS idx_parent ON symbols(parent_name);
                CREATE INDEX IF NOT EXISTS idx_chunk ON symbols(chunk_id);

                -- Full-text search index for prefix matching
                CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                    name,
                    qualified_name,
                    content=symbols,
                    content_rowid=id
                );

                -- Triggers to keep FTS in sync
                CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
                    INSERT INTO symbols_fts(rowid, name, qualified_name)
                    VALUES (new.id, new.name, new.qualified_name);
                END;

                CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
                    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name)
                    VALUES ('delete', old.id, old.name, old.qualified_name);
                END;

                CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
                    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name)
                    VALUES ('delete', old.id, old.name, old.qualified_name);
                    INSERT INTO symbols_fts(rowid, name, qualified_name)
                    VALUES (new.id, new.name, new.qualified_name);
                END;
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add_symbol(self, symbol: Symbol) -> int:
        """Add a symbol to the table.

        Returns:
            The row ID of the inserted symbol
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO symbols
                (name, qualified_name, symbol_type, file_path, repo_name,
                 start_line, end_line, signature, parent_name, language, chunk_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.name,
                    symbol.qualified_name,
                    symbol.symbol_type,
                    symbol.file_path,
                    symbol.repo_name,
                    symbol.start_line,
                    symbol.end_line,
                    symbol.signature,
                    symbol.parent_name,
                    symbol.language,
                    symbol.chunk_id,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def add_symbols_from_chunks(self, chunks: list[CodeChunk]) -> int:
        """Bulk add symbols from code chunks.

        Returns:
            Number of symbols added
        """
        conn = self._get_connection()
        try:
            symbols_data = [
                (
                    chunk.name,
                    chunk.get_qualified_name(),
                    chunk.chunk_type.value,
                    chunk.file_path,
                    chunk.repo_name,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.signature,
                    chunk.parent_name,
                    chunk.language,
                    chunk.id,
                )
                for chunk in chunks
            ]

            conn.executemany(
                """
                INSERT OR REPLACE INTO symbols
                (name, qualified_name, symbol_type, file_path, repo_name,
                 start_line, end_line, signature, parent_name, language, chunk_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                symbols_data,
            )
            conn.commit()
            return len(symbols_data)
        finally:
            conn.close()

    def lookup(self, name: str, exact: bool = True) -> list[Symbol]:
        """Look up symbols by name.

        Args:
            name: Symbol name to search for
            exact: If True, exact match. If False, prefix match.

        Returns:
            List of matching symbols
        """
        conn = self._get_connection()
        try:
            if exact:
                # Exact match - check both name and qualified_name
                cursor = conn.execute(
                    """
                    SELECT * FROM symbols
                    WHERE name = ? OR qualified_name = ?
                    ORDER BY
                        CASE WHEN name = ? THEN 0 ELSE 1 END,
                        repo_name, file_path, start_line
                    """,
                    (name, name, name),
                )
            else:
                # Prefix match using FTS
                cursor = conn.execute(
                    """
                    SELECT s.* FROM symbols s
                    JOIN symbols_fts f ON s.id = f.rowid
                    WHERE symbols_fts MATCH ?
                    ORDER BY rank
                    LIMIT 50
                    """,
                    (f"{name}*",),
                )

            return [self._row_to_symbol(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def lookup_by_type(
        self,
        symbol_type: str | ChunkType,
        repo_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[Symbol]:
        """Look up symbols by type.

        Args:
            symbol_type: Type to filter by (class, method, function, etc.)
            repo_name: Optional repository filter
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        if isinstance(symbol_type, ChunkType):
            symbol_type = symbol_type.value

        conn = self._get_connection()
        try:
            if repo_name:
                cursor = conn.execute(
                    """
                    SELECT * FROM symbols
                    WHERE symbol_type = ? AND repo_name = ?
                    ORDER BY name
                    LIMIT ?
                    """,
                    (symbol_type, repo_name, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM symbols
                    WHERE symbol_type = ?
                    ORDER BY name
                    LIMIT ?
                    """,
                    (symbol_type, limit),
                )

            return [self._row_to_symbol(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def lookup_by_file(self, repo_name: str, file_path: str) -> list[Symbol]:
        """Get all symbols in a specific file.

        Args:
            repo_name: Repository name
            file_path: Relative file path

        Returns:
            List of symbols in the file, ordered by line number
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM symbols
                WHERE repo_name = ? AND file_path = ?
                ORDER BY start_line
                """,
                (repo_name, file_path),
            )
            return [self._row_to_symbol(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def lookup_children(self, parent_name: str, repo_name: Optional[str] = None) -> list[Symbol]:
        """Get all children of a parent symbol (e.g., methods of a class).

        Args:
            parent_name: Name of the parent symbol
            repo_name: Optional repository filter

        Returns:
            List of child symbols
        """
        conn = self._get_connection()
        try:
            if repo_name:
                cursor = conn.execute(
                    """
                    SELECT * FROM symbols
                    WHERE parent_name = ? AND repo_name = ?
                    ORDER BY start_line
                    """,
                    (parent_name, repo_name),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM symbols
                    WHERE parent_name = ?
                    ORDER BY repo_name, file_path, start_line
                    """,
                    (parent_name,),
                )

            return [self._row_to_symbol(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def search(
        self,
        query: str,
        symbol_type: Optional[str] = None,
        repo_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[Symbol]:
        """Full-text search for symbols.

        Args:
            query: Search query (supports FTS5 syntax)
            symbol_type: Optional type filter
            repo_name: Optional repository filter
            limit: Maximum results

        Returns:
            List of matching symbols, ranked by relevance
        """
        conn = self._get_connection()
        try:
            # Build query with optional filters
            sql = """
                SELECT s.* FROM symbols s
                JOIN symbols_fts f ON s.id = f.rowid
                WHERE symbols_fts MATCH ?
            """
            params = [query]

            if symbol_type:
                sql += " AND s.symbol_type = ?"
                params.append(symbol_type)

            if repo_name:
                sql += " AND s.repo_name = ?"
                params.append(repo_name)

            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            return [self._row_to_symbol(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_repo(self, repo_name: str) -> int:
        """Delete all symbols for a repository.

        Returns:
            Number of symbols deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM symbols WHERE repo_name = ?",
                (repo_name,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def delete_file(self, repo_name: str, file_path: str) -> int:
        """Delete all symbols for a specific file.

        Returns:
            Number of symbols deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM symbols WHERE repo_name = ? AND file_path = ?",
                (repo_name, file_path),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Get statistics about the symbol table."""
        conn = self._get_connection()
        try:
            stats = {}

            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM symbols")
            stats["total_symbols"] = cursor.fetchone()[0]

            # By type
            cursor = conn.execute(
                """
                SELECT symbol_type, COUNT(*) as count
                FROM symbols
                GROUP BY symbol_type
                ORDER BY count DESC
                """
            )
            stats["by_type"] = {row["symbol_type"]: row["count"] for row in cursor.fetchall()}

            # By repo
            cursor = conn.execute(
                """
                SELECT repo_name, COUNT(*) as count
                FROM symbols
                GROUP BY repo_name
                ORDER BY count DESC
                """
            )
            stats["by_repo"] = {row["repo_name"]: row["count"] for row in cursor.fetchall()}

            # By language
            cursor = conn.execute(
                """
                SELECT language, COUNT(*) as count
                FROM symbols
                GROUP BY language
                ORDER BY count DESC
                """
            )
            stats["by_language"] = {row["language"]: row["count"] for row in cursor.fetchall()}

            return stats
        finally:
            conn.close()

    def _row_to_symbol(self, row: sqlite3.Row) -> Symbol:
        """Convert a database row to a Symbol object."""
        return Symbol(
            name=row["name"],
            qualified_name=row["qualified_name"],
            symbol_type=row["symbol_type"],
            file_path=row["file_path"],
            repo_name=row["repo_name"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            signature=row["signature"],
            parent_name=row["parent_name"],
            language=row["language"],
            chunk_id=row["chunk_id"],
        )
