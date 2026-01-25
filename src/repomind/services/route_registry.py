"""Route Registry Service with SQLite Storage.

This module provides persistent storage for routes with:
- Endpoint and API call registration
- Route link storage with confidence scores
- Conflict detection
- Ambiguous match tracking
- Query capabilities for analysis

Schema Overview:
- endpoints: Backend API endpoints with normalized paths
- api_calls: Frontend API calls
- route_links: Matched links between calls and endpoints
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from ..config import get_config
from ..models.route import (
    ApiCall,
    NormalizedRoute,
    RouteEndpoint,
    RouteMatch,
    RouteConflict,
    RouteParameter,
    ParameterSource,
)
from .route_normalizer import RouteNormalizer


class RouteRegistry:
    """Central storage for all routes with SQLite persistence.

    Provides:
    - Registration of endpoints and API calls
    - Persistent storage of route links
    - Conflict detection queries
    - Ambiguous match retrieval
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the route registry.

        Args:
            db_path: Custom database path. If None, uses config default.
        """
        self.config = get_config()
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = self.config.index.metadata_dir / "routes.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.normalizer = RouteNormalizer()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Endpoints table: Backend API route handlers
                CREATE TABLE IF NOT EXISTS endpoints (
                    id TEXT PRIMARY KEY,
                    method TEXT NOT NULL,
                    full_path TEXT NOT NULL,
                    path_pattern TEXT NOT NULL,
                    parameters_json TEXT,
                    framework TEXT,
                    repo_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_id TEXT,
                    qualified_name TEXT NOT NULL,
                    language TEXT,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- API calls table: Frontend outgoing API calls
                CREATE TABLE IF NOT EXISTS api_calls (
                    id TEXT PRIMARY KEY,
                    method TEXT NOT NULL,
                    url TEXT NOT NULL,
                    normalized_path TEXT NOT NULL,
                    parameters_json TEXT,
                    repo_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    chunk_id TEXT,
                    caller_qualified_name TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Route links table: Matched call-endpoint pairs
                CREATE TABLE IF NOT EXISTS route_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL,
                    endpoint_id TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_ambiguous BOOLEAN DEFAULT FALSE,
                    alternatives INTEGER DEFAULT 0,
                    match_details_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (call_id) REFERENCES api_calls(id) ON DELETE CASCADE,
                    FOREIGN KEY (endpoint_id) REFERENCES endpoints(id) ON DELETE CASCADE,
                    UNIQUE(call_id, endpoint_id)
                );

                -- Indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_endpoints_method_path ON endpoints(method, path_pattern);
                CREATE INDEX IF NOT EXISTS idx_endpoints_repo ON endpoints(repo_name);
                CREATE INDEX IF NOT EXISTS idx_api_calls_method_path ON api_calls(method, normalized_path);
                CREATE INDEX IF NOT EXISTS idx_api_calls_repo ON api_calls(repo_name);
                CREATE INDEX IF NOT EXISTS idx_route_links_ambiguous ON route_links(is_ambiguous);
                CREATE INDEX IF NOT EXISTS idx_route_links_call ON route_links(call_id);
                CREATE INDEX IF NOT EXISTS idx_route_links_endpoint ON route_links(endpoint_id);
            """)

    @contextmanager
    def _get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================================
    # Registration Methods
    # =========================================================================

    def register_endpoint(self, endpoint: RouteEndpoint) -> str:
        """Register a backend API endpoint.

        Args:
            endpoint: RouteEndpoint to register

        Returns:
            The endpoint ID
        """
        params_json = json.dumps([
            {
                "name": p.name,
                "position": p.position,
                "source": p.source.value,
                "type_hint": p.type_hint,
                "required": p.required,
            }
            for p in endpoint.normalized_route.parameters
        ])

        metadata_json = json.dumps(endpoint.metadata) if endpoint.metadata else None

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO endpoints
                (id, method, full_path, path_pattern, parameters_json, framework,
                 repo_name, file_path, chunk_id, qualified_name, language, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                endpoint.id,
                endpoint.method,
                endpoint.normalized_route.original_path,
                endpoint.normalized_route.path_pattern,
                params_json,
                endpoint.framework,
                endpoint.repo_name,
                endpoint.file_path,
                endpoint.chunk_id,
                endpoint.qualified_name,
                endpoint.language,
                metadata_json,
            ))

        return endpoint.id

    def register_api_call(self, call: ApiCall) -> str:
        """Register a frontend API call.

        Args:
            call: ApiCall to register

        Returns:
            The call ID
        """
        params_json = json.dumps([
            {
                "name": p.name,
                "position": p.position,
                "source": p.source.value,
                "type_hint": p.type_hint,
                "required": p.required,
            }
            for p in call.normalized_route.parameters
        ])

        metadata_json = json.dumps(call.metadata) if call.metadata else None

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO api_calls
                (id, method, url, normalized_path, parameters_json,
                 repo_name, file_path, line_number, chunk_id, caller_qualified_name, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call.id,
                call.method,
                call.url,
                call.normalized_route.path_pattern,
                params_json,
                call.repo_name,
                call.file_path,
                call.line_number,
                call.chunk_id,
                call.caller_qualified_name,
                metadata_json,
            ))

        return call.id

    def register_link(self, match: RouteMatch) -> int:
        """Register a route link (matched call-endpoint pair).

        Args:
            match: RouteMatch to register

        Returns:
            The link row ID
        """
        match_details_json = json.dumps(match.match_details) if match.match_details else None

        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO route_links
                (call_id, endpoint_id, confidence, is_ambiguous, alternatives, match_details_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                match.api_call.id,
                match.endpoint.id,
                match.confidence,
                match.is_ambiguous,
                match.alternatives,
                match_details_json,
            ))

        return cursor.lastrowid

    def register_bulk(
        self,
        endpoints: list[RouteEndpoint],
        calls: list[ApiCall],
        links: list[RouteMatch],
    ) -> dict:
        """Bulk register endpoints, calls, and links.

        Args:
            endpoints: List of endpoints to register
            calls: List of API calls to register
            links: List of route matches to register

        Returns:
            Dict with counts of registered items
        """
        endpoint_count = 0
        call_count = 0
        link_count = 0

        for endpoint in endpoints:
            self.register_endpoint(endpoint)
            endpoint_count += 1

        for call in calls:
            self.register_api_call(call)
            call_count += 1

        for link in links:
            self.register_link(link)
            link_count += 1

        return {
            "endpoints": endpoint_count,
            "calls": call_count,
            "links": link_count,
        }

    # =========================================================================
    # Query Methods
    # =========================================================================

    def find_endpoints(
        self,
        method: Optional[str] = None,
        path_pattern: Optional[str] = None,
        repo_name: Optional[str] = None,
        framework: Optional[str] = None,
    ) -> list[RouteEndpoint]:
        """Find endpoints matching criteria.

        Args:
            method: HTTP method filter
            path_pattern: Path pattern filter (supports LIKE wildcards)
            repo_name: Repository name filter
            framework: Framework filter

        Returns:
            List of matching RouteEndpoint objects
        """
        query = "SELECT * FROM endpoints WHERE 1=1"
        params = []

        if method:
            query += " AND method = ?"
            params.append(method.upper())

        if path_pattern:
            query += " AND path_pattern LIKE ?"
            params.append(path_pattern.replace("*", "%"))

        if repo_name:
            query += " AND repo_name = ?"
            params.append(repo_name)

        if framework:
            query += " AND framework = ?"
            params.append(framework)

        endpoints = []
        with self._get_connection() as conn:
            for row in conn.execute(query, params):
                endpoints.append(self._row_to_endpoint(row))

        return endpoints

    def find_api_calls(
        self,
        method: Optional[str] = None,
        path_pattern: Optional[str] = None,
        repo_name: Optional[str] = None,
    ) -> list[ApiCall]:
        """Find API calls matching criteria.

        Args:
            method: HTTP method filter
            path_pattern: Path pattern filter
            repo_name: Repository name filter

        Returns:
            List of matching ApiCall objects
        """
        query = "SELECT * FROM api_calls WHERE 1=1"
        params = []

        if method:
            query += " AND method = ?"
            params.append(method.upper())

        if path_pattern:
            query += " AND normalized_path LIKE ?"
            params.append(path_pattern.replace("*", "%"))

        if repo_name:
            query += " AND repo_name = ?"
            params.append(repo_name)

        calls = []
        with self._get_connection() as conn:
            for row in conn.execute(query, params):
                calls.append(self._row_to_api_call(row))

        return calls

    def get_ambiguous_links(
        self,
        repo_name: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> list[dict]:
        """Get all ambiguous links requiring manual review.

        Args:
            repo_name: Optional repository filter
            min_confidence: Minimum confidence filter

        Returns:
            List of dicts with link details
        """
        query = """
            SELECT
                rl.*,
                ac.url as call_url,
                ac.caller_qualified_name,
                ac.file_path as call_file,
                ac.line_number,
                e.path_pattern as endpoint_path,
                e.qualified_name as endpoint_handler,
                e.file_path as endpoint_file,
                e.framework
            FROM route_links rl
            JOIN api_calls ac ON rl.call_id = ac.id
            JOIN endpoints e ON rl.endpoint_id = e.id
            WHERE rl.is_ambiguous = TRUE
        """
        params = []

        if repo_name:
            query += " AND (ac.repo_name = ? OR e.repo_name = ?)"
            params.extend([repo_name, repo_name])

        if min_confidence is not None:
            query += " AND rl.confidence >= ?"
            params.append(min_confidence)

        query += " ORDER BY rl.confidence DESC"

        results = []
        with self._get_connection() as conn:
            for row in conn.execute(query, params):
                results.append({
                    "call_url": row["call_url"],
                    "caller": row["caller_qualified_name"],
                    "call_file": row["call_file"],
                    "call_line": row["line_number"],
                    "endpoint_path": row["endpoint_path"],
                    "endpoint_handler": row["endpoint_handler"],
                    "endpoint_file": row["endpoint_file"],
                    "confidence": row["confidence"],
                    "alternatives": row["alternatives"],
                    "framework": row["framework"],
                })

        return results

    def find_conflicts(self) -> list[RouteConflict]:
        """Find conflicting routes (duplicate handlers for same path).

        Returns:
            List of RouteConflict objects
        """
        query = """
            SELECT method, path_pattern, COUNT(*) as count,
                   GROUP_CONCAT(qualified_name, '|') as handlers,
                   GROUP_CONCAT(id, '|') as endpoint_ids
            FROM endpoints
            GROUP BY method, path_pattern
            HAVING count > 1
        """

        conflicts = []
        with self._get_connection() as conn:
            for row in conn.execute(query):
                endpoint_ids = row["endpoint_ids"].split("|")
                handlers = row["handlers"].split("|")

                # Fetch full endpoint objects
                endpoints = []
                for eid in endpoint_ids:
                    ep_row = conn.execute(
                        "SELECT * FROM endpoints WHERE id = ?", (eid,)
                    ).fetchone()
                    if ep_row:
                        endpoints.append(self._row_to_endpoint(ep_row))

                conflicts.append(RouteConflict(
                    endpoints=endpoints,
                    conflict_type="duplicate",
                    description=f"Route {row['method']} {row['path_pattern']} has {row['count']} handlers: {', '.join(handlers)}",
                ))

        return conflicts

    def get_unlinked_calls(self, repo_name: Optional[str] = None) -> list[ApiCall]:
        """Get API calls that have no matching endpoint.

        Args:
            repo_name: Optional repository filter

        Returns:
            List of unlinked ApiCall objects
        """
        query = """
            SELECT ac.* FROM api_calls ac
            LEFT JOIN route_links rl ON ac.id = rl.call_id
            WHERE rl.call_id IS NULL
        """
        params = []

        if repo_name:
            query += " AND ac.repo_name = ?"
            params.append(repo_name)

        calls = []
        with self._get_connection() as conn:
            for row in conn.execute(query, params):
                calls.append(self._row_to_api_call(row))

        return calls

    def get_unlinked_endpoints(self, repo_name: Optional[str] = None) -> list[RouteEndpoint]:
        """Get endpoints that are never called.

        Args:
            repo_name: Optional repository filter

        Returns:
            List of unlinked RouteEndpoint objects
        """
        query = """
            SELECT e.* FROM endpoints e
            LEFT JOIN route_links rl ON e.id = rl.endpoint_id
            WHERE rl.endpoint_id IS NULL
        """
        params = []

        if repo_name:
            query += " AND e.repo_name = ?"
            params.append(repo_name)

        endpoints = []
        with self._get_connection() as conn:
            for row in conn.execute(query, params):
                endpoints.append(self._row_to_endpoint(row))

        return endpoints

    def get_statistics(self, repo_name: Optional[str] = None) -> dict:
        """Get registry statistics.

        Args:
            repo_name: Optional repository filter

        Returns:
            Dict with counts and statistics
        """
        with self._get_connection() as conn:
            base_filter = " WHERE repo_name = ?" if repo_name else ""
            params = [repo_name] if repo_name else []

            endpoint_count = conn.execute(
                f"SELECT COUNT(*) FROM endpoints{base_filter}", params
            ).fetchone()[0]

            call_count = conn.execute(
                f"SELECT COUNT(*) FROM api_calls{base_filter}", params
            ).fetchone()[0]

            link_count = conn.execute(
                "SELECT COUNT(*) FROM route_links"
            ).fetchone()[0]

            ambiguous_count = conn.execute(
                "SELECT COUNT(*) FROM route_links WHERE is_ambiguous = TRUE"
            ).fetchone()[0]

            # Average confidence
            avg_conf = conn.execute(
                "SELECT AVG(confidence) FROM route_links"
            ).fetchone()[0] or 0

        return {
            "total_endpoints": endpoint_count,
            "total_calls": call_count,
            "total_links": link_count,
            "ambiguous_links": ambiguous_count,
            "average_confidence": round(avg_conf, 3),
            "repo_filter": repo_name,
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _row_to_endpoint(self, row: sqlite3.Row) -> RouteEndpoint:
        """Convert a database row to RouteEndpoint."""
        parameters = []
        if row["parameters_json"]:
            params_data = json.loads(row["parameters_json"])
            for p in params_data:
                parameters.append(RouteParameter(
                    name=p["name"],
                    position=p["position"],
                    source=ParameterSource(p["source"]),
                    type_hint=p.get("type_hint"),
                    required=p.get("required", True),
                ))

        normalized_route = NormalizedRoute(
            method=row["method"],
            path_pattern=row["path_pattern"],
            path_segments=self.normalizer._build_segments(row["path_pattern"]),
            parameters=parameters,
            original_path=row["full_path"],
            framework=row["framework"],
        )

        metadata = {}
        if row["metadata_json"]:
            metadata = json.loads(row["metadata_json"])

        return RouteEndpoint(
            id=row["id"],
            method=row["method"],
            normalized_route=normalized_route,
            chunk_id=row["chunk_id"],
            qualified_name=row["qualified_name"],
            file_path=row["file_path"],
            repo_name=row["repo_name"],
            language=row["language"],
            framework=row["framework"],
            metadata=metadata,
        )

    def _row_to_api_call(self, row: sqlite3.Row) -> ApiCall:
        """Convert a database row to ApiCall."""
        parameters = []
        if row["parameters_json"]:
            params_data = json.loads(row["parameters_json"])
            for p in params_data:
                parameters.append(RouteParameter(
                    name=p["name"],
                    position=p["position"],
                    source=ParameterSource(p["source"]),
                    type_hint=p.get("type_hint"),
                    required=p.get("required", True),
                ))

        normalized_route = NormalizedRoute(
            method=row["method"],
            path_pattern=row["normalized_path"],
            path_segments=self.normalizer._build_segments(row["normalized_path"]),
            parameters=parameters,
            original_path=row["url"],
        )

        metadata = {}
        if row["metadata_json"]:
            metadata = json.loads(row["metadata_json"])

        return ApiCall(
            id=row["id"],
            method=row["method"],
            url=row["url"],
            normalized_route=normalized_route,
            chunk_id=row["chunk_id"],
            caller_qualified_name=row["caller_qualified_name"],
            file_path=row["file_path"],
            repo_name=row["repo_name"],
            line_number=row["line_number"],
            metadata=metadata,
        )

    def clear(self, repo_name: Optional[str] = None) -> dict:
        """Clear registry data.

        Args:
            repo_name: If provided, only clear data for this repo

        Returns:
            Dict with counts of deleted items
        """
        with self._get_connection() as conn:
            if repo_name:
                # Delete links associated with the repo
                conn.execute("""
                    DELETE FROM route_links WHERE
                    call_id IN (SELECT id FROM api_calls WHERE repo_name = ?) OR
                    endpoint_id IN (SELECT id FROM endpoints WHERE repo_name = ?)
                """, (repo_name, repo_name))

                ep_result = conn.execute(
                    "DELETE FROM endpoints WHERE repo_name = ?", (repo_name,)
                )
                call_result = conn.execute(
                    "DELETE FROM api_calls WHERE repo_name = ?", (repo_name,)
                )

                return {
                    "endpoints_deleted": ep_result.rowcount,
                    "calls_deleted": call_result.rowcount,
                }
            else:
                conn.execute("DELETE FROM route_links")
                conn.execute("DELETE FROM api_calls")
                conn.execute("DELETE FROM endpoints")

                return {"cleared": "all"}
