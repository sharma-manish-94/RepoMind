"""Route Data Models for Cross-Language API Linking.

This module defines the core data models for route matching:

1. **RouteParameter**: A parameter in a route path (path or query)
2. **NormalizedRoute**: A normalized representation of a route for matching
3. **RouteEndpoint**: An API endpoint extracted from backend code
4. **ApiCall**: An outgoing API call from frontend code
5. **RouteMatch**: A matched link between a call and endpoint
6. **RouteConflict**: Detected conflicts between routes

These models preserve parameter names (not just positions) to enable
accurate matching across different frameworks and languages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class ParameterSource(str, Enum):
    """Source of a route parameter."""
    PATH = "path"       # Parameter in URL path: /users/{id}
    QUERY = "query"     # Query parameter: /users?limit=10
    BODY = "body"       # Request body parameter
    HEADER = "header"   # Header parameter


@dataclass
class RouteParameter:
    """A parameter in a route path or query string.

    Preserves the original parameter name for better matching accuracy.

    Attributes:
        name: Original parameter name (e.g., "userId", "id")
        position: 0-indexed position in path (for path params)
        source: Where the parameter comes from (path, query, body)
        type_hint: Optional type annotation (e.g., "int", "string")
        required: Whether the parameter is required

    Example:
        # For route /users/{userId}/orders/{orderId}?limit=10
        params = [
            RouteParameter(name="userId", position=1, source=ParameterSource.PATH),
            RouteParameter(name="orderId", position=3, source=ParameterSource.PATH),
            RouteParameter(name="limit", position=-1, source=ParameterSource.QUERY),
        ]
    """
    name: str
    position: int  # -1 for query params
    source: ParameterSource = ParameterSource.PATH
    type_hint: Optional[str] = None
    required: bool = True


@dataclass
class NormalizedRoute:
    """A normalized representation of a route for matching.

    Converts various framework-specific path formats to a unified format
    while preserving semantic information for accurate matching.

    Attributes:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        path_pattern: Normalized path pattern ("/users/{userId}/orders")
        path_segments: List of path segments (["users", "{userId}", "orders"])
        parameters: List of RouteParameter objects
        base_path: Optional API base path (e.g., "/api/v1")
        query_params: Query parameter names if present
        framework: Source framework ("fastapi", "spring", "express", "nestjs")
        original_path: The original path before normalization

    Example:
        # FastAPI route: @app.get("/users/{user_id}")
        route = NormalizedRoute(
            method="GET",
            path_pattern="/users/{user_id}",
            path_segments=["users", "{user_id}"],
            parameters=[RouteParameter("user_id", 1, ParameterSource.PATH)],
            framework="fastapi",
            original_path="/users/{user_id}"
        )
    """
    method: str
    path_pattern: str
    path_segments: list[str]
    parameters: list[RouteParameter] = field(default_factory=list)
    base_path: Optional[str] = None
    query_params: list[str] = field(default_factory=list)
    framework: Optional[str] = None
    original_path: str = ""

    def get_static_segments(self) -> list[tuple[int, str]]:
        """Get all static (non-parameter) segments with positions."""
        static = []
        for i, seg in enumerate(self.path_segments):
            if not self._is_param_segment(seg):
                static.append((i, seg))
        return static

    def get_param_segments(self) -> list[tuple[int, str]]:
        """Get all parameter segments with positions."""
        params = []
        for i, seg in enumerate(self.path_segments):
            if self._is_param_segment(seg):
                # Extract parameter name from {name}
                name = seg.strip("{}")
                params.append((i, name))
        return params

    def _is_param_segment(self, segment: str) -> bool:
        """Check if a segment is a parameter placeholder."""
        return segment.startswith("{") and segment.endswith("}")

    def matches_structure(self, other: "NormalizedRoute") -> bool:
        """Check if two routes have compatible structure."""
        if self.method != other.method:
            return False
        if len(self.path_segments) != len(other.path_segments):
            return False
        # Check that static segments match in same positions
        my_static = self.get_static_segments()
        other_static = other.get_static_segments()
        return my_static == other_static


@dataclass
class RouteEndpoint:
    """Represents an API route endpoint from backend code.

    Attributes:
        id: Unique identifier for the endpoint
        method: HTTP method
        normalized_route: The normalized route information
        chunk_id: ID of the CodeChunk containing this endpoint
        qualified_name: Function/method name handling the route
        file_path: Path to the source file
        repo_name: Repository name
        language: Programming language
        framework: Web framework (fastapi, spring, express, nestjs)
        metadata: Additional framework-specific metadata
    """
    id: str
    method: str
    normalized_route: NormalizedRoute
    chunk_id: str
    qualified_name: str
    file_path: str
    repo_name: str
    language: str
    framework: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        """Get the normalized path pattern."""
        return self.normalized_route.path_pattern

    @property
    def original_path(self) -> str:
        """Get the original path before normalization."""
        return self.normalized_route.original_path


@dataclass
class ApiCall:
    """Represents an outgoing API call from frontend code.

    Attributes:
        id: Unique identifier for the call
        method: HTTP method
        url: Original URL string
        normalized_route: The normalized route for matching
        chunk_id: ID of the CodeChunk containing this call
        caller_qualified_name: Calling function name
        file_path: Path to the source file
        repo_name: Repository name
        line_number: Line number of the call
        metadata: Additional metadata (headers, body schema, etc.)
    """
    id: str
    method: str
    url: str
    normalized_route: NormalizedRoute
    chunk_id: str
    caller_qualified_name: str
    file_path: str
    repo_name: str
    line_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteMatch:
    """A matched link between an API call and an endpoint.

    Attributes:
        api_call: The frontend API call
        endpoint: The matched backend endpoint
        confidence: Match confidence score (0.0 to 1.0)
        is_ambiguous: Whether multiple endpoints matched
        alternatives: Number of alternative matches (if ambiguous)
        match_details: Breakdown of how confidence was calculated
    """
    api_call: ApiCall
    endpoint: RouteEndpoint
    confidence: float
    is_ambiguous: bool = False
    alternatives: int = 0
    match_details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "from": self.api_call.caller_qualified_name,
            "from_file": self.api_call.file_path,
            "from_line": self.api_call.line_number,
            "to": self.endpoint.qualified_name,
            "to_file": self.endpoint.file_path,
            "method": self.api_call.method,
            "call_url": self.api_call.url,
            "endpoint_path": self.endpoint.path,
            "confidence": self.confidence,
            "is_ambiguous": self.is_ambiguous,
            "alternatives": self.alternatives,
            "match_details": self.match_details,
        }


@dataclass
class RouteConflict:
    """Detected conflict between routes (duplicate or overlapping).

    Attributes:
        endpoints: List of conflicting endpoints
        conflict_type: Type of conflict (duplicate, overlapping)
        description: Human-readable description of the conflict
    """
    endpoints: list[RouteEndpoint]
    conflict_type: str  # "duplicate" | "overlapping"
    description: str


@dataclass
class RouteLinkingResult:
    """Result of a complete route linking analysis.

    Attributes:
        linked: Successfully matched call→endpoint pairs
        ambiguous: Matches requiring manual review
        unlinked_calls: API calls with no matching endpoint
        unlinked_endpoints: Endpoints never called
        conflicts: Detected route conflicts
        statistics: Summary statistics
    """
    linked: list[RouteMatch] = field(default_factory=list)
    ambiguous: list[RouteMatch] = field(default_factory=list)
    unlinked_calls: list[ApiCall] = field(default_factory=list)
    unlinked_endpoints: list[RouteEndpoint] = field(default_factory=list)
    conflicts: list[RouteConflict] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "linked": [m.to_dict() for m in self.linked],
            "ambiguous": [m.to_dict() for m in self.ambiguous],
            "unlinked_calls": [
                {
                    "method": c.method,
                    "url": c.url,
                    "caller": c.caller_qualified_name,
                    "file": c.file_path,
                    "line": c.line_number,
                }
                for c in self.unlinked_calls
            ],
            "unlinked_endpoints": [
                {
                    "method": e.method,
                    "path": e.path,
                    "handler": e.qualified_name,
                    "file": e.file_path,
                }
                for e in self.unlinked_endpoints
            ],
            "conflicts": [
                {
                    "type": c.conflict_type,
                    "description": c.description,
                    "endpoints": [e.qualified_name for e in c.endpoints],
                }
                for c in self.conflicts
            ],
            "statistics": self.statistics,
        }
