"""Route Normalizer Service for Path Normalization and Parameter Extraction.

This module provides comprehensive path normalization for cross-language API linking:

- Converts various parameter syntaxes to a unified format
- Preserves parameter names for accurate matching
- Handles base URLs and query parameters
- Supports multiple frameworks (FastAPI, Spring, Express, NestJS)

Key features:
- Parameter name preservation: {userId} stays as {userId}, not :param
- Multiple syntax support: {id}, :id, ${id}, <id>
- Base URL extraction: https://api.example.com/users -> /users
- Query parameter handling: /users?limit=10 -> query params extracted separately
"""

import re
from typing import Optional
from urllib.parse import urlparse, parse_qs

from ..models.route import (
    NormalizedRoute,
    RouteParameter,
    ParameterSource,
)
from ..config import get_config


class RouteNormalizer:
    """Service for normalizing route paths across different frameworks.

    Converts various path parameter syntaxes to a unified format while
    preserving semantic information for accurate matching.

    Supported frameworks:
    - FastAPI/Python: /users/{user_id}
    - Spring/Java: /users/{userId}
    - Express/Node: /users/:id
    - NestJS: /users/:id
    - Template literals: /users/${id}
    - OpenAPI: /users/{id}

    Example:
        >>> normalizer = RouteNormalizer()
        >>> route = normalizer.normalize("/users/{userId}/orders", "GET")
        >>> route.path_pattern
        '/users/{userId}/orders'
        >>> route.get_param_segments()
        [(1, 'userId')]
    """

    # Regex patterns for different parameter syntaxes
    PARAM_PATTERNS = {
        "curly": re.compile(r'\{([^}]+)\}'),           # {userId}
        "colon": re.compile(r':([a-zA-Z_][a-zA-Z0-9_]*)'),  # :userId
        "template": re.compile(r'\$\{([^}]+)\}'),      # ${userId}
        "angle": re.compile(r'<([^>]+)>'),             # <userId>
    }

    # Common path type annotations (Spring, FastAPI)
    TYPE_ANNOTATIONS = re.compile(r':(?:int|str|string|path|uuid|float)\b', re.IGNORECASE)

    def __init__(self):
        self.config = get_config().route_linking

    def normalize(
        self,
        path: str,
        method: str = "GET",
        framework: Optional[str] = None,
    ) -> NormalizedRoute:
        """Normalize a route path to a unified format.

        Args:
            path: The original path string
            method: HTTP method (GET, POST, etc.)
            framework: Optional framework hint

        Returns:
            NormalizedRoute with normalized path and extracted parameters
        """
        if not path:
            return NormalizedRoute(
                method=method.upper(),
                path_pattern="/",
                path_segments=[],
                original_path="/",
            )

        original_path = path.strip()

        # Extract base URL if present
        base_path, path_only = self.extract_base_url(original_path)

        # Extract query parameters
        path_only, query_params = self._extract_query_params(path_only)

        # Normalize the path format
        normalized_path, parameters = self._normalize_path_segments(path_only)

        # Build segments list
        segments = self._build_segments(normalized_path)

        return NormalizedRoute(
            method=method.upper(),
            path_pattern=normalized_path,
            path_segments=segments,
            parameters=parameters,
            base_path=base_path if base_path else None,
            query_params=query_params,
            framework=framework,
            original_path=original_path,
        )

    def extract_base_url(self, url: str) -> tuple[str, str]:
        """Split a URL into base URL and path.

        Args:
            url: Full URL or path

        Returns:
            Tuple of (base_url, path)

        Example:
            >>> normalizer.extract_base_url("https://api.example.com/api/v1/users")
            ('https://api.example.com/api/v1', '/users')
        """
        if not url:
            return ("", "/")

        # Handle full URLs
        if url.startswith(("http://", "https://")):
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            path = parsed.path or "/"

            # Check for common API versioning patterns in path
            api_prefix_match = re.match(r'^(/api(?:/v\d+)?)', path)
            if api_prefix_match:
                base = base + api_prefix_match.group(1)
                path = path[len(api_prefix_match.group(1)):] or "/"

            return (base, path)

        # Handle paths with API prefix
        for pattern in self.config.base_url_patterns:
            match = re.match(pattern, url)
            if match:
                base = match.group(0)
                remaining = url[len(base):]
                return (base, remaining or "/")

        return ("", url)

    def _extract_query_params(self, path: str) -> tuple[str, list[str]]:
        """Extract query parameters from a path.

        Args:
            path: Path potentially containing query string

        Returns:
            Tuple of (path_without_query, query_param_names)
        """
        if "?" not in path:
            return (path, [])

        path_part, query_string = path.split("?", 1)
        query_params = []

        # Parse query string
        try:
            parsed = parse_qs(query_string, keep_blank_values=True)
            query_params = list(parsed.keys())
        except ValueError:
            # Handle template variables in query string
            # e.g., ?limit=${limit}&offset=${offset}
            for pattern in self.PARAM_PATTERNS.values():
                matches = pattern.findall(query_string)
                query_params.extend(matches)

        return (path_part, query_params)

    def _normalize_path_segments(
        self, path: str
    ) -> tuple[str, list[RouteParameter]]:
        """Normalize path segments and extract parameters.

        Converts all parameter syntaxes to {paramName} format while
        preserving the original parameter names.

        Args:
            path: Path to normalize

        Returns:
            Tuple of (normalized_path, parameters)
        """
        if not path:
            return ("/", [])

        # Ensure leading slash
        if not path.startswith("/"):
            path = "/" + path

        # Remove trailing slash (except for root)
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]

        # Normalize multiple slashes
        path = re.sub(r'/+', '/', path)

        # Remove type annotations from path (Spring: {id:int})
        path = self.TYPE_ANNOTATIONS.sub('', path)

        parameters = []
        normalized_parts = []

        segments = path.split("/")
        for i, segment in enumerate(segments):
            if not segment:
                continue

            normalized_segment, params = self._normalize_segment(segment, i - 1)
            normalized_parts.append(normalized_segment)
            parameters.extend(params)

        normalized_path = "/" + "/".join(normalized_parts) if normalized_parts else "/"
        return (normalized_path, parameters)

    def _normalize_segment(
        self, segment: str, position: int
    ) -> tuple[str, list[RouteParameter]]:
        """Normalize a single path segment.

        Args:
            segment: Path segment to normalize
            position: Position in path (0-indexed)

        Returns:
            Tuple of (normalized_segment, parameters)
        """
        parameters = []

        # Check each parameter pattern
        for syntax, pattern in self.PARAM_PATTERNS.items():
            match = pattern.fullmatch(segment)
            if match:
                param_name = match.group(1)
                # Remove any type annotation (e.g., "id:int" -> "id")
                if ":" in param_name:
                    param_name = param_name.split(":")[0]
                param_name = param_name.strip()

                parameters.append(
                    RouteParameter(
                        name=param_name,
                        position=position,
                        source=ParameterSource.PATH,
                    )
                )
                return (f"{{{param_name}}}", parameters)

            # Check for embedded parameters (e.g., "users-{id}" or "file.{ext}")
            matches = list(pattern.finditer(segment))
            if matches:
                normalized = segment
                for m in matches:
                    param_name = m.group(1)
                    if ":" in param_name:
                        param_name = param_name.split(":")[0]
                    param_name = param_name.strip()

                    # Replace with unified format
                    if syntax == "colon":
                        normalized = normalized.replace(m.group(0), f"{{{param_name}}}")
                    elif syntax == "template":
                        normalized = normalized.replace(m.group(0), f"{{{param_name}}}")
                    elif syntax == "angle":
                        normalized = normalized.replace(m.group(0), f"{{{param_name}}}")

                    parameters.append(
                        RouteParameter(
                            name=param_name,
                            position=position,
                            source=ParameterSource.PATH,
                        )
                    )

                if parameters:
                    return (normalized, parameters)

        # No parameters found, return as-is (lowercased for consistency)
        return (segment.lower(), parameters)

    def _build_segments(self, path: str) -> list[str]:
        """Build a list of path segments from a normalized path.

        Args:
            path: Normalized path

        Returns:
            List of segments (including parameter placeholders)
        """
        if not path or path == "/":
            return []

        return [s for s in path.split("/") if s]

    def normalize_for_matching(self, route: NormalizedRoute) -> str:
        """Convert a route to a canonical form optimized for comparison.

        This creates a simplified representation where all parameters
        are replaced with a generic placeholder for structural matching.

        Args:
            route: NormalizedRoute to convert

        Returns:
            Canonical path string for comparison
        """
        canonical_segments = []

        for segment in route.path_segments:
            if segment.startswith("{") and segment.endswith("}"):
                canonical_segments.append("{}")  # Generic placeholder
            else:
                canonical_segments.append(segment.lower())

        return "/" + "/".join(canonical_segments) if canonical_segments else "/"

    def extract_parameters(self, path: str) -> list[RouteParameter]:
        """Extract all parameters from a path.

        Args:
            path: Path string with parameters

        Returns:
            List of RouteParameter objects
        """
        route = self.normalize(path)
        return route.parameters

    def paths_match(
        self,
        path1: str,
        path2: str,
        strict: bool = False,
    ) -> bool:
        """Check if two paths could match the same route.

        Args:
            path1: First path
            path2: Second path
            strict: If True, parameter names must also match

        Returns:
            True if paths are structurally compatible
        """
        route1 = self.normalize(path1)
        route2 = self.normalize(path2)

        if len(route1.path_segments) != len(route2.path_segments):
            return False

        for seg1, seg2 in zip(route1.path_segments, route2.path_segments):
            is_param1 = seg1.startswith("{") and seg1.endswith("}")
            is_param2 = seg2.startswith("{") and seg2.endswith("}")

            # Both are static segments - must match exactly
            if not is_param1 and not is_param2:
                if seg1.lower() != seg2.lower():
                    return False
            # One is static, one is param - always matches
            elif is_param1 != is_param2:
                continue
            # Both are params
            else:
                if strict:
                    # Parameter names must match
                    name1 = seg1.strip("{}")
                    name2 = seg2.strip("{}")
                    if not self._param_names_match(name1, name2):
                        return False

        return True

    def _param_names_match(self, name1: str, name2: str) -> bool:
        """Check if two parameter names are semantically equivalent.

        Handles different naming conventions:
        - user_id vs userId vs userID
        - id vs Id vs ID

        Args:
            name1: First parameter name
            name2: Second parameter name

        Returns:
            True if names are considered equivalent
        """
        # Exact match
        if name1.lower() == name2.lower():
            return True

        # Normalize to snake_case and compare
        normalized1 = self._to_snake_case(name1)
        normalized2 = self._to_snake_case(name2)

        return normalized1 == normalized2

    def _to_snake_case(self, name: str) -> str:
        """Convert a name to snake_case.

        Args:
            name: Input name in any case convention

        Returns:
            snake_case version
        """
        # Insert underscore before uppercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        # Insert underscore before uppercase followed by lowercase
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower().replace("-", "_")
