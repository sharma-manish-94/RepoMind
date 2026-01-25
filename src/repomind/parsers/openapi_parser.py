"""OpenAPI/Swagger Specification Parser.

This module provides parsing of OpenAPI (3.x) and Swagger (2.x) specifications
to extract route endpoints as a fallback source when code parsing is insufficient.

Features:
- Supports OpenAPI 3.0, 3.1, and Swagger 2.0
- Extracts paths, methods, parameters, and request/response schemas
- Integrates with the route registry
- Validates extracted routes against spec

Use Cases:
- Fallback when code parsing misses routes
- Authoritative source for request/response types
- Validation of code-extracted routes against spec
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Any

from ..models.route import (
    NormalizedRoute,
    RouteEndpoint,
    RouteParameter,
    ParameterSource,
)
from ..services.route_normalizer import RouteNormalizer


# Common OpenAPI spec file patterns
OPENAPI_FILE_PATTERNS = [
    "openapi.yaml",
    "openapi.yml",
    "openapi.json",
    "swagger.yaml",
    "swagger.yml",
    "swagger.json",
    "api.yaml",
    "api.yml",
    "api.json",
    "**/openapi*.yaml",
    "**/openapi*.yml",
    "**/openapi*.json",
    "**/swagger*.yaml",
    "**/swagger*.yml",
    "**/swagger*.json",
]


class OpenAPIParser:
    """Parser for OpenAPI and Swagger specifications.

    Extracts route endpoints from OpenAPI/Swagger specs to use as a
    fallback source when code parsing is insufficient.

    Supports:
    - OpenAPI 3.0.x and 3.1.x
    - Swagger 2.0 (OpenAPI 2.0)

    Example:
        >>> parser = OpenAPIParser()
        >>> specs = parser.detect_spec_files(Path("/my/repo"))
        >>> for spec_path in specs:
        ...     endpoints = parser.parse_spec(spec_path, "my-repo")
        ...     print(f"Found {len(endpoints)} endpoints")
    """

    def __init__(self):
        self.normalizer = RouteNormalizer()

    def detect_spec_files(self, repo_path: Path) -> list[Path]:
        """Find OpenAPI/Swagger specification files in a repository.

        Args:
            repo_path: Path to the repository root

        Returns:
            List of paths to spec files found
        """
        spec_files = []

        # Check common locations
        common_locations = [
            repo_path / "openapi.yaml",
            repo_path / "openapi.yml",
            repo_path / "openapi.json",
            repo_path / "swagger.yaml",
            repo_path / "swagger.yml",
            repo_path / "swagger.json",
            repo_path / "api.yaml",
            repo_path / "api.yml",
            repo_path / "api.json",
            repo_path / "docs" / "openapi.yaml",
            repo_path / "docs" / "openapi.yml",
            repo_path / "docs" / "swagger.yaml",
            repo_path / "docs" / "swagger.yml",
            repo_path / "spec" / "openapi.yaml",
            repo_path / "spec" / "openapi.yml",
            repo_path / "api" / "openapi.yaml",
            repo_path / "api" / "openapi.yml",
        ]

        for path in common_locations:
            if path.exists() and path.is_file():
                spec_files.append(path)

        # Also search recursively for any openapi/swagger files
        for pattern in ["**/openapi*.yaml", "**/openapi*.yml", "**/swagger*.yaml", "**/swagger*.yml"]:
            try:
                for match in repo_path.glob(pattern):
                    if match.is_file() and match not in spec_files:
                        # Exclude node_modules and other common non-spec directories
                        if not any(part in str(match) for part in ["node_modules", ".git", "dist", "build"]):
                            spec_files.append(match)
            except Exception:
                continue

        return spec_files

    def parse_spec(
        self,
        spec_path: Path,
        repo_name: str,
    ) -> list[RouteEndpoint]:
        """Parse an OpenAPI/Swagger specification file.

        Args:
            spec_path: Path to the spec file
            repo_name: Repository name for the endpoints

        Returns:
            List of RouteEndpoint objects extracted from the spec
        """
        spec = self._load_spec(spec_path)
        if not spec:
            return []

        # Detect spec version
        version = self._detect_version(spec)
        if not version:
            return []

        # Parse based on version
        if version.startswith("3."):
            return self._parse_openapi_3(spec, spec_path, repo_name)
        elif version.startswith("2."):
            return self._parse_swagger_2(spec, spec_path, repo_name)

        return []

    def _load_spec(self, spec_path: Path) -> Optional[dict]:
        """Load a spec file (YAML or JSON).

        Args:
            spec_path: Path to the spec file

        Returns:
            Parsed spec dict or None on error
        """
        try:
            content = spec_path.read_text(encoding="utf-8")

            if spec_path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml
                    return yaml.safe_load(content)
                except ImportError:
                    # Fallback: try to parse as JSON anyway
                    pass

            return json.loads(content)
        except Exception:
            return None

    def _detect_version(self, spec: dict) -> Optional[str]:
        """Detect the OpenAPI/Swagger version.

        Args:
            spec: Parsed spec dict

        Returns:
            Version string or None
        """
        # OpenAPI 3.x
        if "openapi" in spec:
            return spec["openapi"]

        # Swagger 2.x
        if "swagger" in spec:
            return spec["swagger"]

        return None

    def _parse_openapi_3(
        self,
        spec: dict,
        spec_path: Path,
        repo_name: str,
    ) -> list[RouteEndpoint]:
        """Parse OpenAPI 3.x specification.

        Args:
            spec: Parsed spec dict
            spec_path: Path to the spec file
            repo_name: Repository name

        Returns:
            List of RouteEndpoint objects
        """
        endpoints = []
        paths = spec.get("paths", {})
        servers = spec.get("servers", [])

        # Get base path from servers
        base_path = ""
        if servers:
            server_url = servers[0].get("url", "")
            # Extract path from URL
            if "://" in server_url:
                # Full URL
                parts = server_url.split("/", 3)
                if len(parts) > 3:
                    base_path = "/" + parts[3]
            elif server_url.startswith("/"):
                base_path = server_url

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
                if method not in path_item:
                    continue

                operation = path_item[method]
                if not isinstance(operation, dict):
                    continue

                full_path = base_path + path if base_path else path

                endpoint = self._create_endpoint_from_operation(
                    method=method.upper(),
                    path=full_path,
                    operation=operation,
                    path_item=path_item,
                    spec_path=spec_path,
                    repo_name=repo_name,
                    spec_version="openapi3",
                )
                endpoints.append(endpoint)

        return endpoints

    def _parse_swagger_2(
        self,
        spec: dict,
        spec_path: Path,
        repo_name: str,
    ) -> list[RouteEndpoint]:
        """Parse Swagger 2.x specification.

        Args:
            spec: Parsed spec dict
            spec_path: Path to the spec file
            repo_name: Repository name

        Returns:
            List of RouteEndpoint objects
        """
        endpoints = []
        paths = spec.get("paths", {})
        base_path = spec.get("basePath", "")

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
                if method not in path_item:
                    continue

                operation = path_item[method]
                if not isinstance(operation, dict):
                    continue

                full_path = base_path + path if base_path else path

                endpoint = self._create_endpoint_from_operation(
                    method=method.upper(),
                    path=full_path,
                    operation=operation,
                    path_item=path_item,
                    spec_path=spec_path,
                    repo_name=repo_name,
                    spec_version="swagger2",
                )
                endpoints.append(endpoint)

        return endpoints

    def _create_endpoint_from_operation(
        self,
        method: str,
        path: str,
        operation: dict,
        path_item: dict,
        spec_path: Path,
        repo_name: str,
        spec_version: str,
    ) -> RouteEndpoint:
        """Create a RouteEndpoint from an OpenAPI operation.

        Args:
            method: HTTP method
            path: Route path
            operation: Operation dict from spec
            path_item: Parent path item dict
            spec_path: Path to the spec file
            repo_name: Repository name
            spec_version: Spec version string

        Returns:
            RouteEndpoint object
        """
        # Combine path-level and operation-level parameters
        parameters = self._extract_parameters(
            path_item.get("parameters", []) + operation.get("parameters", [])
        )

        # Create normalized route
        normalized_route = self.normalizer.normalize(path, method)
        normalized_route.parameters = parameters

        # Generate operation ID or use existing
        operation_id = operation.get("operationId", f"{method}_{path}")

        # Generate unique ID
        endpoint_id = self._generate_id(repo_name, str(spec_path), method, path)

        # Build metadata
        metadata = {
            "summary": operation.get("summary", ""),
            "description": operation.get("description", ""),
            "tags": operation.get("tags", []),
            "deprecated": operation.get("deprecated", False),
            "source": "openapi",
            "spec_version": spec_version,
        }

        # Include request body info for OpenAPI 3.x
        if "requestBody" in operation:
            metadata["request_body"] = self._extract_request_body_info(
                operation["requestBody"]
            )

        # Include response info
        if "responses" in operation:
            metadata["responses"] = self._extract_responses_info(
                operation["responses"]
            )

        return RouteEndpoint(
            id=endpoint_id,
            method=method,
            normalized_route=normalized_route,
            chunk_id=f"openapi:{operation_id}",
            qualified_name=operation_id,
            file_path=str(spec_path.relative_to(spec_path.parent.parent) if spec_path.parent.parent.exists() else spec_path.name),
            repo_name=repo_name,
            language="openapi",
            framework="openapi",
            metadata=metadata,
        )

    def _extract_parameters(self, params: list) -> list[RouteParameter]:
        """Extract RouteParameter objects from OpenAPI parameters.

        Args:
            params: List of parameter dicts from spec

        Returns:
            List of RouteParameter objects
        """
        route_params = []

        for param in params:
            if not isinstance(param, dict):
                continue

            name = param.get("name", "")
            if not name:
                continue

            # Determine source
            location = param.get("in", "query")
            source_map = {
                "path": ParameterSource.PATH,
                "query": ParameterSource.QUERY,
                "header": ParameterSource.HEADER,
                "body": ParameterSource.BODY,
                "formData": ParameterSource.BODY,
            }
            source = source_map.get(location, ParameterSource.QUERY)

            # Determine type
            param_type = None
            if "schema" in param:
                param_type = param["schema"].get("type")
            elif "type" in param:
                param_type = param["type"]

            # Position is -1 for non-path params
            position = -1

            route_params.append(RouteParameter(
                name=name,
                position=position,
                source=source,
                type_hint=param_type,
                required=param.get("required", False),
            ))

        return route_params

    def _extract_request_body_info(self, request_body: dict) -> dict:
        """Extract request body information.

        Args:
            request_body: RequestBody dict from spec

        Returns:
            Simplified request body info
        """
        info = {
            "required": request_body.get("required", False),
            "content_types": [],
        }

        content = request_body.get("content", {})
        for content_type in content:
            info["content_types"].append(content_type)

        return info

    def _extract_responses_info(self, responses: dict) -> dict:
        """Extract response information.

        Args:
            responses: Responses dict from spec

        Returns:
            Simplified response info
        """
        info = {}

        for status_code, response in responses.items():
            if not isinstance(response, dict):
                continue

            info[status_code] = {
                "description": response.get("description", ""),
            }

            # Include content types for OpenAPI 3.x
            if "content" in response:
                info[status_code]["content_types"] = list(response["content"].keys())

        return info

    def _generate_id(self, *parts: str) -> str:
        """Generate a unique ID from parts."""
        content = ":".join(str(p) for p in parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def validate_against_spec(
        self,
        endpoints: list[RouteEndpoint],
        spec_path: Path,
    ) -> dict:
        """Validate code-extracted endpoints against an OpenAPI spec.

        Args:
            endpoints: Endpoints extracted from code
            spec_path: Path to the OpenAPI spec

        Returns:
            Validation report with matches and mismatches
        """
        spec_endpoints = self.parse_spec(spec_path, "")
        spec_routes = {
            f"{e.method}:{e.normalized_route.path_pattern}": e
            for e in spec_endpoints
        }

        matching = []
        missing_from_spec = []
        missing_from_code = []

        for endpoint in endpoints:
            key = f"{endpoint.method}:{endpoint.normalized_route.path_pattern}"
            if key in spec_routes:
                matching.append({
                    "code_endpoint": endpoint.qualified_name,
                    "spec_endpoint": spec_routes[key].qualified_name,
                    "path": endpoint.path,
                })
                del spec_routes[key]
            else:
                missing_from_spec.append({
                    "endpoint": endpoint.qualified_name,
                    "path": endpoint.path,
                    "method": endpoint.method,
                })

        # Remaining spec routes are missing from code
        for key, spec_ep in spec_routes.items():
            missing_from_code.append({
                "operation_id": spec_ep.qualified_name,
                "path": spec_ep.path,
                "method": spec_ep.method,
            })

        return {
            "matching": matching,
            "missing_from_spec": missing_from_spec,
            "missing_from_code": missing_from_code,
            "summary": {
                "total_code_endpoints": len(endpoints),
                "total_spec_endpoints": len(spec_endpoints),
                "matching_count": len(matching),
                "coverage_percent": round(
                    len(matching) / len(spec_endpoints) * 100
                    if spec_endpoints else 0,
                    1
                ),
            },
        }
