"""Route Linker Service for Cross-Language API Linking.

This module provides production-ready functionality to link frontend API calls
to backend API endpoints across different languages and frameworks.

Features:
- Multi-factor matching algorithm with configurable weights
- Parameter name preservation for accurate matching
- Ambiguity detection with manual review flagging
- Support for TypeScript, Python, Java frameworks
- OpenAPI spec fallback support

Scoring Formula:
    final_score = (
        path_structure_score * 0.40 +   # Segment count, static segments match
        param_position_score * 0.25 +   # Parameters in same positions
        param_name_score * 0.20 +       # Parameter name similarity
        context_score * 0.15            # Same repo, same language bonus
    )
"""

import hashlib
from typing import Optional

from ..config import get_config
from ..models.chunk import CodeChunk
from ..models.route import (
    ApiCall,
    NormalizedRoute,
    RouteEndpoint,
    RouteMatch,
    RouteConflict,
    RouteLinkingResult,
)
from .call_graph import CallGraphService, CallRelation
from .route_normalizer import RouteNormalizer
from .storage import StorageService


class RouteMatchingEngine:
    """Engine for matching API calls to endpoints using multi-factor scoring.

    The matching algorithm uses multiple factors to determine match confidence:
    1. Path structure: Static segments must match, segment count must match
    2. Parameter positions: Parameters should be in the same positions
    3. Parameter names: Semantic similarity of parameter names
    4. Context: Same repository or language provides a boost
    """

    def __init__(self):
        self.config = get_config().route_linking
        self.normalizer = RouteNormalizer()

    def match(
        self,
        call: ApiCall,
        endpoint: RouteEndpoint,
    ) -> Optional[RouteMatch]:
        """Attempt to match an API call to an endpoint.

        Args:
            call: The frontend API call
            endpoint: The backend endpoint to match against

        Returns:
            RouteMatch if matched above threshold, None otherwise
        """
        # Gate: HTTP method must match
        if call.method != endpoint.method:
            return None

        call_route = call.normalized_route
        endpoint_route = endpoint.normalized_route

        # Gate: Segment count must match
        if len(call_route.path_segments) != len(endpoint_route.path_segments):
            return None

        # Gate: Static segments must match
        if not self._static_segments_match(call_route, endpoint_route):
            return None

        # Calculate individual scores
        path_score = self._score_path_segments(call_route, endpoint_route)
        param_pos_score = self._score_param_positions(call_route, endpoint_route)
        param_name_score = self._score_param_names(call_route, endpoint_route)
        context_score = self._score_context(call, endpoint)

        # Calculate weighted final score
        final_score = (
            path_score * self.config.weight_path_structure +
            param_pos_score * self.config.weight_param_position +
            param_name_score * self.config.weight_param_name +
            context_score * self.config.weight_context
        )

        # Check threshold
        if final_score < self.config.min_confidence:
            return None

        return RouteMatch(
            api_call=call,
            endpoint=endpoint,
            confidence=final_score,
            is_ambiguous=False,
            match_details={
                "path_structure": path_score,
                "param_position": param_pos_score,
                "param_name": param_name_score,
                "context": context_score,
            },
        )

    def _static_segments_match(
        self,
        call_route: NormalizedRoute,
        endpoint_route: NormalizedRoute,
    ) -> bool:
        """Check if all static segments match between routes."""
        call_static = call_route.get_static_segments()
        endpoint_static = endpoint_route.get_static_segments()

        if len(call_static) != len(endpoint_static):
            return False

        for (pos1, seg1), (pos2, seg2) in zip(call_static, endpoint_static):
            if pos1 != pos2:
                return False
            if seg1.lower() != seg2.lower():
                return False

        return True

    def _score_path_segments(
        self,
        call_route: NormalizedRoute,
        endpoint_route: NormalizedRoute,
    ) -> float:
        """Score based on path structure similarity.

        Perfect score (1.0) when:
        - All static segments match exactly
        - Parameter placeholders are in same positions
        """
        if not call_route.path_segments:
            return 1.0 if not endpoint_route.path_segments else 0.0

        matches = 0
        total = len(call_route.path_segments)

        for i, (call_seg, end_seg) in enumerate(
            zip(call_route.path_segments, endpoint_route.path_segments)
        ):
            call_is_param = call_seg.startswith("{") and call_seg.endswith("}")
            end_is_param = end_seg.startswith("{") and end_seg.endswith("}")

            if call_is_param and end_is_param:
                # Both are parameters - full match
                matches += 1
            elif not call_is_param and not end_is_param:
                # Both are static - must be equal
                if call_seg.lower() == end_seg.lower():
                    matches += 1
            elif end_is_param:
                # Endpoint has param, call has value - this is expected
                matches += 1
            # else: call has param, endpoint has static - mismatch (0)

        return matches / total

    def _score_param_positions(
        self,
        call_route: NormalizedRoute,
        endpoint_route: NormalizedRoute,
    ) -> float:
        """Score based on parameter position matching.

        Parameters should appear in the same positions in both routes.
        """
        call_params = call_route.get_param_segments()
        endpoint_params = endpoint_route.get_param_segments()

        if not endpoint_params:
            # No parameters in endpoint - perfect score if call also has none
            return 1.0 if not call_params else 0.8

        # Count matching positions
        call_positions = {pos for pos, _ in call_params}
        endpoint_positions = {pos for pos, _ in endpoint_params}

        # Perfect match if positions are identical or call has values where endpoint has params
        overlap = endpoint_positions  # Endpoint defines required param positions
        call_has_value_at = set()

        for i, seg in enumerate(call_route.path_segments):
            if not (seg.startswith("{") and seg.endswith("}")):
                call_has_value_at.add(i)

        # Score based on how many endpoint param positions have values in call
        matching = len(overlap.intersection(call_has_value_at))
        matching += len(overlap.intersection(call_positions))

        return min(1.0, matching / len(endpoint_positions)) if endpoint_positions else 1.0

    def _score_param_names(
        self,
        call_route: NormalizedRoute,
        endpoint_route: NormalizedRoute,
    ) -> float:
        """Score based on parameter name similarity.

        Compares parameter names when both routes have named parameters.
        Uses fuzzy matching for different naming conventions.
        """
        call_params = {pos: name for pos, name in call_route.get_param_segments()}
        endpoint_params = {pos: name for pos, name in endpoint_route.get_param_segments()}

        if not endpoint_params:
            return 1.0

        if not call_params:
            # Call has no explicit params (values in URL) - neutral score
            return 0.8

        # Compare parameter names at matching positions
        total = 0
        matches = 0

        for pos, endpoint_name in endpoint_params.items():
            if pos in call_params:
                total += 1
                call_name = call_params[pos]
                if self._param_names_similar(call_name, endpoint_name):
                    matches += 1

        return matches / total if total > 0 else 0.8

    def _param_names_similar(self, name1: str, name2: str) -> bool:
        """Check if two parameter names are similar.

        Handles different naming conventions:
        - userId vs user_id
        - orderId vs order_id
        """
        if name1.lower() == name2.lower():
            return True

        # Normalize both to snake_case
        n1 = self.normalizer._to_snake_case(name1)
        n2 = self.normalizer._to_snake_case(name2)

        return n1 == n2

    def _score_context(
        self,
        call: ApiCall,
        endpoint: RouteEndpoint,
    ) -> float:
        """Score based on contextual factors.

        Provides bonus for:
        - Same repository
        - Compatible languages (TypeScript calling Python is common)
        """
        score = 0.5  # Base score

        # Same repository bonus
        if call.repo_name == endpoint.repo_name:
            score += 0.3

        # Language compatibility bonus
        frontend_langs = {"typescript", "javascript"}
        backend_langs = {"python", "java"}

        call_lang = "typescript"  # API calls typically from TypeScript
        endpoint_lang = endpoint.language.lower()

        # Frontend → Backend is the expected pattern
        if call_lang in frontend_langs and endpoint_lang in backend_langs:
            score += 0.2
        # Same language (full-stack TypeScript)
        elif call_lang in frontend_langs and endpoint_lang in frontend_langs:
            score += 0.2

        return min(1.0, score)

    def find_matches(
        self,
        call: ApiCall,
        endpoints: list[RouteEndpoint],
    ) -> list[RouteMatch]:
        """Find all matching endpoints for an API call.

        Args:
            call: The API call to match
            endpoints: List of candidate endpoints

        Returns:
            List of RouteMatch objects, sorted by confidence
        """
        matches = []

        for endpoint in endpoints:
            match = self.match(call, endpoint)
            if match:
                matches.append(match)

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)

        # Limit number of matches
        matches = matches[:self.config.max_matches_per_call]

        # Flag ambiguity if multiple high-confidence matches
        if len(matches) > 1 and self.config.flag_ambiguous:
            top_score = matches[0].confidence
            for match in matches:
                if top_score - match.confidence <= self.config.ambiguity_threshold:
                    match.is_ambiguous = True
                    match.alternatives = len(matches)

        return matches


class RouteLinker:
    """Service for linking frontend API calls to backend endpoints.

    This service:
    1. Collects all route endpoints from backend code (metadata['route'])
    2. Collects all outgoing API calls from frontend code (metadata['outgoing_api_calls'])
    3. Uses multi-factor matching to link calls to endpoints
    4. Detects ambiguous matches for manual review
    5. Creates synthetic call graph edges for cross-language tracing
    """

    def __init__(
        self,
        storage: Optional[StorageService] = None,
        call_graph: Optional[CallGraphService] = None,
    ):
        self.storage = storage or StorageService()
        self.call_graph = call_graph or CallGraphService()
        self.normalizer = RouteNormalizer()
        self.matching_engine = RouteMatchingEngine()
        self.config = get_config().route_linking

    def _generate_id(self, *parts: str) -> str:
        """Generate a unique ID from parts."""
        content = ":".join(str(p) for p in parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def collect_endpoints(
        self, repo_name: Optional[str] = None
    ) -> list[RouteEndpoint]:
        """Collect all route endpoints from indexed code.

        Scans chunks for route metadata and returns normalized endpoints.

        Args:
            repo_name: Optional repository filter

        Returns:
            List of RouteEndpoint objects
        """
        endpoints = []
        chunk_map = self.storage._load_chunk_metadata()

        for chunk in chunk_map.values():
            if repo_name and chunk.repo_name != repo_name:
                continue

            route = chunk.metadata.get("route")
            if route:
                method = route.get("method", "GET").upper()
                path = route.get("path", "/")
                framework = route.get("framework")

                normalized_route = self.normalizer.normalize(path, method, framework)

                endpoint_id = self._generate_id(
                    chunk.repo_name, chunk.file_path, method, path
                )

                endpoints.append(
                    RouteEndpoint(
                        id=endpoint_id,
                        method=method,
                        normalized_route=normalized_route,
                        chunk_id=chunk.id,
                        qualified_name=chunk.get_qualified_name(),
                        file_path=chunk.file_path,
                        repo_name=chunk.repo_name,
                        language=chunk.language,
                        framework=framework,
                        metadata=route,
                    )
                )

        return endpoints

    def collect_api_calls(
        self, repo_name: Optional[str] = None
    ) -> list[ApiCall]:
        """Collect all outgoing API calls from indexed code.

        Scans chunks for outgoing_api_calls metadata.

        Args:
            repo_name: Optional repository filter

        Returns:
            List of ApiCall objects
        """
        api_calls = []
        chunk_map = self.storage._load_chunk_metadata()

        for chunk in chunk_map.values():
            if repo_name and chunk.repo_name != repo_name:
                continue

            outgoing = chunk.metadata.get("outgoing_api_calls", [])
            for call in outgoing:
                method = call.get("method", "GET").upper()
                url = call.get("url", "")

                if url:
                    normalized_route = self.normalizer.normalize(url, method)

                    call_id = self._generate_id(
                        chunk.repo_name, chunk.file_path, chunk.start_line,
                        method, url
                    )

                    api_calls.append(
                        ApiCall(
                            id=call_id,
                            method=method,
                            url=url,
                            normalized_route=normalized_route,
                            chunk_id=chunk.id,
                            caller_qualified_name=chunk.get_qualified_name(),
                            file_path=chunk.file_path,
                            repo_name=chunk.repo_name,
                            line_number=chunk.start_line,
                            metadata=call,
                        )
                    )

        return api_calls

    def match_routes(
        self,
        api_calls: list[ApiCall],
        endpoints: list[RouteEndpoint],
    ) -> list[RouteMatch]:
        """Match API calls to their corresponding endpoints.

        Uses multi-factor matching algorithm.

        Args:
            api_calls: List of outgoing API calls
            endpoints: List of backend endpoints

        Returns:
            List of RouteMatch objects representing matches
        """
        all_matches = []

        for call in api_calls:
            matches = self.matching_engine.find_matches(call, endpoints)
            all_matches.extend(matches)

        return all_matches

    def find_conflicts(
        self, endpoints: list[RouteEndpoint]
    ) -> list[RouteConflict]:
        """Detect conflicting routes (duplicates or overlapping).

        Args:
            endpoints: List of endpoints to check

        Returns:
            List of RouteConflict objects
        """
        conflicts = []
        seen = {}  # canonical_path -> list of endpoints

        for endpoint in endpoints:
            canonical = self.normalizer.normalize_for_matching(
                endpoint.normalized_route
            )
            key = f"{endpoint.method}:{canonical}"

            if key in seen:
                # Potential conflict
                existing = seen[key]
                existing.append(endpoint)
            else:
                seen[key] = [endpoint]

        # Find actual conflicts
        for key, eps in seen.items():
            if len(eps) > 1:
                # Check if they're truly duplicates (same handler) or conflicts
                handlers = {e.qualified_name for e in eps}
                if len(handlers) > 1:
                    conflicts.append(
                        RouteConflict(
                            endpoints=eps,
                            conflict_type="duplicate",
                            description=f"Multiple handlers for route {key}: {', '.join(handlers)}",
                        )
                    )

        return conflicts

    def create_synthetic_edges(
        self, matches: list[RouteMatch], repo_name: str
    ) -> int:
        """Create synthetic call graph edges for route matches.

        Args:
            matches: List of RouteMatch objects
            repo_name: Repository name for the edges

        Returns:
            Number of edges created
        """
        relations = []

        for match in matches:
            relations.append(
                CallRelation(
                    caller=match.api_call.caller_qualified_name,
                    callee=match.endpoint.qualified_name,
                    caller_file=match.api_call.file_path,
                    caller_line=match.api_call.line_number,
                    repo_name=repo_name,
                    call_type="http_request",
                )
            )

        return self.call_graph.add_calls_bulk(relations)

    def link_all_routes(
        self, repo_name: Optional[str] = None
    ) -> RouteLinkingResult:
        """Run the full route linking process.

        1. Collects all endpoints
        2. Collects all API calls
        3. Matches them using multi-factor algorithm
        4. Detects ambiguous matches and conflicts
        5. Creates synthetic call graph edges

        Args:
            repo_name: Optional repository filter

        Returns:
            RouteLinkingResult with comprehensive analysis
        """
        endpoints = self.collect_endpoints(repo_name)
        api_calls = self.collect_api_calls(repo_name)
        matches = self.match_routes(api_calls, endpoints)
        conflicts = self.find_conflicts(endpoints)

        # Separate linked and ambiguous
        linked = [m for m in matches if not m.is_ambiguous]
        ambiguous = [m for m in matches if m.is_ambiguous]

        # Find unlinked calls (calls with no matches)
        matched_call_ids = {m.api_call.id for m in matches}
        unlinked_calls = [c for c in api_calls if c.id not in matched_call_ids]

        # Find unlinked endpoints (endpoints never matched)
        matched_endpoint_ids = {m.endpoint.id for m in matches}
        unlinked_endpoints = [e for e in endpoints if e.id not in matched_endpoint_ids]

        # Create synthetic edges
        edges_created = 0
        if linked and repo_name:
            edges_created = self.create_synthetic_edges(linked, repo_name)

        result = RouteLinkingResult(
            linked=linked,
            ambiguous=ambiguous,
            unlinked_calls=unlinked_calls,
            unlinked_endpoints=unlinked_endpoints,
            conflicts=conflicts,
            statistics={
                "endpoints_found": len(endpoints),
                "api_calls_found": len(api_calls),
                "links_matched": len(matches),
                "links_confident": len(linked),
                "links_ambiguous": len(ambiguous),
                "unlinked_calls": len(unlinked_calls),
                "unlinked_endpoints": len(unlinked_endpoints),
                "conflicts_detected": len(conflicts),
                "edges_created": edges_created,
            },
        )

        return result

    # Backward compatibility methods

    def normalize_path(self, path: str, framework: Optional[str] = None) -> str:
        """Normalize a route path (backward compatible).

        Deprecated: Use RouteNormalizer directly.
        """
        route = self.normalizer.normalize(path, framework=framework)
        return route.path_pattern
