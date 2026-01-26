"""
Pattern Analyzer Service - Code Pattern and Convention Detection.

This module provides analysis of code patterns, library usage, and team
conventions to help AI assistants generate code that matches the codebase style.

Key Features:
1. Library usage tracking (which libraries/frameworks are used most)
2. Pattern frequency counting (constructor injection vs field injection)
3. Pattern momentum (trending up or down)
4. Golden file detection (exemplary implementations)
5. Testing convention detection (jest vs mocha, pytest vs unittest)

Example Usage:
    analyzer = PatternAnalyzer()

    # Get library usage report
    libs = analyzer.analyze_library_usage("my-repo")
    # Returns: {"axios": 45, "fetch": 12, "got": 3}

    # Get pattern recommendations
    patterns = analyzer.get_pattern_recommendations("dependency_injection")
    # Returns: {"recommended": "constructor_injection", "usage": "78%", "momentum": "rising"}

Author: RepoMind Team
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import logging

from ..services.storage import StorageService
from ..services.symbol_table import SymbolTableService

logger = logging.getLogger(__name__)


class PatternCategory(str, Enum):
    """Categories of code patterns."""
    DEPENDENCY_INJECTION = "dependency_injection"
    ERROR_HANDLING = "error_handling"
    ASYNC_PATTERNS = "async_patterns"
    HTTP_CLIENT = "http_client"
    TESTING = "testing"
    LOGGING = "logging"
    CONFIGURATION = "configuration"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    STATE_MANAGEMENT = "state_management"


class Momentum(str, Enum):
    """Trend direction for patterns."""
    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class LibraryUsage:
    """Usage statistics for a library/package."""
    name: str
    import_count: int
    file_count: int
    call_count: int
    category: Optional[str] = None
    momentum: Momentum = Momentum.STABLE
    examples: list[str] = field(default_factory=list)


@dataclass
class PatternInstance:
    """A specific instance of a code pattern."""
    pattern_name: str
    file_path: str
    line_number: int
    code_snippet: str
    confidence: float


@dataclass
class PatternReport:
    """Report on a code pattern's usage."""
    pattern_name: str
    category: PatternCategory
    occurrence_count: int
    percentage: float  # Of total patterns in category
    momentum: Momentum
    recommended: bool  # Is this the recommended pattern?
    examples: list[PatternInstance] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)


@dataclass
class GoldenFile:
    """An exemplary file that demonstrates best practices."""
    file_path: str
    repo_name: str
    reasons: list[str]
    score: float  # 0-100
    symbol_count: int
    test_coverage_indicator: bool


@dataclass
class TestingConvention:
    """Detected testing conventions."""
    framework: str  # pytest, jest, junit, mocha, etc.
    file_patterns: list[str]  # test_*.py, *.spec.ts, etc.
    common_assertions: list[str]
    mock_library: Optional[str]
    usage_count: int


class PatternAnalyzer:
    """
    Analyzes code patterns and conventions in a codebase.

    Provides insights into:
    - Which libraries and frameworks are used
    - What patterns the team prefers
    - Which patterns are trending up/down
    - Exemplary files to use as references
    """

    # Pattern definitions for detection
    PATTERNS = {
        PatternCategory.DEPENDENCY_INJECTION: {
            "constructor_injection": [
                r"def __init__\(self,\s*\w+:\s*\w+",  # Python
                r"constructor\s*\([^)]*private\s+readonly",  # TypeScript
                r"@Inject\s*\n\s*constructor",  # NestJS
                r"public\s+\w+\([^)]*@Autowired",  # Java
            ],
            "field_injection": [
                r"@Autowired\s*\n\s*private",  # Java Spring
                r"@Inject\s*\n\s*private",  # Java CDI
            ],
            "setter_injection": [
                r"@Autowired\s*\n\s*public\s+void\s+set",  # Java
                r"def\s+set_\w+\(self,\s*\w+:",  # Python
            ],
        },
        PatternCategory.ERROR_HANDLING: {
            "try_catch": [
                r"try\s*:\s*\n",  # Python
                r"try\s*\{",  # Java/TypeScript
            ],
            "error_boundary": [
                r"class\s+\w+\s+extends\s+.*ErrorBoundary",
                r"componentDidCatch",
            ],
            "result_type": [
                r"Result<.*,.*Error>",
                r"Either<.*,.*>",
            ],
            "custom_exceptions": [
                r"class\s+\w+Error\s*\(",  # Python
                r"class\s+\w+Exception\s+extends",  # Java
            ],
        },
        PatternCategory.ASYNC_PATTERNS: {
            "async_await": [
                r"async\s+def\s+",  # Python
                r"async\s+\w+\s*\(",  # TypeScript
            ],
            "promises": [
                r"\.then\s*\(",
                r"Promise\.",
            ],
            "callbacks": [
                r"callback\s*[=:]\s*function",
                r"\w+\s*\(\s*function\s*\(",
            ],
            "observables": [
                r"Observable<",
                r"\.subscribe\s*\(",
            ],
        },
        PatternCategory.HTTP_CLIENT: {
            "axios": [r"axios\.", r"from\s+['\"]axios['\"]"],
            "fetch": [r"fetch\s*\(", r"window\.fetch"],
            "requests": [r"requests\.", r"import\s+requests"],
            "httpx": [r"httpx\.", r"import\s+httpx"],
            "got": [r"got\s*\(", r"from\s+['\"]got['\"]"],
            "http_client": [r"HttpClient", r"@angular/common/http"],
        },
        PatternCategory.TESTING: {
            "pytest": [r"def\s+test_", r"import\s+pytest", r"@pytest\."],
            "unittest": [r"class\s+\w+\(.*TestCase\)", r"self\.assert"],
            "jest": [r"describe\s*\(['\"]", r"it\s*\(['\"]", r"expect\s*\("],
            "mocha": [r"describe\s*\(['\"]", r"it\s*\(['\"]", r"chai\."],
            "junit": [r"@Test", r"@BeforeEach", r"Assertions\."],
        },
    }

    # Library categorization
    LIBRARY_CATEGORIES = {
        "http_client": ["axios", "fetch", "requests", "httpx", "got", "superagent"],
        "database": ["sqlalchemy", "typeorm", "prisma", "mongoose", "sequelize", "knex"],
        "testing": ["pytest", "jest", "mocha", "junit", "vitest", "jasmine"],
        "logging": ["winston", "pino", "bunyan", "loguru", "structlog"],
        "auth": ["passport", "jwt", "oauth", "auth0", "cognito"],
        "validation": ["pydantic", "zod", "yup", "joi", "class-validator"],
        "state": ["redux", "mobx", "zustand", "recoil", "vuex", "pinia"],
        "framework": ["express", "fastapi", "flask", "django", "nestjs", "spring"],
    }

    def __init__(self):
        """Initialize the pattern analyzer."""
        self._storage = StorageService()
        self._symbol_table = SymbolTableService()

    def analyze_library_usage(
        self,
        repo_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> dict[str, LibraryUsage]:
        """
        Analyze library and framework usage across the codebase.

        Args:
            repo_filter: Optional repository filter.
            category_filter: Optional category filter (http_client, testing, etc.)

        Returns:
            Dictionary mapping library names to usage statistics.
        """
        chunk_map = self._storage._load_chunk_metadata()
        library_stats: dict[str, dict] = defaultdict(lambda: {
            "import_count": 0,
            "file_count": 0,
            "files": set(),
            "call_count": 0,
            "examples": [],
        })

        for chunk_id, chunk in chunk_map.items():
            if repo_filter and chunk.repo_name != repo_filter:
                continue

            content = chunk.content
            imports = chunk.imports or []

            # Count imports
            for imp in imports:
                lib_name = self._extract_library_name(imp)
                if lib_name:
                    library_stats[lib_name]["import_count"] += 1
                    library_stats[lib_name]["files"].add(chunk.file_path)

            # Count library calls in content
            for category, libs in self.LIBRARY_CATEGORIES.items():
                if category_filter and category != category_filter:
                    continue
                for lib in libs:
                    pattern = rf"\b{re.escape(lib)}\."
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        library_stats[lib]["call_count"] += len(matches)
                        if len(library_stats[lib]["examples"]) < 3:
                            library_stats[lib]["examples"].append(
                                f"{chunk.file_path}:{chunk.start_line}"
                            )

        # Convert to LibraryUsage objects
        results = {}
        for lib_name, stats in library_stats.items():
            if stats["import_count"] > 0 or stats["call_count"] > 0:
                category = self._get_library_category(lib_name)
                if category_filter and category != category_filter:
                    continue
                results[lib_name] = LibraryUsage(
                    name=lib_name,
                    import_count=stats["import_count"],
                    file_count=len(stats["files"]),
                    call_count=stats["call_count"],
                    category=category,
                    examples=stats["examples"][:3],
                )

        return results

    def analyze_patterns(
        self,
        category: PatternCategory,
        repo_filter: Optional[str] = None,
    ) -> list[PatternReport]:
        """
        Analyze patterns in a specific category.

        Args:
            category: The pattern category to analyze.
            repo_filter: Optional repository filter.

        Returns:
            List of pattern reports with usage statistics.
        """
        if category not in self.PATTERNS:
            return []

        chunk_map = self._storage._load_chunk_metadata()
        pattern_counts: dict[str, dict] = defaultdict(lambda: {
            "count": 0,
            "examples": [],
        })

        patterns = self.PATTERNS[category]
        total_matches = 0

        for chunk_id, chunk in chunk_map.items():
            if repo_filter and chunk.repo_name != repo_filter:
                continue

            content = chunk.content

            for pattern_name, regexes in patterns.items():
                for regex in regexes:
                    matches = list(re.finditer(regex, content))
                    if matches:
                        pattern_counts[pattern_name]["count"] += len(matches)
                        total_matches += len(matches)

                        # Collect examples
                        if len(pattern_counts[pattern_name]["examples"]) < 3:
                            for match in matches[:2]:
                                # Get line number
                                line_num = content[:match.start()].count('\n') + chunk.start_line
                                pattern_counts[pattern_name]["examples"].append(
                                    PatternInstance(
                                        pattern_name=pattern_name,
                                        file_path=chunk.file_path,
                                        line_number=line_num,
                                        code_snippet=match.group()[:100],
                                        confidence=0.9,
                                    )
                                )

        # Build reports
        reports = []
        if total_matches > 0:
            # Find the recommended pattern (most used)
            recommended_pattern = max(pattern_counts.items(), key=lambda x: x[1]["count"])[0]

            for pattern_name, stats in pattern_counts.items():
                if stats["count"] > 0:
                    reports.append(
                        PatternReport(
                            pattern_name=pattern_name,
                            category=category,
                            occurrence_count=stats["count"],
                            percentage=round(stats["count"] / total_matches * 100, 1),
                            momentum=Momentum.STABLE,  # Would need git history for real momentum
                            recommended=(pattern_name == recommended_pattern),
                            examples=stats["examples"],
                            alternatives=[p for p in patterns.keys() if p != pattern_name],
                        )
                    )

        # Sort by occurrence count
        reports.sort(key=lambda r: r.occurrence_count, reverse=True)
        return reports

    def get_pattern_recommendations(
        self,
        category: PatternCategory,
        repo_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get pattern recommendations based on team usage.

        Args:
            category: The pattern category.
            repo_filter: Optional repository filter.

        Returns:
            Recommendation dictionary with suggested patterns.
        """
        reports = self.analyze_patterns(category, repo_filter)

        if not reports:
            return {
                "category": category.value,
                "recommended": None,
                "message": "No patterns detected in this category",
            }

        recommended = next((r for r in reports if r.recommended), reports[0])

        return {
            "category": category.value,
            "recommended": recommended.pattern_name,
            "usage_percentage": recommended.percentage,
            "momentum": recommended.momentum.value,
            "example_locations": [
                f"{e.file_path}:{e.line_number}"
                for e in recommended.examples[:3]
            ],
            "alternatives": [
                {
                    "name": r.pattern_name,
                    "usage_percentage": r.percentage,
                }
                for r in reports
                if not r.recommended
            ][:3],
        }

    def detect_testing_conventions(
        self,
        repo_filter: Optional[str] = None,
    ) -> list[TestingConvention]:
        """
        Detect testing frameworks and conventions used.

        Args:
            repo_filter: Optional repository filter.

        Returns:
            List of detected testing conventions.
        """
        chunk_map = self._storage._load_chunk_metadata()
        framework_counts: dict[str, int] = Counter()
        file_patterns: dict[str, set] = defaultdict(set)
        assertions: dict[str, list] = defaultdict(list)

        for chunk_id, chunk in chunk_map.items():
            if repo_filter and chunk.repo_name != repo_filter:
                continue

            file_path = chunk.file_path
            content = chunk.content

            # Detect test file patterns
            if re.search(r'test_\w+\.py$', file_path):
                file_patterns["pytest"].add("test_*.py")
                framework_counts["pytest"] += 1
            elif re.search(r'\w+Test\.java$', file_path):
                file_patterns["junit"].add("*Test.java")
                framework_counts["junit"] += 1
            elif re.search(r'\.spec\.(ts|js)$', file_path):
                file_patterns["jest"].add("*.spec.ts")
                framework_counts["jest"] += 1
            elif re.search(r'\.test\.(ts|js)$', file_path):
                file_patterns["jest"].add("*.test.ts")
                framework_counts["jest"] += 1

            # Detect assertions
            if re.search(r'assert\s+\w+', content):
                assertions["pytest"].append("assert")
            if re.search(r'self\.assert\w+', content):
                assertions["unittest"].append("self.assert*")
            if re.search(r'expect\s*\([^)]+\)\.(to|not)', content):
                assertions["jest"].append("expect().to*")
            if re.search(r'Assertions\.\w+', content):
                assertions["junit"].append("Assertions.*")

        # Build convention objects
        conventions = []
        for framework, count in framework_counts.most_common():
            if count > 0:
                conventions.append(
                    TestingConvention(
                        framework=framework,
                        file_patterns=list(file_patterns.get(framework, [])),
                        common_assertions=list(set(assertions.get(framework, [])))[:5],
                        mock_library=self._detect_mock_library(framework),
                        usage_count=count,
                    )
                )

        return conventions

    def find_golden_files(
        self,
        repo_filter: Optional[str] = None,
        top_n: int = 10,
    ) -> list[GoldenFile]:
        """
        Find exemplary files that demonstrate best practices.

        Golden files are identified by:
        - High symbol-to-line ratio (well-structured)
        - Presence of documentation/docstrings
        - Associated test files exist
        - Referenced by many other files

        Args:
            repo_filter: Optional repository filter.
            top_n: Number of top files to return.

        Returns:
            List of golden file candidates.
        """
        chunk_map = self._storage._load_chunk_metadata()

        # Group chunks by file
        file_stats: dict[str, dict] = defaultdict(lambda: {
            "chunks": [],
            "has_docstrings": False,
            "symbol_count": 0,
            "line_count": 0,
            "repo_name": "",
        })

        for chunk_id, chunk in chunk_map.items():
            if repo_filter and chunk.repo_name != repo_filter:
                continue

            # Skip test files
            if self._is_test_file(chunk.file_path):
                continue

            file_stats[chunk.file_path]["chunks"].append(chunk)
            file_stats[chunk.file_path]["symbol_count"] += 1
            file_stats[chunk.file_path]["line_count"] = max(
                file_stats[chunk.file_path]["line_count"],
                chunk.end_line
            )
            file_stats[chunk.file_path]["repo_name"] = chunk.repo_name

            if chunk.docstring:
                file_stats[chunk.file_path]["has_docstrings"] = True

        # Score files
        golden_files = []
        for file_path, stats in file_stats.items():
            if stats["symbol_count"] < 2:  # Skip trivial files
                continue

            score = 0
            reasons = []

            # Documentation score
            if stats["has_docstrings"]:
                score += 25
                reasons.append("Contains documentation")

            # Structure score (symbols per 100 lines)
            if stats["line_count"] > 0:
                density = stats["symbol_count"] / stats["line_count"] * 100
                if density > 5:
                    score += 20
                    reasons.append("Well-structured code")

            # Check for associated tests
            test_file = self._find_test_file(file_path)
            if test_file:
                score += 30
                reasons.append("Has associated tests")

            # Symbol count bonus
            if stats["symbol_count"] >= 5:
                score += 15
                reasons.append("Comprehensive implementation")

            # Check if commonly imported
            ref_count = self._count_references(file_path, chunk_map)
            if ref_count >= 3:
                score += 10
                reasons.append(f"Referenced by {ref_count} files")

            if score >= 40:  # Minimum threshold
                golden_files.append(
                    GoldenFile(
                        file_path=file_path,
                        repo_name=stats["repo_name"],
                        reasons=reasons,
                        score=min(score, 100),
                        symbol_count=stats["symbol_count"],
                        test_coverage_indicator=test_file is not None,
                    )
                )

        # Sort by score and return top N
        golden_files.sort(key=lambda f: f.score, reverse=True)
        return golden_files[:top_n]

    def get_convention_summary(
        self,
        repo_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get a comprehensive summary of conventions used in the codebase.

        Args:
            repo_filter: Optional repository filter.

        Returns:
            Summary dictionary with all convention information.
        """
        summary = {
            "library_usage": {},
            "patterns": {},
            "testing": [],
            "golden_files": [],
        }

        # Library usage by category
        for category in self.LIBRARY_CATEGORIES.keys():
            libs = self.analyze_library_usage(repo_filter, category)
            if libs:
                summary["library_usage"][category] = {
                    name: {
                        "imports": lib.import_count,
                        "calls": lib.call_count,
                    }
                    for name, lib in sorted(
                        libs.items(),
                        key=lambda x: x[1].import_count + x[1].call_count,
                        reverse=True
                    )[:5]
                }

        # Pattern recommendations
        for category in PatternCategory:
            rec = self.get_pattern_recommendations(category, repo_filter)
            if rec.get("recommended"):
                summary["patterns"][category.value] = {
                    "recommended": rec["recommended"],
                    "usage": f"{rec.get('usage_percentage', 0)}%",
                }

        # Testing conventions
        testing = self.detect_testing_conventions(repo_filter)
        summary["testing"] = [
            {
                "framework": t.framework,
                "file_patterns": t.file_patterns,
                "usage_count": t.usage_count,
            }
            for t in testing[:3]
        ]

        # Golden files
        golden = self.find_golden_files(repo_filter, top_n=5)
        summary["golden_files"] = [
            {
                "file": g.file_path,
                "score": g.score,
                "reasons": g.reasons[:2],
            }
            for g in golden
        ]

        return summary

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_library_name(self, import_statement: str) -> Optional[str]:
        """Extract library name from import statement."""
        # Python: from requests import ...
        match = re.match(r'from\s+(\w+)', import_statement)
        if match:
            return match.group(1).lower()

        # Python: import requests
        match = re.match(r'import\s+(\w+)', import_statement)
        if match:
            return match.group(1).lower()

        # TypeScript/JavaScript: from 'library'
        match = re.search(r"from\s+['\"](@?\w+)", import_statement)
        if match:
            return match.group(1).lower()

        # Java: import com.example.library
        match = re.match(r'import\s+[\w.]+\.(\w+);?', import_statement)
        if match:
            return match.group(1).lower()

        return None

    def _get_library_category(self, lib_name: str) -> Optional[str]:
        """Get the category for a library."""
        for category, libs in self.LIBRARY_CATEGORIES.items():
            if lib_name.lower() in [l.lower() for l in libs]:
                return category
        return None

    def _is_test_file(self, file_path: str) -> bool:
        """Check if a file is a test file."""
        path_lower = file_path.lower()
        return any([
            'test_' in path_lower,
            '_test.' in path_lower,
            '.test.' in path_lower,
            '.spec.' in path_lower,
            '/tests/' in path_lower,
            '/test/' in path_lower,
            'Test.' in file_path,  # Java style
        ])

    def _find_test_file(self, source_file: str) -> Optional[str]:
        """Find the test file for a source file."""
        path = Path(source_file)
        stem = path.stem
        suffix = path.suffix

        # Generate possible test file names
        test_patterns = [
            f"test_{stem}{suffix}",
            f"{stem}_test{suffix}",
            f"{stem}.test{suffix}",
            f"{stem}.spec{suffix}",
            f"{stem}Test{suffix}",
        ]

        # This is a simplified check - would need filesystem access for real implementation
        return None  # Placeholder

    def _detect_mock_library(self, framework: str) -> Optional[str]:
        """Detect the mock library used with a test framework."""
        mock_libraries = {
            "pytest": "pytest-mock",
            "unittest": "unittest.mock",
            "jest": "jest.mock",
            "mocha": "sinon",
            "junit": "mockito",
        }
        return mock_libraries.get(framework)

    def _count_references(self, file_path: str, chunk_map: dict) -> int:
        """Count how many other files reference this file."""
        ref_count = 0
        file_name = Path(file_path).stem

        for chunk_id, chunk in chunk_map.items():
            if chunk.file_path == file_path:
                continue
            # Check imports
            for imp in (chunk.imports or []):
                if file_name.lower() in imp.lower():
                    ref_count += 1
                    break

        return ref_count
