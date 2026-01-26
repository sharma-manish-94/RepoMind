"""
Complexity Metrics Service - Code Quality and Complexity Analysis.

This module provides various code complexity and quality metrics:
1. Cyclomatic complexity (decision paths)
2. Cognitive complexity (human readability)
3. Lines of code metrics (LOC, SLOC, comment ratio)
4. Maintainability index
5. Halstead metrics (volume, difficulty)
6. Function/method size analysis

Example Usage:
    metrics_service = MetricsService()

    # Analyze a single function
    result = metrics_service.analyze_function(code, "python")

    # Analyze all indexed code
    report = metrics_service.analyze_repository("my-repo")

Author: RepoMind Team
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import math
import logging

logger = logging.getLogger(__name__)


class ComplexityRating(str, Enum):
    """Complexity rating for code quality."""
    LOW = "low"           # 1-5: Simple
    MODERATE = "moderate"  # 6-10: Moderate
    HIGH = "high"         # 11-20: Complex
    VERY_HIGH = "very_high"  # 21+: Very complex, consider refactoring


@dataclass
class CyclomaticResult:
    """Result of cyclomatic complexity analysis."""
    complexity: int
    rating: ComplexityRating
    decision_points: list[str]  # Types of decisions found


@dataclass
class CognitiveResult:
    """Result of cognitive complexity analysis."""
    complexity: int
    rating: ComplexityRating
    nesting_depth_max: int
    nested_penalties: int  # Extra cost from nesting


@dataclass
class LOCMetrics:
    """Lines of code metrics."""
    total_lines: int
    source_lines: int  # Non-blank, non-comment
    comment_lines: int
    blank_lines: int
    comment_ratio: float  # comment_lines / source_lines


@dataclass
class HalsteadMetrics:
    """Halstead complexity metrics."""
    distinct_operators: int
    distinct_operands: int
    total_operators: int
    total_operands: int
    vocabulary: int        # n = n1 + n2
    length: int            # N = N1 + N2
    volume: float          # V = N * log2(n)
    difficulty: float      # D = (n1/2) * (N2/n2)
    effort: float          # E = D * V


@dataclass
class FunctionMetrics:
    """Complete metrics for a function/method."""
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    cyclomatic: CyclomaticResult
    cognitive: CognitiveResult
    loc: LOCMetrics
    halstead: Optional[HalsteadMetrics] = None
    parameter_count: int = 0
    return_count: int = 0
    is_complex: bool = False  # True if any metric exceeds thresholds


@dataclass
class FileMetrics:
    """Metrics for a single file."""
    file_path: str
    repo_name: str
    language: str
    loc: LOCMetrics
    function_count: int
    class_count: int
    avg_cyclomatic: float
    max_cyclomatic: int
    avg_cognitive: float
    max_cognitive: int
    complex_functions: list[str]  # Names of functions exceeding thresholds
    maintainability_index: float  # 0-100 scale


@dataclass
class RepositoryMetrics:
    """Metrics for an entire repository."""
    repo_name: str
    total_files: int
    total_functions: int
    total_classes: int
    loc: LOCMetrics
    avg_cyclomatic: float
    avg_cognitive: float
    complex_function_count: int
    complex_function_percentage: float
    maintainability_index: float
    hotspots: list[dict[str, Any]]  # Most complex files/functions
    language_breakdown: dict[str, int]


# Complexity thresholds
CYCLOMATIC_THRESHOLD = 10
COGNITIVE_THRESHOLD = 15
FUNCTION_LOC_THRESHOLD = 50
PARAMETER_THRESHOLD = 5


class MetricsService:
    """
    Service for calculating code complexity and quality metrics.

    Analyzes code chunks from the storage service and provides
    per-function, per-file, and per-repository metrics.
    """

    # Decision point patterns by language
    DECISION_PATTERNS = {
        "python": {
            "if": r'\bif\b(?!\s*__name__)',
            "elif": r'\belif\b',
            "else": r'\belse\b:',
            "for": r'\bfor\b',
            "while": r'\bwhile\b',
            "except": r'\bexcept\b',
            "and": r'\band\b',
            "or": r'\bor\b',
            "assert": r'\bassert\b',
            "comprehension_if": r'\bif\b.*\bfor\b',
            "ternary": r'\bif\b.*\belse\b(?!:)',
            "with": r'\bwith\b',
        },
        "java": {
            "if": r'\bif\s*\(',
            "else if": r'\belse\s+if\s*\(',
            "else": r'\belse\s*\{',
            "for": r'\bfor\s*\(',
            "while": r'\bwhile\s*\(',
            "catch": r'\bcatch\s*\(',
            "case": r'\bcase\b',
            "&&": r'&&',
            "||": r'\|\|',
            "?:": r'\?[^?]',
            "switch": r'\bswitch\s*\(',
        },
        "typescript": {
            "if": r'\bif\s*\(',
            "else if": r'\belse\s+if\s*\(',
            "else": r'\belse\s*\{',
            "for": r'\bfor\s*\(',
            "while": r'\bwhile\s*\(',
            "catch": r'\bcatch\s*\(',
            "case": r'\bcase\b',
            "&&": r'&&',
            "||": r'\|\|',
            "?:": r'\?[^?]',
            "?.": r'\?\.',
            "??": r'\?\?',
            "switch": r'\bswitch\s*\(',
        },
        "javascript": {},  # Same as typescript (will copy)
    }

    def __init__(self):
        """Initialize the metrics service."""
        # Copy typescript patterns to javascript
        self.DECISION_PATTERNS["javascript"] = self.DECISION_PATTERNS["typescript"]

    def calculate_cyclomatic_complexity(
        self,
        code: str,
        language: str = "python",
    ) -> CyclomaticResult:
        """
        Calculate cyclomatic complexity for a code fragment.

        Cyclomatic complexity = number of linearly independent paths.
        M = E - N + 2P (edges - nodes + 2*connected_components)

        Simplified: 1 + number of decision points.

        Args:
            code: Source code to analyze.
            language: Programming language.

        Returns:
            CyclomaticResult with complexity score and rating.
        """
        complexity = 1  # Base path
        decision_points = []

        patterns = self.DECISION_PATTERNS.get(language, self.DECISION_PATTERNS.get("python", {}))

        for decision_type, pattern in patterns.items():
            matches = re.findall(pattern, code)
            count = len(matches)
            if count > 0:
                complexity += count
                decision_points.extend([decision_type] * count)

        return CyclomaticResult(
            complexity=complexity,
            rating=self._rate_cyclomatic(complexity),
            decision_points=decision_points,
        )

    def calculate_cognitive_complexity(
        self,
        code: str,
        language: str = "python",
    ) -> CognitiveResult:
        """
        Calculate cognitive complexity (Sonar-style).

        Cognitive complexity penalizes:
        1. Breaks in linear flow (+1 for each)
        2. Nesting (+1 for each nesting level)
        3. Nested complexity (nesting * break penalty)

        Args:
            code: Source code to analyze.
            language: Programming language.

        Returns:
            CognitiveResult with complexity score.
        """
        complexity = 0
        max_nesting = 0
        current_nesting = 0
        nested_penalties = 0

        lines = code.split('\n')

        # Track nesting by indentation
        base_indent = self._get_base_indent(lines)

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(('#', '//', '/*', '*')):
                continue

            # Calculate nesting level
            indent = len(line) - len(line.lstrip())
            nesting_level = max(0, (indent - base_indent) // 4)  # Assume 4-space indent

            max_nesting = max(max_nesting, nesting_level)

            # Check for flow-breaking constructs
            is_flow_break = False

            if language == "python":
                if re.match(r'\s*(if|elif|for|while|except|with)\b', line):
                    is_flow_break = True
                elif re.match(r'\s*(and|or)\b', line):
                    complexity += 1  # Boolean operators add 1
            else:
                if re.match(r'\s*(if|else\s+if|for|while|catch|switch)\s*[\({]', line):
                    is_flow_break = True
                elif '&&' in line or '||' in line:
                    complexity += line.count('&&') + line.count('||')

            if is_flow_break:
                # Base increment
                complexity += 1

                # Nesting penalty (extra cost for nested structures)
                if nesting_level > 1:
                    nesting_penalty = nesting_level - 1
                    complexity += nesting_penalty
                    nested_penalties += nesting_penalty

        return CognitiveResult(
            complexity=complexity,
            rating=self._rate_cognitive(complexity),
            nesting_depth_max=max_nesting,
            nested_penalties=nested_penalties,
        )

    def calculate_loc_metrics(self, code: str, language: str = "python") -> LOCMetrics:
        """
        Calculate lines of code metrics.

        Args:
            code: Source code to analyze.
            language: Programming language.

        Returns:
            LOCMetrics with line count breakdown.
        """
        lines = code.split('\n')
        total = len(lines)
        blank = 0
        comment = 0
        source = 0

        in_multiline_comment = False

        for line in lines:
            stripped = line.strip()

            # Handle multi-line comments
            if language == "python":
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if in_multiline_comment:
                        in_multiline_comment = False
                        comment += 1
                        continue
                    elif stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                        comment += 1
                        continue
                    else:
                        in_multiline_comment = True
                        comment += 1
                        continue
            else:
                if '/*' in stripped:
                    in_multiline_comment = True
                if '*/' in stripped:
                    in_multiline_comment = False
                    comment += 1
                    continue

            if in_multiline_comment:
                comment += 1
                continue

            # Categorize line
            if not stripped:
                blank += 1
            elif stripped.startswith(('#', '//')):
                comment += 1
            else:
                source += 1

        comment_ratio = comment / source if source > 0 else 0

        return LOCMetrics(
            total_lines=total,
            source_lines=source,
            comment_lines=comment,
            blank_lines=blank,
            comment_ratio=round(comment_ratio, 3),
        )

    def calculate_halstead(self, code: str, language: str = "python") -> HalsteadMetrics:
        """
        Calculate Halstead complexity metrics.

        Args:
            code: Source code to analyze.
            language: Programming language.

        Returns:
            HalsteadMetrics with volume, difficulty, and effort.
        """
        operators = set()
        operands = set()
        total_operators = 0
        total_operands = 0

        # Define operators by language
        if language == "python":
            op_patterns = [
                r'[+\-*/%]={0,1}', r'[<>=!]=', r'[<>]', r'\b(and|or|not|in|is)\b',
                r'\b(def|class|if|elif|else|for|while|return|import|from|try|except|finally|with|yield|raise|pass|break|continue)\b',
                r'[(\[{}\])]', r'[:,;.]', r'->',
            ]
        else:
            op_patterns = [
                r'[+\-*/%]={0,1}', r'[<>=!]=', r'[<>]', r'&&', r'\|\|', r'!',
                r'\b(if|else|for|while|return|import|try|catch|finally|switch|case|break|continue|throw|new|delete|typeof|instanceof)\b',
                r'[(\[{}\])]', r'[:,;.]', r'=>', r'\?[.:]?',
            ]

        # Count operators
        for pattern in op_patterns:
            matches = re.findall(pattern, code)
            for m in matches:
                operators.add(m)
                total_operators += 1

        # Count operands (identifiers and literals)
        identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
        literals = re.findall(r'\b\d+\.?\d*\b|"[^"]*"|\'[^\']*\'', code)

        for ident in identifiers:
            operands.add(ident)
            total_operands += 1
        for lit in literals:
            operands.add(lit)
            total_operands += 1

        n1 = len(operators)   # distinct operators
        n2 = max(len(operands), 1)  # distinct operands (avoid div by 0)
        big_n1 = total_operators
        big_n2 = total_operands

        vocabulary = n1 + n2
        length = big_n1 + big_n2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (big_n2 / n2) if n2 > 0 else 0
        effort = difficulty * volume

        return HalsteadMetrics(
            distinct_operators=n1,
            distinct_operands=n2,
            total_operators=big_n1,
            total_operands=big_n2,
            vocabulary=vocabulary,
            length=length,
            volume=round(volume, 2),
            difficulty=round(difficulty, 2),
            effort=round(effort, 2),
        )

    def analyze_function(
        self,
        code: str,
        language: str = "python",
        name: str = "<unknown>",
        qualified_name: str = "<unknown>",
        file_path: str = "<unknown>",
        start_line: int = 0,
        end_line: int = 0,
    ) -> FunctionMetrics:
        """
        Analyze all metrics for a single function.

        Args:
            code: The function source code.
            language: Programming language.
            name: Function name.
            qualified_name: Fully qualified name.
            file_path: File path.
            start_line: Start line number.
            end_line: End line number.

        Returns:
            Complete FunctionMetrics.
        """
        cyclomatic = self.calculate_cyclomatic_complexity(code, language)
        cognitive = self.calculate_cognitive_complexity(code, language)
        loc = self.calculate_loc_metrics(code, language)

        # Count parameters
        param_count = self._count_parameters(code, language)
        return_count = len(re.findall(r'\breturn\b', code))

        # Determine if function is complex
        is_complex = (
            cyclomatic.complexity > CYCLOMATIC_THRESHOLD
            or cognitive.complexity > COGNITIVE_THRESHOLD
            or loc.source_lines > FUNCTION_LOC_THRESHOLD
            or param_count > PARAMETER_THRESHOLD
        )

        return FunctionMetrics(
            name=name,
            qualified_name=qualified_name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            language=language,
            cyclomatic=cyclomatic,
            cognitive=cognitive,
            loc=loc,
            parameter_count=param_count,
            return_count=return_count,
            is_complex=is_complex,
        )

    def analyze_repository(
        self,
        repo_filter: Optional[str] = None,
    ) -> RepositoryMetrics:
        """
        Analyze metrics for all indexed code in a repository.

        Args:
            repo_filter: Optional repository filter.

        Returns:
            RepositoryMetrics with aggregated statistics.
        """
        from ..services.storage import StorageService
        from ..models.chunk import ChunkType

        storage = StorageService()
        chunk_map = storage._load_chunk_metadata()

        function_metrics: list[FunctionMetrics] = []
        file_metrics_map: dict[str, dict] = defaultdict(lambda: {
            "functions": [],
            "classes": 0,
            "language": "",
            "repo": "",
        })

        total_loc = LOCMetrics(0, 0, 0, 0, 0)
        language_counts: dict[str, int] = Counter()

        for chunk_id, chunk in chunk_map.items():
            if repo_filter and chunk.repo_name != repo_filter:
                continue

            language_counts[chunk.language] += 1

            # Analyze functions and methods
            if chunk.chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.CONSTRUCTOR):
                fm = self.analyze_function(
                    code=chunk.content,
                    language=chunk.language,
                    name=chunk.name,
                    qualified_name=chunk.get_qualified_name(),
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
                function_metrics.append(fm)
                file_metrics_map[chunk.file_path]["functions"].append(fm)
                file_metrics_map[chunk.file_path]["language"] = chunk.language
                file_metrics_map[chunk.file_path]["repo"] = chunk.repo_name

            elif chunk.chunk_type in (ChunkType.CLASS, ChunkType.INTERFACE):
                file_metrics_map[chunk.file_path]["classes"] += 1

            # Accumulate LOC
            loc = self.calculate_loc_metrics(chunk.content, chunk.language)
            total_loc = LOCMetrics(
                total_lines=total_loc.total_lines + loc.total_lines,
                source_lines=total_loc.source_lines + loc.source_lines,
                comment_lines=total_loc.comment_lines + loc.comment_lines,
                blank_lines=total_loc.blank_lines + loc.blank_lines,
                comment_ratio=0,  # Recalculate below
            )

        # Recalculate comment ratio
        if total_loc.source_lines > 0:
            total_loc = LOCMetrics(
                total_lines=total_loc.total_lines,
                source_lines=total_loc.source_lines,
                comment_lines=total_loc.comment_lines,
                blank_lines=total_loc.blank_lines,
                comment_ratio=round(total_loc.comment_lines / total_loc.source_lines, 3),
            )

        # Calculate averages
        complex_functions = [fm for fm in function_metrics if fm.is_complex]
        avg_cyclomatic = (
            sum(fm.cyclomatic.complexity for fm in function_metrics) / len(function_metrics)
            if function_metrics else 0
        )
        avg_cognitive = (
            sum(fm.cognitive.complexity for fm in function_metrics) / len(function_metrics)
            if function_metrics else 0
        )

        # Find hotspots
        hotspots = sorted(function_metrics, key=lambda fm: fm.cyclomatic.complexity, reverse=True)[:10]
        hotspot_data = [
            {
                "name": fm.qualified_name,
                "file": f"{fm.file_path}:{fm.start_line}",
                "cyclomatic": fm.cyclomatic.complexity,
                "cognitive": fm.cognitive.complexity,
                "lines": fm.loc.source_lines,
                "rating": fm.cyclomatic.rating.value,
            }
            for fm in hotspots
        ]

        # Calculate maintainability index
        # MI = MAX(0, (171 - 5.2*ln(V) - 0.23*CC - 16.2*ln(LOC)) * 100/171)
        avg_volume = (
            sum(self.calculate_halstead(fm.name).volume for fm in function_metrics[:100]) / min(len(function_metrics), 100)
            if function_metrics else 0
        )
        avg_loc = total_loc.source_lines / len(file_metrics_map) if file_metrics_map else 1
        mi = max(0, (171 - 5.2 * math.log(max(avg_volume, 1)) - 0.23 * avg_cyclomatic - 16.2 * math.log(max(avg_loc, 1))) * 100 / 171)

        return RepositoryMetrics(
            repo_name=repo_filter or "all",
            total_files=len(file_metrics_map),
            total_functions=len(function_metrics),
            total_classes=sum(fm["classes"] for fm in file_metrics_map.values()),
            loc=total_loc,
            avg_cyclomatic=round(avg_cyclomatic, 2),
            avg_cognitive=round(avg_cognitive, 2),
            complex_function_count=len(complex_functions),
            complex_function_percentage=round(
                len(complex_functions) / len(function_metrics) * 100
                if function_metrics else 0, 1
            ),
            maintainability_index=round(mi, 1),
            hotspots=hotspot_data,
            language_breakdown=dict(language_counts),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _rate_cyclomatic(self, complexity: int) -> ComplexityRating:
        """Rate cyclomatic complexity."""
        if complexity <= 5:
            return ComplexityRating.LOW
        elif complexity <= 10:
            return ComplexityRating.MODERATE
        elif complexity <= 20:
            return ComplexityRating.HIGH
        else:
            return ComplexityRating.VERY_HIGH

    def _rate_cognitive(self, complexity: int) -> ComplexityRating:
        """Rate cognitive complexity."""
        if complexity <= 5:
            return ComplexityRating.LOW
        elif complexity <= 15:
            return ComplexityRating.MODERATE
        elif complexity <= 25:
            return ComplexityRating.HIGH
        else:
            return ComplexityRating.VERY_HIGH

    def _get_base_indent(self, lines: list[str]) -> int:
        """Get the base indentation level."""
        for line in lines:
            stripped = line.lstrip()
            if stripped and not stripped.startswith(('#', '//', '/*', '*')):
                return len(line) - len(stripped)
        return 0

    def _count_parameters(self, code: str, language: str) -> int:
        """Count the number of parameters in a function."""
        # Find the parameter list
        if language == "python":
            match = re.search(r'def\s+\w+\s*\(([^)]*)\)', code)
        else:
            match = re.search(r'(?:function|(?:async\s+)?(?:\w+\s+)*\w+)\s*\(([^)]*)\)', code)

        if not match:
            return 0

        params_str = match.group(1).strip()
        if not params_str:
            return 0

        # Count parameters, handling default values and type annotations
        params = self._split_params(params_str)

        # Filter out 'self' and 'cls' in Python
        if language == "python":
            params = [p for p in params if p.strip() not in ('self', 'cls')]

        return len(params)

    def _split_params(self, params_str: str) -> list[str]:
        """Split parameter string handling nested generics."""
        params = []
        current = ""
        depth = 0

        for char in params_str:
            if char in ('[', '<', '('):
                depth += 1
                current += char
            elif char in (']', '>', ')'):
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                if current.strip():
                    params.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            params.append(current.strip())

        return params
