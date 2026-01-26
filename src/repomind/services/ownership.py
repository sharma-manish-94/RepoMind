"""
Ownership Service - CODEOWNERS and Git Blame Integration.

This module provides code ownership information by analyzing:
1. CODEOWNERS files (GitHub/GitLab format)
2. Git blame data (recent contributors)
3. Commit frequency analysis

Key Features:
- Parse CODEOWNERS files for ownership rules
- Identify reviewers for code changes
- Analyze contributor history
- Suggest experts for specific areas

Example Usage:
    service = OwnershipService("/path/to/repo")

    # Get owners for a file
    owners = service.get_owners("src/services/auth.py")
    # Returns: ["@backend-team", "@security-team"]

    # Suggest reviewers for changes
    reviewers = service.suggest_reviewers(["src/auth.py", "src/user.py"])

Author: RepoMind Team
"""

import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import fnmatch
import logging

logger = logging.getLogger(__name__)


@dataclass
class OwnershipRule:
    """A rule from a CODEOWNERS file."""
    pattern: str
    owners: list[str]
    line_number: int
    file_path: str


@dataclass
class FileOwnership:
    """Ownership information for a file."""
    file_path: str
    codeowners: list[str]  # From CODEOWNERS file
    top_contributors: list[tuple[str, int]]  # (author, commit_count)
    last_modified_by: Optional[str] = None
    last_modified_date: Optional[datetime] = None


@dataclass
class ContributorStats:
    """Statistics for a contributor."""
    name: str
    email: str
    commit_count: int
    files_touched: int
    lines_added: int
    lines_removed: int
    last_commit_date: Optional[datetime] = None
    expertise_areas: list[str] = field(default_factory=list)


@dataclass
class ReviewerSuggestion:
    """A suggested reviewer for a code change."""
    name: str
    reason: str
    confidence: float  # 0-1
    expertise_match: bool
    recent_activity: bool


class OwnershipService:
    """
    Service for determining code ownership and suggesting reviewers.

    Combines CODEOWNERS rules with git history analysis to provide
    comprehensive ownership information.
    """

    # Common CODEOWNERS file locations
    CODEOWNERS_PATHS = [
        "CODEOWNERS",
        ".github/CODEOWNERS",
        ".gitlab/CODEOWNERS",
        "docs/CODEOWNERS",
    ]

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize the ownership service.

        Args:
            repo_path: Path to the git repository.
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self._codeowners_rules: list[OwnershipRule] = []
        self._contributor_cache: dict[str, ContributorStats] = {}

        if self.repo_path:
            self._load_codeowners()

    def _load_codeowners(self):
        """Load and parse CODEOWNERS file."""
        if not self.repo_path:
            return

        for codeowners_path in self.CODEOWNERS_PATHS:
            full_path = self.repo_path / codeowners_path
            if full_path.exists():
                self._parse_codeowners(full_path)
                break

    def _parse_codeowners(self, file_path: Path):
        """
        Parse a CODEOWNERS file.

        Format:
            # Comment
            *.js    @frontend-team
            /src/   @backend-team @lead-dev
            /docs/  docs@example.com
        """
        try:
            content = file_path.read_text()
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse pattern and owners
                parts = line.split()
                if len(parts) >= 2:
                    pattern = parts[0]
                    owners = parts[1:]
                    self._codeowners_rules.append(
                        OwnershipRule(
                            pattern=pattern,
                            owners=owners,
                            line_number=line_num,
                            file_path=str(file_path),
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to parse CODEOWNERS: {e}")

    def get_owners(self, file_path: str) -> list[str]:
        """
        Get owners for a file based on CODEOWNERS rules.

        Rules are matched in order, with later matches taking precedence.

        Args:
            file_path: Path to the file (relative to repo root).

        Returns:
            List of owner identifiers (e.g., @username, email).
        """
        owners = []

        for rule in self._codeowners_rules:
            if self._matches_pattern(file_path, rule.pattern):
                owners = rule.owners  # Later rules override

        return owners

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """
        Check if a file path matches a CODEOWNERS pattern.

        Supports:
        - Glob patterns (*.js, *.py)
        - Directory patterns (/src/, src/)
        - Exact paths
        """
        # Normalize paths
        file_path = file_path.lstrip('/')

        # Handle directory patterns
        if pattern.endswith('/'):
            pattern_dir = pattern.rstrip('/').lstrip('/')
            return file_path.startswith(pattern_dir + '/')

        # Handle leading slash (root-relative)
        if pattern.startswith('/'):
            pattern = pattern[1:]
            return fnmatch.fnmatch(file_path, pattern)

        # Handle glob patterns anywhere in path
        if '*' in pattern:
            # Check if pattern matches file name or full path
            file_name = Path(file_path).name
            return fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(file_path, pattern)

        # Exact match
        return file_path == pattern or file_path.endswith('/' + pattern)

    def get_file_ownership(self, file_path: str) -> FileOwnership:
        """
        Get comprehensive ownership info for a file.

        Combines CODEOWNERS data with git blame analysis.

        Args:
            file_path: Path to the file.

        Returns:
            FileOwnership object with all ownership info.
        """
        # Get CODEOWNERS
        codeowners = self.get_owners(file_path)

        # Get git contributors
        contributors = self._get_file_contributors(file_path)
        top_contributors = [(name, count) for name, count in contributors.most_common(5)]

        # Get last modifier
        last_modified = self._get_last_modifier(file_path)

        return FileOwnership(
            file_path=file_path,
            codeowners=codeowners,
            top_contributors=top_contributors,
            last_modified_by=last_modified.get("author") if last_modified else None,
            last_modified_date=last_modified.get("date") if last_modified else None,
        )

    def _get_file_contributors(self, file_path: str) -> Counter:
        """Get contributor counts from git blame."""
        if not self.repo_path:
            return Counter()

        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return Counter()

            result = subprocess.run(
                ["git", "blame", "--porcelain", file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return Counter()

            # Parse blame output
            contributors = Counter()
            for line in result.stdout.split('\n'):
                if line.startswith('author '):
                    author = line[7:].strip()
                    contributors[author] += 1

            return contributors

        except Exception as e:
            logger.warning(f"Failed to get git blame: {e}")
            return Counter()

    def _get_last_modifier(self, file_path: str) -> Optional[dict]:
        """Get the last person to modify a file."""
        if not self.repo_path:
            return None

        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%an|%ae|%ai", "--", file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0 or not result.stdout.strip():
                return None

            parts = result.stdout.strip().split('|')
            if len(parts) >= 3:
                return {
                    "author": parts[0],
                    "email": parts[1],
                    "date": datetime.fromisoformat(parts[2].split()[0]),
                }

        except Exception as e:
            logger.warning(f"Failed to get last modifier: {e}")

        return None

    def suggest_reviewers(
        self,
        changed_files: list[str],
        exclude_author: Optional[str] = None,
        max_reviewers: int = 3,
    ) -> list[ReviewerSuggestion]:
        """
        Suggest reviewers for a set of changed files.

        Considers:
        1. CODEOWNERS rules
        2. Recent contributors to affected files
        3. Expertise in affected areas

        Args:
            changed_files: List of changed file paths.
            exclude_author: Author to exclude (e.g., PR creator).
            max_reviewers: Maximum number of reviewers to suggest.

        Returns:
            List of reviewer suggestions with confidence scores.
        """
        suggestions: dict[str, ReviewerSuggestion] = {}

        for file_path in changed_files:
            # Get CODEOWNERS
            owners = self.get_owners(file_path)
            for owner in owners:
                if owner != exclude_author:
                    if owner not in suggestions:
                        suggestions[owner] = ReviewerSuggestion(
                            name=owner,
                            reason=f"CODEOWNERS: {file_path}",
                            confidence=0.9,
                            expertise_match=True,
                            recent_activity=True,
                        )
                    else:
                        suggestions[owner].confidence = min(
                            suggestions[owner].confidence + 0.1, 1.0
                        )

            # Get recent contributors
            contributors = self._get_file_contributors(file_path)
            for author, count in contributors.most_common(3):
                if author != exclude_author and author not in suggestions:
                    suggestions[author] = ReviewerSuggestion(
                        name=author,
                        reason=f"Contributed {count} lines to {file_path}",
                        confidence=min(0.5 + count * 0.05, 0.85),
                        expertise_match=False,
                        recent_activity=True,
                    )

        # Sort by confidence and return top N
        sorted_suggestions = sorted(
            suggestions.values(),
            key=lambda s: s.confidence,
            reverse=True
        )
        return sorted_suggestions[:max_reviewers]

    def get_contributor_stats(
        self,
        since_days: int = 90,
    ) -> list[ContributorStats]:
        """
        Get statistics for all contributors.

        Args:
            since_days: Number of days to look back.

        Returns:
            List of contributor statistics.
        """
        if not self.repo_path:
            return []

        try:
            since_date = datetime.now() - timedelta(days=since_days)
            since_str = since_date.strftime("%Y-%m-%d")

            result = subprocess.run(
                [
                    "git", "shortlog", "-sne", "--all",
                    f"--since={since_str}",
                ],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return []

            stats = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue

                # Parse: "  123\tAuthor Name <email@example.com>"
                match = re.match(r'\s*(\d+)\t(.+?)\s*<(.+?)>', line)
                if match:
                    count = int(match.group(1))
                    name = match.group(2).strip()
                    email = match.group(3)

                    stats.append(ContributorStats(
                        name=name,
                        email=email,
                        commit_count=count,
                        files_touched=0,  # Would need additional queries
                        lines_added=0,
                        lines_removed=0,
                    ))

            return sorted(stats, key=lambda s: s.commit_count, reverse=True)

        except Exception as e:
            logger.warning(f"Failed to get contributor stats: {e}")
            return []

    def get_ownership_summary(
        self,
        file_paths: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Get a summary of code ownership.

        Args:
            file_paths: Optional list of files to analyze.
                       If None, analyzes all CODEOWNERS rules.

        Returns:
            Summary dictionary with ownership statistics.
        """
        summary = {
            "codeowners_rules": len(self._codeowners_rules),
            "ownership_by_team": defaultdict(list),
            "unowned_patterns": [],
        }

        # Group rules by owner
        for rule in self._codeowners_rules:
            for owner in rule.owners:
                summary["ownership_by_team"][owner].append(rule.pattern)

        # Convert defaultdict to regular dict
        summary["ownership_by_team"] = dict(summary["ownership_by_team"])

        # Get contributor stats
        contributors = self.get_contributor_stats(since_days=30)
        summary["top_contributors"] = [
            {"name": c.name, "commits": c.commit_count}
            for c in contributors[:10]
        ]

        # If specific files provided, get their ownership
        if file_paths:
            summary["file_ownership"] = {}
            for file_path in file_paths[:20]:  # Limit for performance
                owners = self.get_owners(file_path)
                summary["file_ownership"][file_path] = owners if owners else ["unowned"]

        return summary
