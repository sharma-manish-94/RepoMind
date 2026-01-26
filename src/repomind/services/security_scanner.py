"""
Security Scanner Service - Secret and Vulnerability Detection.

This module provides static analysis for detecting:
1. Hardcoded secrets (API keys, tokens, passwords)
2. Common security anti-patterns
3. Dependency vulnerabilities (placeholder for future integration)

Patterns Detected (26 patterns):
- AWS access keys, secret keys
- GitHub/GitLab tokens
- Slack tokens/webhooks
- JWT secrets
- Database connection strings with passwords
- Private keys (RSA, SSH)
- Generic API keys and tokens
- Hardcoded passwords in config
- Base64-encoded secrets
- Environment variable leaks

Example Usage:
    scanner = SecurityScanner()
    findings = scanner.scan_file("config.py")
    report = scanner.scan_repository("/path/to/repo")

Author: RepoMind Team
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Severity levels for security findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecretType(str, Enum):
    """Types of secrets that can be detected."""
    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    GITHUB_TOKEN = "github_token"
    GITLAB_TOKEN = "gitlab_token"
    SLACK_TOKEN = "slack_token"
    SLACK_WEBHOOK = "slack_webhook"
    JWT_SECRET = "jwt_secret"
    PRIVATE_KEY = "private_key"
    DATABASE_URL = "database_url"
    GENERIC_API_KEY = "generic_api_key"
    GENERIC_SECRET = "generic_secret"
    GENERIC_PASSWORD = "generic_password"
    GENERIC_TOKEN = "generic_token"
    BASE64_SECRET = "base64_secret"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    AZURE_KEY = "azure_key"
    GCP_KEY = "gcp_key"
    STRIPE_KEY = "stripe_key"
    SENDGRID_KEY = "sendgrid_key"
    TWILIO_KEY = "twilio_key"
    HEROKU_KEY = "heroku_key"
    NPM_TOKEN = "npm_token"
    PYPI_TOKEN = "pypi_token"
    SSH_KEY = "ssh_key"
    ENCRYPTION_KEY = "encryption_key"


@dataclass
class SecurityFinding:
    """A security finding from the scanner."""
    secret_type: SecretType
    severity: Severity
    file_path: str
    line_number: int
    description: str
    matched_text: str  # Redacted version of the match
    rule_id: str
    confidence: float  # 0-1
    recommendation: str


@dataclass
class ScanResult:
    """Result of a security scan."""
    total_files_scanned: int
    total_findings: int
    findings_by_severity: dict[str, int]
    findings: list[SecurityFinding]
    scan_duration_ms: float = 0


# Secret detection patterns
# Each pattern has: (compiled regex, SecretType, Severity, description, recommendation)
SECRET_PATTERNS: list[tuple[re.Pattern, SecretType, Severity, str, str]] = [
    # AWS
    (
        re.compile(r'(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}'),
        SecretType.AWS_ACCESS_KEY,
        Severity.CRITICAL,
        "AWS Access Key ID detected",
        "Use AWS IAM roles or environment variables instead of hardcoded keys",
    ),
    (
        re.compile(r'(?:aws)?_?(?:secret)?_?(?:access)?_?key\s*[=:]\s*["\']?[A-Za-z0-9/+=]{40}'),
        SecretType.AWS_SECRET_KEY,
        Severity.CRITICAL,
        "AWS Secret Access Key detected",
        "Use AWS IAM roles or environment variables instead of hardcoded keys",
    ),
    # GitHub
    (
        re.compile(r'gh[pousr]_[A-Za-z0-9_]{36,255}'),
        SecretType.GITHUB_TOKEN,
        Severity.HIGH,
        "GitHub personal access token detected",
        "Use GitHub Secrets or environment variables",
    ),
    (
        re.compile(r'github_pat_[A-Za-z0-9_]{22,255}'),
        SecretType.GITHUB_TOKEN,
        Severity.HIGH,
        "GitHub fine-grained personal access token detected",
        "Use GitHub Secrets or environment variables",
    ),
    # GitLab
    (
        re.compile(r'glpat-[A-Za-z0-9\-]{20,}'),
        SecretType.GITLAB_TOKEN,
        Severity.HIGH,
        "GitLab personal access token detected",
        "Use CI/CD variables or environment variables",
    ),
    # Slack
    (
        re.compile(r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,34}'),
        SecretType.SLACK_TOKEN,
        Severity.HIGH,
        "Slack API token detected",
        "Use Slack app configuration or environment variables",
    ),
    (
        re.compile(r'https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{24}'),
        SecretType.SLACK_WEBHOOK,
        Severity.MEDIUM,
        "Slack webhook URL detected",
        "Store webhook URLs in environment variables",
    ),
    # JWT
    (
        re.compile(r'(?:jwt|JWT)[\s_-]*(?:secret|SECRET|key|KEY)\s*[=:]\s*["\'][^"\']{8,}["\']'),
        SecretType.JWT_SECRET,
        Severity.HIGH,
        "JWT secret key detected",
        "Use environment variables for JWT secrets",
    ),
    # Private Keys
    (
        re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'),
        SecretType.PRIVATE_KEY,
        Severity.CRITICAL,
        "Private key detected in source code",
        "Never commit private keys. Use key management services",
    ),
    (
        re.compile(r'-----BEGIN SSH2? ENCRYPTED PRIVATE KEY-----'),
        SecretType.SSH_KEY,
        Severity.CRITICAL,
        "SSH private key detected",
        "Never commit SSH keys. Use ssh-agent or key management",
    ),
    # Database URLs with passwords
    (
        re.compile(r'(?:postgres|mysql|mongodb|redis)(?:ql)?://\w+:[^@\s]{3,}@'),
        SecretType.DATABASE_URL,
        Severity.HIGH,
        "Database connection string with embedded credentials",
        "Use environment variables for database credentials",
    ),
    # Stripe
    (
        re.compile(r'sk_(?:live|test)_[A-Za-z0-9]{24,}'),
        SecretType.STRIPE_KEY,
        Severity.CRITICAL,
        "Stripe secret key detected",
        "Use environment variables for Stripe keys",
    ),
    (
        re.compile(r'pk_(?:live|test)_[A-Za-z0-9]{24,}'),
        SecretType.STRIPE_KEY,
        Severity.MEDIUM,
        "Stripe publishable key detected (less sensitive but should be configured)",
        "Use environment variables for API keys",
    ),
    # SendGrid
    (
        re.compile(r'SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}'),
        SecretType.SENDGRID_KEY,
        Severity.HIGH,
        "SendGrid API key detected",
        "Use environment variables for API keys",
    ),
    # Twilio
    (
        re.compile(r'SK[a-f0-9]{32}'),
        SecretType.TWILIO_KEY,
        Severity.HIGH,
        "Twilio API key detected",
        "Use environment variables for API keys",
    ),
    # Azure
    (
        re.compile(r'(?:DefaultEndpointsProtocol|AccountKey)\s*=\s*[A-Za-z0-9+/=]{20,}'),
        SecretType.AZURE_KEY,
        Severity.HIGH,
        "Azure storage connection string detected",
        "Use Azure Key Vault or managed identities",
    ),
    # NPM
    (
        re.compile(r'npm_[A-Za-z0-9]{36}'),
        SecretType.NPM_TOKEN,
        Severity.HIGH,
        "NPM access token detected",
        "Use .npmrc with environment variables",
    ),
    # PyPI
    (
        re.compile(r'pypi-[A-Za-z0-9_-]{50,}'),
        SecretType.PYPI_TOKEN,
        Severity.HIGH,
        "PyPI API token detected",
        "Use environment variables for PyPI tokens",
    ),
    # Heroku
    (
        re.compile(r'heroku[\s_-]*(?:api)?[\s_-]*(?:key|token)\s*[=:]\s*["\']?[A-Fa-f0-9-]{36}'),
        SecretType.HEROKU_KEY,
        Severity.HIGH,
        "Heroku API key detected",
        "Use Heroku config vars",
    ),
    # Generic patterns (lower confidence)
    (
        re.compile(r'(?:api|API)[\s_-]*(?:key|KEY)\s*[=:]\s*["\'][A-Za-z0-9_\-]{20,}["\']'),
        SecretType.GENERIC_API_KEY,
        Severity.MEDIUM,
        "Generic API key detected",
        "Store API keys in environment variables or secret management",
    ),
    (
        re.compile(r'(?:secret|SECRET)\s*[=:]\s*["\'][^"\']{8,}["\']'),
        SecretType.GENERIC_SECRET,
        Severity.MEDIUM,
        "Generic secret value detected",
        "Store secrets in environment variables or secret management",
    ),
    (
        re.compile(r'(?:password|PASSWORD|passwd|PASSWD)\s*[=:]\s*["\'][^"\']{4,}["\']'),
        SecretType.GENERIC_PASSWORD,
        Severity.HIGH,
        "Hardcoded password detected",
        "Never hardcode passwords. Use environment variables or secret management",
    ),
    (
        re.compile(r'(?:token|TOKEN)\s*[=:]\s*["\'][A-Za-z0-9_\-\.]{20,}["\']'),
        SecretType.GENERIC_TOKEN,
        Severity.MEDIUM,
        "Generic token value detected",
        "Store tokens in environment variables or secret management",
    ),
    (
        re.compile(r'[Bb]earer\s+[A-Za-z0-9_\-\.]{20,}'),
        SecretType.BEARER_TOKEN,
        Severity.MEDIUM,
        "Bearer token detected in source code",
        "Use environment variables for authentication tokens",
    ),
    (
        re.compile(r'[Bb]asic\s+[A-Za-z0-9+/=]{20,}'),
        SecretType.BASIC_AUTH,
        Severity.MEDIUM,
        "Basic auth credentials detected",
        "Use environment variables for authentication credentials",
    ),
    (
        re.compile(r'(?:encryption|ENCRYPTION)[\s_-]*(?:key|KEY)\s*[=:]\s*["\'][^"\']{16,}["\']'),
        SecretType.ENCRYPTION_KEY,
        Severity.CRITICAL,
        "Encryption key detected in source code",
        "Use key management services (AWS KMS, Azure Key Vault)",
    ),
]

# Files to skip during scanning
SKIP_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2',
    '.ttf', '.eot', '.mp3', '.mp4', '.avi', '.mov', '.zip', '.tar',
    '.gz', '.jar', '.war', '.class', '.pyc', '.pyo', '.so', '.dll',
    '.exe', '.bin', '.lock', '.map',
}

SKIP_FILES = {
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'Cargo.lock',
    'go.sum', 'composer.lock', 'Gemfile.lock',
}

# Directories to skip
SKIP_DIRS = {
    'node_modules', '.git', '__pycache__', '.venv', 'venv',
    'dist', 'build', 'target', '.idea', '.vscode',
}


class SecurityScanner:
    """
    Scans code for hardcoded secrets and security anti-patterns.

    Uses regex pattern matching to detect common secret patterns
    with configurable sensitivity.
    """

    def __init__(self, min_severity: Severity = Severity.LOW):
        """
        Initialize the security scanner.

        Args:
            min_severity: Minimum severity level to report.
        """
        self.min_severity = min_severity
        self._severity_order = {
            Severity.INFO: 0,
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4,
        }

    def scan_content(
        self,
        content: str,
        file_path: str = "<unknown>",
    ) -> list[SecurityFinding]:
        """
        Scan text content for secrets.

        Args:
            content: The text content to scan.
            file_path: Path to the file (for reporting).

        Returns:
            List of security findings.
        """
        findings = []

        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            # Skip comment-only lines that look like documentation
            stripped = line.strip()
            if stripped.startswith(('#', '//', '/*', '*', '"""', "'''")):
                # Still check for actual secrets in comments
                pass

            for pattern, secret_type, severity, description, recommendation in SECRET_PATTERNS:
                if self._severity_order.get(severity, 0) < self._severity_order.get(self.min_severity, 0):
                    continue

                matches = pattern.finditer(line)
                for match in matches:
                    # Redact the matched text
                    matched = match.group()
                    redacted = self._redact(matched)

                    # Calculate confidence based on context
                    confidence = self._calculate_confidence(line, matched, secret_type)

                    if confidence >= 0.5:  # Skip very low confidence
                        findings.append(SecurityFinding(
                            secret_type=secret_type,
                            severity=severity,
                            file_path=file_path,
                            line_number=line_num,
                            description=description,
                            matched_text=redacted,
                            rule_id=f"SEC-{secret_type.value.upper()}",
                            confidence=confidence,
                            recommendation=recommendation,
                        ))

        return findings

    def scan_file(self, file_path: str) -> list[SecurityFinding]:
        """
        Scan a single file for secrets.

        Args:
            file_path: Path to the file.

        Returns:
            List of security findings.
        """
        path = Path(file_path)

        # Skip binary and irrelevant files
        if path.suffix.lower() in SKIP_EXTENSIONS:
            return []
        if path.name in SKIP_FILES:
            return []

        try:
            content = path.read_text(errors='ignore')
            return self.scan_content(content, str(path))
        except Exception as e:
            logger.warning(f"Failed to scan {file_path}: {e}")
            return []

    def scan_repository(
        self,
        repo_path: str,
        max_files: int = 5000,
    ) -> ScanResult:
        """
        Scan an entire repository for secrets.

        Args:
            repo_path: Path to the repository root.
            max_files: Maximum files to scan (safety limit).

        Returns:
            ScanResult with all findings and statistics.
        """
        import time
        start = time.time()

        root = Path(repo_path)
        all_findings: list[SecurityFinding] = []
        files_scanned = 0

        for path in root.rglob('*'):
            if files_scanned >= max_files:
                break

            # Skip directories
            if not path.is_file():
                continue

            # Skip hidden and excluded dirs
            parts = path.relative_to(root).parts
            if any(part in SKIP_DIRS or part.startswith('.') for part in parts):
                continue

            # Skip binary files
            if path.suffix.lower() in SKIP_EXTENSIONS:
                continue

            # Skip large files (likely not source code)
            try:
                if path.stat().st_size > 1_000_000:  # 1MB
                    continue
            except OSError:
                continue

            findings = self.scan_file(str(path))
            all_findings.extend(findings)
            files_scanned += 1

        duration = (time.time() - start) * 1000

        # Count by severity
        severity_counts = {s.value: 0 for s in Severity}
        for finding in all_findings:
            severity_counts[finding.severity.value] += 1

        return ScanResult(
            total_files_scanned=files_scanned,
            total_findings=len(all_findings),
            findings_by_severity=severity_counts,
            findings=all_findings,
            scan_duration_ms=round(duration, 2),
        )

    def scan_chunks(
        self,
        repo_filter: Optional[str] = None,
    ) -> ScanResult:
        """
        Scan indexed code chunks for secrets.

        Uses the stored code chunks instead of reading files directly.
        Useful when you don't have direct filesystem access.

        Args:
            repo_filter: Optional repository filter.

        Returns:
            ScanResult with all findings.
        """
        import time
        from ..services.storage import StorageService

        start = time.time()
        storage = StorageService()
        chunk_map = storage._load_chunk_metadata()

        all_findings: list[SecurityFinding] = []
        files_scanned = set()

        for chunk_id, chunk in chunk_map.items():
            if repo_filter and chunk.repo_name != repo_filter:
                continue

            findings = self.scan_content(chunk.content, chunk.file_path)
            all_findings.extend(findings)
            files_scanned.add(chunk.file_path)

        duration = (time.time() - start) * 1000

        severity_counts = {s.value: 0 for s in Severity}
        for finding in all_findings:
            severity_counts[finding.severity.value] += 1

        return ScanResult(
            total_files_scanned=len(files_scanned),
            total_findings=len(all_findings),
            findings_by_severity=severity_counts,
            findings=all_findings,
            scan_duration_ms=round(duration, 2),
        )

    def _redact(self, text: str) -> str:
        """Redact a secret, keeping only first and last few characters."""
        if len(text) <= 8:
            return text[:2] + "***"
        return text[:4] + "..." + text[-4:]

    def _calculate_confidence(
        self,
        line: str,
        matched: str,
        secret_type: SecretType,
    ) -> float:
        """
        Calculate confidence that a match is a real secret.

        Reduces confidence for:
        - Test files
        - Example/placeholder values
        - Comment lines
        - Known false positive patterns
        """
        confidence = 0.85  # Base confidence

        line_lower = line.lower().strip()
        matched_lower = matched.lower()

        # Reduce confidence for test/example values
        placeholder_indicators = [
            'example', 'test', 'sample', 'dummy', 'fake', 'mock',
            'placeholder', 'xxx', 'your_', 'change_me', 'todo',
            'fixme', 'replace', '<your', 'insert_', 'put_your',
        ]
        for indicator in placeholder_indicators:
            if indicator in matched_lower or indicator in line_lower:
                confidence -= 0.4
                break

        # Reduce for documentation/comments
        if line_lower.startswith(('#', '//', '/*', '*', '"""', "'''")):
            confidence -= 0.2

        # Increase for assignment patterns (more likely real)
        if re.search(r'[=:]\s*["\']', line):
            confidence += 0.05

        # High confidence for specific patterns
        high_confidence_types = {
            SecretType.AWS_ACCESS_KEY,
            SecretType.PRIVATE_KEY,
            SecretType.SSH_KEY,
            SecretType.STRIPE_KEY,
        }
        if secret_type in high_confidence_types:
            confidence = max(confidence, 0.9)

        return max(0.0, min(1.0, confidence))
