"""
Logging Configuration for RepoMind.

This module provides a centralized, production-grade logging setup that:
- Supports both console and file logging
- Uses structured logging for machine-parseable output
- Provides context-aware logging with request tracking
- Follows best practices for observability

Usage:
    from repomind.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started", repo_name="Actions", file_count=150)

Configuration:
    Set LOG_LEVEL environment variable to control verbosity:
    - DEBUG: Detailed debugging information
    - INFO: General operational messages (default)
    - WARNING: Warning messages for potentially harmful situations
    - ERROR: Error messages for serious problems
    - CRITICAL: Critical errors that may prevent operation
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler


# ============================================================================
# Constants
# ============================================================================

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = Path.home() / ".repomind" / "logs"


# ============================================================================
# Custom Formatter for Structured Logging
# ============================================================================

class StructuredFormatter(logging.Formatter):
    """
    A formatter that produces structured, easily-parseable log output.

    Includes:
    - Timestamp in ISO format
    - Log level
    - Logger name (usually module path)
    - Message with structured key-value pairs

    Example output:
        2026-01-24 10:30:45 | INFO     | repomind.tools.index_repo |
        Indexing started | repo_name=Actions | file_count=150
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with structured metadata."""
        # Base formatting
        base_message = super().format(record)

        # Add any extra fields as key=value pairs
        extra_fields = []
        for key, value in record.__dict__.items():
            if key not in (
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'asctime'
            ):
                extra_fields.append(f"{key}={value}")

        if extra_fields:
            return f"{base_message} | {' | '.join(extra_fields)}"
        return base_message


# ============================================================================
# Logger Factory
# ============================================================================

_loggers_initialized = False
_file_handler: Optional[logging.FileHandler] = None


def setup_logging(
    log_level: Optional[str] = None,
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
    use_rich_console: bool = True,
) -> None:
    """
    Configure the logging system for RepoMind.

    This should be called once at application startup. Subsequent calls
    will be ignored to prevent duplicate handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   Defaults to LOG_LEVEL environment variable or INFO.
        log_to_file: Whether to write logs to a file.
        log_dir: Directory for log files. Defaults to ~/.repomind/logs/
        use_rich_console: Use Rich library for colorful console output.

    Example:
        from repomind.logging import setup_logging

        # Basic setup
        setup_logging()

        # Debug mode with custom log directory
        setup_logging(log_level="DEBUG", log_dir=Path("/var/log/repomind"))
    """
    global _loggers_initialized, _file_handler

    if _loggers_initialized:
        return

    # Determine log level
    level_str = log_level or os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Configure root logger for our package
    root_logger = logging.getLogger("repomind")
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove any existing handlers

    # Console handler
    if use_rich_console:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(StructuredFormatter(LOG_FORMAT, LOG_DATE_FORMAT))

    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_directory = log_dir or DEFAULT_LOG_DIR
        log_directory.mkdir(parents=True, exist_ok=True)

        log_file = log_directory / f"repomind-{datetime.now():%Y-%m-%d}.log"
        _file_handler = logging.FileHandler(log_file, encoding="utf-8")
        _file_handler.setFormatter(StructuredFormatter(LOG_FORMAT, LOG_DATE_FORMAT))
        _file_handler.setLevel(level)
        root_logger.addHandler(_file_handler)

    _loggers_initialized = True

    # Log startup message
    root_logger.info(
        "RepoMind logging initialized",
        extra={"log_level": level_str, "log_to_file": log_to_file}
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    This is the primary interface for obtaining loggers throughout
    the application. It ensures consistent configuration and naming.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured Logger instance.

    Example:
        from repomind.logging import get_logger

        logger = get_logger(__name__)

        # Simple message
        logger.info("Processing completed")

        # Message with structured data
        logger.info("Repository indexed", extra={
            "repo_name": "Actions",
            "chunks_created": 150,
            "duration_seconds": 45.2
        })

        # Warning with context
        logger.warning("File skipped", extra={
            "file_path": "test.py",
            "reason": "parse_error"
        })
    """
    # Ensure logging is set up
    if not _loggers_initialized:
        setup_logging()

    return logging.getLogger(name)


# ============================================================================
# Convenience Functions
# ============================================================================

def log_operation_start(
    logger: logging.Logger,
    operation: str,
    **context: Any
) -> datetime:
    """
    Log the start of an operation and return the start time.

    Use with log_operation_end for timing operations.

    Args:
        logger: Logger instance to use.
        operation: Name of the operation being started.
        **context: Additional context to log.

    Returns:
        Start timestamp for duration calculation.

    Example:
        start_time = log_operation_start(logger, "indexing", repo="Actions")
        # ... do work ...
        log_operation_end(logger, "indexing", start_time, chunks=150)
    """
    logger.info(f"{operation} started", extra=context)
    return datetime.now()


def log_operation_end(
    logger: logging.Logger,
    operation: str,
    start_time: datetime,
    success: bool = True,
    **context: Any
) -> float:
    """
    Log the end of an operation with duration.

    Args:
        logger: Logger instance to use.
        operation: Name of the operation that completed.
        start_time: Timestamp from log_operation_start.
        success: Whether the operation succeeded.
        **context: Additional context to log.

    Returns:
        Duration in seconds.

    Example:
        start_time = log_operation_start(logger, "indexing", repo="Actions")
        try:
            # ... do work ...
            duration = log_operation_end(logger, "indexing", start_time, chunks=150)
        except Exception as e:
            log_operation_end(logger, "indexing", start_time, success=False, error=str(e))
            raise
    """
    duration = (datetime.now() - start_time).total_seconds()
    status = "completed" if success else "failed"

    log_method = logger.info if success else logger.error
    log_method(
        f"{operation} {status}",
        extra={"duration_seconds": round(duration, 3), **context}
    )

    return duration


# ============================================================================
# Module Initialization
# ============================================================================

# Auto-setup logging when module is imported
# This ensures logging works even if setup_logging() isn't called explicitly
if not _loggers_initialized:
    setup_logging()
